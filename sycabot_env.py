import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SycaBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        num_robots=2,
        num_tasks=2,
        max_steps=1000,
        fire_spread_prob=0.020,
        fire_kill_prob=0.5,
        fire_cell_size=0.08,
        robot_radius=0.08,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.dt = 0.2
        self.num_robots = int(num_robots)
        self.num_tasks = int(num_tasks)
        self.max_steps = int(max_steps)

        # Motion/action limits
        self.v_min, self.v_max = -0.20, 0.20
        self.w_min, self.w_max = -np.pi / 6.0, np.pi / 6.0

        # Workspace
        self.x_min, self.x_max = -1.55, 1.55
        self.y_min, self.y_max = -3.10, 3.10
        self.robot_radius = float(robot_radius)
        self.collision_distance = 2.0 * self.robot_radius

        # Hazard parameters
        self.fire_spread_prob = float(fire_spread_prob)
        self.fire_kill_prob = float(fire_kill_prob)
        self.fire_cell_size = float(fire_cell_size)

        # Environment geometry
        self.obstacles = self._add_obstacles()
        self.exits = self._add_exits()
        self.grid_shape = self._grid_shape_from_bounds()

        # Action is joint action [v1,w1,v2,w2,...]
        action_low = np.tile(np.array([self.v_min, self.w_min], dtype=np.float32), self.num_robots)
        action_high = np.tile(np.array([self.v_max, self.w_max], dtype=np.float32), self.num_robots)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation = robot block + task block + fire grid + global indicators
        robot_feature_dim = 8
        task_feature_dim = 4
        obs_dim = (
            self.num_robots * robot_feature_dim
            + self.num_tasks * task_feature_dim
            + int(np.prod(self.grid_shape))
            + 2
        )
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32) * 10.0,
            high=np.ones(obs_dim, dtype=np.float32) * 10.0,
            dtype=np.float32,
        )

        # Runtime state
        self.robot_states = np.zeros((self.num_robots, 3), dtype=np.float32)  # x, y, theta
        self.robot_alive = np.ones(self.num_robots, dtype=np.float32)
        self.robot_carrying = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_safety = np.ones(self.num_robots, dtype=np.float32)
        self.robot_departed_exit = np.zeros(self.num_robots, dtype=np.float32)

        self.tasks = np.zeros((self.num_tasks, 2), dtype=np.float32)
        self.task_status = np.zeros(self.num_tasks, dtype=np.int32)  # 0 pending, 1 carried, 2 delivered, 3 contaminated
        self.task_carrier = -np.ones(self.num_tasks, dtype=np.int32)

        self.fire_grid = np.zeros(self.grid_shape, dtype=np.int8)

        self.global_safety_indicator = 1.0
        self.global_task_indicator = 0.0  # 0 none, 1 picked, 2 delivered
        self.prev_task_dist = 0.0
        self.prev_exit_dist = 0.0

        self.step_count = 0

        # Render
        self.window = None
        self.clock = None
        self.screen_size = 800

    def _add_obstacles(self):
        return [
            [[-1.498, 3.001], [0.001, 3.000]],
            [[1.051, 3.001], [1.494, 3.000]],
            [[1.494, 3.000], [1.493, 0.430]],
            [[1.494, -0.374], [1.497, -2.998]],
            [[0.002, -2.999], [1.497, -2.998]],
            [[-1.498, -2.999], [-1.047, -2.999]],
            [[-1.496, -0.500], [-1.498, -2.999]],
            [[-1.496, 0.750], [-1.495, 0.299]],
            [[-1.498, 2.998], [-1.498, 1.553]],
            [[-0.481, 2.382], [0.879, 1.356]],
            [[-1.498, 1.553], [-0.700, 1.551]],
            [[1.018, 0.429], [1.493, 0.430]],
            [[0.141, 1.040], [-0.269, 0.524]],
            [[-1.496, 0.526], [-0.269, 0.524]],
            [[-0.269, 0.524], [-0.261, -0.008]],
            [[-0.261, -0.008], [0.480, -0.008]],
            [[0.011, -0.008], [0.011, -0.486]],
            [[-1.496, -0.859], [-0.492, -0.860]],
            [[0.922, -0.613], [0.924, -2.093]],
            [[0.260, -1.084], [0.260, -2.093]],
            [[-0.665, -2.094], [0.924, -2.093]],
            [[-0.685, -2.103], [-0.931, -2.414]],
        ]

    def _add_exits(self):
        return np.array(
            [
                [-1.497, 1.1515],
                [-1.4945, -0.1005],
                [-0.524, -2.999],
                [1.4955, 0.028],
                [0.526, 3.001],
            ],
            dtype=np.float32,
        )

    def _grid_shape_from_bounds(self):
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        gx = int(np.ceil(width / self.fire_cell_size))
        gy = int(np.ceil(height / self.fire_cell_size))
        return (gx, gy)

    def _world_to_grid(self, x, y):
        gx = int(np.clip((x - self.x_min) / self.fire_cell_size, 0, self.grid_shape[0] - 1))
        gy = int(np.clip((y - self.y_min) / self.fire_cell_size, 0, self.grid_shape[1] - 1))
        return gx, gy

    def _grid_to_world_center(self, gx, gy):
        x = self.x_min + (gx + 0.5) * self.fire_cell_size
        y = self.y_min + (gy + 0.5) * self.fire_cell_size
        return np.array([x, y], dtype=np.float32)

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        p = np.array(point, dtype=np.float32)
        a = np.array(seg_start, dtype=np.float32)
        b = np.array(seg_end, dtype=np.float32)
        ab = b - a
        denom = np.dot(ab, ab)
        if denom <= 1e-9:
            return float(np.linalg.norm(p - a))
        t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _is_obstacle_collision(self, point):
        for segment in self.obstacles:
            if self._point_to_segment_distance(point, segment[0], segment[1]) < self.robot_radius:
                return True
        return False

    def _min_obstacle_distance(self, point):
        d = []
        for segment in self.obstacles:
            d.append(self._point_to_segment_distance(point, segment[0], segment[1]))
        return float(np.min(d)) if d else 10.0

    def _is_out_of_boundary(self, point):
        x, y = point
        return not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max)

    def _sample_free_point(self, margin=0.15):
        for _ in range(2000):
            x = self.np_random.uniform(self.x_min + margin, self.x_max - margin)
            y = self.np_random.uniform(self.y_min + margin, self.y_max - margin)
            if not self._is_obstacle_collision((x, y)):
                return np.array([x, y], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

    def _sample_robot_starts_from_exits(self):
        states = np.zeros((self.num_robots, 3), dtype=np.float32)
        if self.num_robots <= len(self.exits):
            exit_indices = self.np_random.permutation(len(self.exits))[: self.num_robots]
        else:
            exit_indices = self.np_random.integers(0, len(self.exits), size=self.num_robots)

        for i in range(self.num_robots):
            idx = int(exit_indices[i])
            center = self.exits[idx].copy()
            best = center.copy()
            best_sep = -1.0

            for _ in range(150):
                jitter = self.np_random.uniform(low=-0.12, high=0.12, size=2)
                cand = center + jitter
                if self._is_out_of_boundary(cand) or self._is_obstacle_collision(cand):
                    continue
                if self._min_obstacle_distance(cand) < self.robot_radius * 2.0:
                    continue
                if i == 0:
                    best = cand
                    break
                d = np.linalg.norm(states[:i, :2] - cand, axis=1)
                min_sep = float(np.min(d))
                if min_sep > best_sep:
                    best_sep = min_sep
                    best = cand
                if min_sep > self.collision_distance * 1.4:
                    break

            states[i, :2] = best
            states[i, 2] = self.np_random.uniform(-np.pi, np.pi)
        return states

    def _nearest_exit_distance(self, point):
        d = np.linalg.norm(self.exits - point, axis=1)
        return float(np.min(d))

    def _nearest_task_distance(self, point):
        pending_idx = np.where(self.task_status == 0)[0]
        if len(pending_idx) == 0:
            return 0.0
        d = np.linalg.norm(self.tasks[pending_idx] - point, axis=1)
        return float(np.min(d))

    def _nearest_fire_distance(self, point):
        burning = np.argwhere(self.fire_grid > 0)
        if len(burning) == 0:
            return 10.0
        fire_points = np.array([self._grid_to_world_center(gx, gy) for gx, gy in burning], dtype=np.float32)
        d = np.linalg.norm(fire_points - point, axis=1)
        return float(np.min(d))

    def _spawn_fire(self):
        for _ in range(3000):
            p = self._sample_free_point(margin=0.1)
            gx, gy = self._world_to_grid(p[0], p[1])
            if self.fire_grid[gx, gy] == 0:
                self.fire_grid[gx, gy] = 1
                return

    def _propagate_fire(self):
        new_fire = self.fire_grid.copy()
        active = np.argwhere(self.fire_grid > 0)
        if len(active) == 0:
            return

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for gx, gy in active:
            for dx, dy in neighbors:
                nx, ny = gx + dx, gy + dy
                if nx < 0 or ny < 0 or nx >= self.grid_shape[0] or ny >= self.grid_shape[1]:
                    continue
                if new_fire[nx, ny] > 0:
                    continue
                c = self._grid_to_world_center(nx, ny)
                if self._is_obstacle_collision(c):
                    continue
                if self.np_random.random() < self.fire_spread_prob:
                    new_fire[nx, ny] = 1

        self.fire_grid = new_fire

    def _update_task_contamination(self):
        for i in range(self.num_tasks):
            if self.task_status[i] != 0:
                continue
            gx, gy = self._world_to_grid(self.tasks[i, 0], self.tasks[i, 1])
            if self.fire_grid[gx, gy] > 0:
                self.task_status[i] = 3

    def _apply_robot_motion(self, actions):
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                continue
            v = float(np.clip(actions[2 * i], self.v_min, self.v_max))
            w = float(np.clip(actions[2 * i + 1], self.w_min, self.w_max))

            x, y, theta = self.robot_states[i]
            x_new = x + v * np.cos(theta) * self.dt
            y_new = y + v * np.sin(theta) * self.dt
            theta_new = self.wrap_angle(theta + w * self.dt)

            self.robot_states[i] = np.array([x_new, y_new, theta_new], dtype=np.float32)
            if self._nearest_exit_distance(self.robot_states[i, :2]) > 0.28:
                self.robot_departed_exit[i] = 1.0

    def _check_robot_failures(self):
        safety_events = 0

        # Boundary + obstacle
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                continue
            p = self.robot_states[i, :2]
            if self._is_out_of_boundary(p) or self._is_obstacle_collision(p):
                self.robot_alive[i] = 0.0
                self.robot_safety[i] = 0.0
                safety_events += 1

        # Mutual collisions
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                continue
            for j in range(i + 1, self.num_robots):
                if self.robot_alive[j] < 0.5:
                    continue
                d = np.linalg.norm(self.robot_states[i, :2] - self.robot_states[j, :2])
                if d < self.collision_distance:
                    self.robot_alive[i] = 0.0
                    self.robot_alive[j] = 0.0
                    self.robot_safety[i] = 0.0
                    self.robot_safety[j] = 0.0
                    safety_events += 2

        # Fire kill zone
        fire_radius = 0.5 * np.sqrt(2.0) * self.fire_cell_size + self.robot_radius
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                continue
            gx, gy = self._world_to_grid(self.robot_states[i, 0], self.robot_states[i, 1])
            affected = []
            for nx in range(max(0, gx - 1), min(self.grid_shape[0], gx + 2)):
                for ny in range(max(0, gy - 1), min(self.grid_shape[1], gy + 2)):
                    if self.fire_grid[nx, ny] > 0:
                        affected.append((nx, ny))
            if not affected:
                continue
            rpos = self.robot_states[i, :2]
            close_to_fire = False
            for fx, fy in affected:
                c = self._grid_to_world_center(fx, fy)
                if np.linalg.norm(rpos - c) <= fire_radius:
                    close_to_fire = True
                    break
            if close_to_fire and self.np_random.random() < self.fire_kill_prob:
                self.robot_alive[i] = 0.0
                self.robot_safety[i] = 0.0
                safety_events += 1

        if safety_events > 0:
            self.global_safety_indicator = 0.0

        # Drop carried tasks for destroyed robots (task becomes contaminated)
        for i in range(self.num_robots):
            if self.robot_alive[i] >= 0.5:
                continue
            task_ids = np.where(self.task_carrier == i)[0]
            for tid in task_ids:
                if self.task_status[tid] == 1:
                    self.task_status[tid] = 3
                    self.task_carrier[tid] = -1
                    self.robot_carrying[i] = 0.0

        return safety_events

    def _update_task_logic(self):
        picked_count = 0
        delivered_count = 0

        # Pick up tasks
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5 or self.robot_carrying[i] > 0.5:
                continue
            pending = np.where(self.task_status == 0)[0]
            if len(pending) == 0:
                continue
            d = np.linalg.norm(self.tasks[pending] - self.robot_states[i, :2], axis=1)
            idx = np.argmin(d)
            if d[idx] < 0.18:
                task_id = int(pending[idx])
                self.task_status[task_id] = 1
                self.task_carrier[task_id] = i
                self.robot_carrying[i] = 1.0
                picked_count += 1

        # Tasks carried follow robot position
        carrying_tasks = np.where(self.task_status == 1)[0]
        for tid in carrying_tasks:
            rid = self.task_carrier[tid]
            if rid >= 0:
                self.tasks[tid] = self.robot_states[rid, :2]

        # Deliver tasks at exits
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5 or self.robot_carrying[i] < 0.5:
                continue
            if self._nearest_exit_distance(self.robot_states[i, :2]) < 0.20:
                carried = np.where((self.task_status == 1) & (self.task_carrier == i))[0]
                for tid in carried:
                    self.task_status[tid] = 2
                    self.task_carrier[tid] = -1
                    delivered_count += 1
                self.robot_carrying[i] = 0.0

        if picked_count > 0:
            self.global_task_indicator = max(self.global_task_indicator, 1.0)
        if delivered_count > 0:
            self.global_task_indicator = 2.0

        return picked_count, delivered_count

    def _helper_progress_terms(self):
        task_d = 0.0
        exit_d = 0.0

        # Distances for searching robots to pending tasks
        pending = np.where(self.task_status == 0)[0]
        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                continue
            p = self.robot_states[i, :2]
            if self.robot_carrying[i] < 0.5 and len(pending) > 0:
                task_d += self._nearest_task_distance(p)
            if self.robot_carrying[i] > 0.5:
                exit_d += self._nearest_exit_distance(p)

        progress_task = self.prev_task_dist - task_d
        progress_exit = self.prev_exit_dist - exit_d

        self.prev_task_dist = task_d
        self.prev_exit_dist = exit_d

        return progress_task, progress_exit

    def _spread_reward(self):
        alive_idx = np.where(self.robot_alive > 0.5)[0]
        if len(alive_idx) < 2:
            return 0.0
        positions = self.robot_states[alive_idx, :2]
        pairwise = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pairwise.append(np.linalg.norm(positions[i] - positions[j]))
        mean_dist = float(np.mean(pairwise)) if pairwise else 0.0
        return float(np.clip(mean_dist, 0.0, 1.0))

    def _build_observation(self):
        robot_features = []
        for i in range(self.num_robots):
            x, y, th = self.robot_states[i]
            alive = self.robot_alive[i]
            carrying = self.robot_carrying[i]
            task_d = self._nearest_task_distance(self.robot_states[i, :2])
            exit_d = self._nearest_exit_distance(self.robot_states[i, :2])
            fire_d = self._nearest_fire_distance(self.robot_states[i, :2])
            robot_features.extend([x, y, th, alive, carrying, task_d, exit_d, fire_d])

        task_features = []
        for i in range(self.num_tasks):
            x, y = self.tasks[i]
            status = float(self.task_status[i])
            carried = 1.0 if self.task_status[i] == 1 else 0.0
            task_features.extend([x, y, status, carried])

        fire_flat = self.fire_grid.astype(np.float32).flatten()
        indicators = np.array([self.global_safety_indicator, self.global_task_indicator], dtype=np.float32)

        obs = np.concatenate(
            [
                np.array(robot_features, dtype=np.float32),
                np.array(task_features, dtype=np.float32),
                fire_flat,
                indicators,
            ]
        )
        return obs.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")

        self.step_count += 1

        self._apply_robot_motion(action)
        self._propagate_fire()

        safety_events = self._check_robot_failures()
        self._update_task_contamination()
        picked_count, delivered_count = self._update_task_logic()

        progress_task, progress_exit = self._helper_progress_terms()
        spread = self._spread_reward()

        reward = 0.0
        reward += -1.0 * safety_events
        reward += 1.0 * picked_count
        reward += 2.0 * delivered_count
        reward += 0.25 * progress_task
        reward += 0.35 * progress_exit
        reward += 0.05 * spread
        reward += -0.01

        all_tasks_contaminated = bool(np.all(self.task_status == 3))
        all_robots_destroyed = np.all(self.robot_alive < 0.5)

        d_all = np.array([self._nearest_exit_distance(self.robot_states[i, :2]) for i in range(self.num_robots)])
        all_robots_at_exit = bool(np.all(d_all < 0.20) and np.all(self.robot_departed_exit > 0.5))

        terminated = bool(all_tasks_contaminated or all_robots_destroyed or all_robots_at_exit)
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._build_observation()
        info = {
            "alive_robots": int(np.sum(self.robot_alive > 0.5)),
            "pending_tasks": int(np.sum(self.task_status == 0)),
            "carried_tasks": int(np.sum(self.task_status == 1)),
            "delivered_tasks": int(np.sum(self.task_status == 2)),
            "contaminated_tasks": int(np.sum(self.task_status == 3)),
            "safety_indicator": float(self.global_safety_indicator),
            "task_indicator": float(self.global_task_indicator),
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.robot_states = self._sample_robot_starts_from_exits()
        self.robot_alive = np.ones(self.num_robots, dtype=np.float32)
        self.robot_carrying = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_safety = np.ones(self.num_robots, dtype=np.float32)
        self.robot_departed_exit = np.zeros(self.num_robots, dtype=np.float32)

        self.tasks = np.zeros((self.num_tasks, 2), dtype=np.float32)
        for i in range(self.num_tasks):
            for _ in range(2000):
                p = self._sample_free_point(margin=0.15)
                if np.min(np.linalg.norm(self.exits - p, axis=1)) < 0.35:
                    continue
                if i > 0:
                    d = np.linalg.norm(self.tasks[:i] - p, axis=1)
                    if np.any(d < 0.20):
                        continue
                self.tasks[i] = p
                break

        self.task_status = np.zeros(self.num_tasks, dtype=np.int32)
        self.task_carrier = -np.ones(self.num_tasks, dtype=np.int32)

        self.fire_grid = np.zeros(self.grid_shape, dtype=np.int8)
        self._spawn_fire()

        self.global_safety_indicator = 1.0
        self.global_task_indicator = 0.0

        self.prev_task_dist = 0.0
        self.prev_exit_dist = 0.0
        _ = self._helper_progress_terms()

        return self._build_observation(), {}

    def _to_screen(self, point):
        x, y = point
        sx = int((x - self.x_min) / (self.x_max - self.x_min) * self.screen_size)
        sy = int((self.y_max - y) / (self.y_max - self.y_min) * self.screen_size)
        return sx, sy

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((245, 245, 245))

        # Fire cells
        for gx in range(self.grid_shape[0]):
            for gy in range(self.grid_shape[1]):
                if self.fire_grid[gx, gy] <= 0:
                    continue
                c = self._grid_to_world_center(gx, gy)
                px, py = self._to_screen(c)
                half = int(0.5 * self.fire_cell_size / (self.x_max - self.x_min) * self.screen_size)
                rect = pygame.Rect(px - half, py - half, 2 * half, 2 * half)
                pygame.draw.rect(self.window, (255, 120, 50), rect)

        # Obstacles
        for obstacle in self.obstacles:
            start = self._to_screen(obstacle[0])
            end = self._to_screen(obstacle[1])
            pygame.draw.line(self.window, (20, 20, 20), start, end, 4)

        # Exits
        for p in self.exits:
            pygame.draw.circle(self.window, (40, 180, 40), self._to_screen(p), 8)

        # Tasks
        for i in range(self.num_tasks):
            color = (0, 120, 255)
            if self.task_status[i] == 2:
                color = (0, 200, 80)
            elif self.task_status[i] == 3:
                color = (180, 60, 60)
            pygame.draw.circle(self.window, color, self._to_screen(self.tasks[i]), 6)

        # Robots
        for i in range(self.num_robots):
            x, y, th = self.robot_states[i]
            if self.robot_alive[i] > 0.5:
                color = (50, 50, 220)
                if self.robot_carrying[i] > 0.5:
                    color = (150, 40, 220)
            else:
                color = (90, 90, 90)
            center = self._to_screen((x, y))
            pygame.draw.circle(self.window, color, center, 8)
            head = (int(center[0] + 14 * np.cos(th)), int(center[1] - 14 * np.sin(th)))
            pygame.draw.line(self.window, (255, 255, 255), center, head, 2)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    @staticmethod
    def wrap_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

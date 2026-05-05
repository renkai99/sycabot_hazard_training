import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment_configs import get_lab_environment_config
from sycabot_render import SycaBotRenderer


class SycaBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        num_robots=2,
        num_tasks=2,
        max_steps=1000,
        fire_spread_prob=0.020,
        fire_kill_prob=0.2,
        fire_cell_size=0.08,
        robot_radius=0.08,
        pickup_reward=1000.0,
        delivery_reward=1000.0,
        smooth_action_weight=1.0,
        turn_smooth_weight=0.20,
        jerk_weight=0.08,
        direction_flip_weight=0.10,
        death_penalty=3000.0,
        survival_reward_weight=0.05,
        task_progress_reward_weight=8.0,
        exit_progress_reward_weight=10.0,
        environment_config=None,
        renderer=None,
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

        if environment_config is None:
            environment_config = get_lab_environment_config()

        bounds = environment_config["bounds"]
        self.x_min = float(bounds["x_min"])
        self.x_max = float(bounds["x_max"])
        self.y_min = float(bounds["y_min"])
        self.y_max = float(bounds["y_max"])
        self.robot_radius = float(robot_radius)
        self.collision_distance = 2.0 * self.robot_radius

        # Hazard parameters
        self.fire_spread_prob = float(fire_spread_prob)
        self.fire_kill_prob = float(fire_kill_prob)
        self.fire_cell_size = float(fire_cell_size)

        # Reward parameters
        self.pickup_reward = float(pickup_reward)
        self.delivery_reward = float(delivery_reward)
        self.smooth_action_weight = float(smooth_action_weight)
        self.turn_smooth_weight = float(turn_smooth_weight)
        self.jerk_weight = float(jerk_weight)
        self.direction_flip_weight = float(direction_flip_weight)
        self.death_penalty = float(death_penalty)
        self.survival_reward_weight = float(survival_reward_weight)
        self.task_progress_reward_weight = float(task_progress_reward_weight)
        self.exit_progress_reward_weight = float(exit_progress_reward_weight)

        # Environment geometry
        self.obstacles = environment_config["obstacles"]
        self.exits = np.array(environment_config["exits"], dtype=np.float32)
        self.grid_shape = self._grid_shape_from_bounds()

        # Action space: [v1, w1, v2, w2, ...]
        action_low = np.tile(np.array([self.v_min, self.w_min], dtype=np.float32), self.num_robots)
        action_high = np.tile(np.array([self.v_max, self.w_max], dtype=np.float32), self.num_robots)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation space:
        #   robot block : 22 features × num_robots
        #   task block  :  4 features × num_tasks
        #   global      :  2 scalars (safety_indicator, task_indicator)
        #   action hist :  2 × num_robots (prev_joint_action)
        robot_feature_dim = 22
        task_feature_dim = 4
        obs_dim = (
            self.num_robots * robot_feature_dim
            + self.num_tasks * task_feature_dim
            + 2
            + 2 * self.num_robots
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Runtime state
        self.robot_states = np.zeros((self.num_robots, 3), dtype=np.float32)  # x, y, theta
        self.robot_alive = np.ones(self.num_robots, dtype=np.float32)
        self.robot_carrying = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_departed_exit = np.zeros(self.num_robots, dtype=np.float32)

        self.tasks = np.zeros((self.num_tasks, 2), dtype=np.float32)
        self.task_status = np.zeros(self.num_tasks, dtype=np.int32)  # 0 pending 1 carried 2 delivered 3 contaminated
        self.task_carrier = -np.ones(self.num_tasks, dtype=np.int32)

        self.fire_grid = np.zeros(self.grid_shape, dtype=np.int8)

        self.global_safety_indicator = 1.0
        self.prev_visible_task_dist = np.full(self.num_robots, np.nan, dtype=np.float32)
        self.prev_visible_exit_dist = np.full(self.num_robots, np.nan, dtype=np.float32)
        self.prev_joint_action = np.zeros(2 * self.num_robots, dtype=np.float32)
        self.prev_prev_joint_action = np.zeros(2 * self.num_robots, dtype=np.float32)

        self.step_count = 0

        self.renderer = renderer if renderer is not None else SycaBotRenderer()

    # ------------------------------------------------------------------ #
    #  Grid helpers                                                       #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                   #
    # ------------------------------------------------------------------ #

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

    def _orientation(self, a, b, c):
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def _on_segment(self, a, b, c, eps=1e-6):
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )

    def _segments_intersect(self, p1, p2, q1, q2, eps=1e-6):
        o1 = self._orientation(p1, p2, q1)
        o2 = self._orientation(p1, p2, q2)
        o3 = self._orientation(q1, q2, p1)
        o4 = self._orientation(q1, q2, p2)

        if o1 * o2 < -eps and o3 * o4 < -eps:
            return True
        if abs(o1) <= eps and self._on_segment(p1, p2, q1, eps):
            return True
        if abs(o2) <= eps and self._on_segment(p1, p2, q2, eps):
            return True
        if abs(o3) <= eps and self._on_segment(q1, q2, p1, eps):
            return True
        if abs(o4) <= eps and self._on_segment(q1, q2, p2, eps):
            return True
        return False

    def _has_clear_line_of_sight(self, start_point, end_point):
        start = np.array(start_point, dtype=np.float32)
        end = np.array(end_point, dtype=np.float32)
        for obstacle in self.obstacles:
            if self._segments_intersect(start, end, obstacle[0], obstacle[1]):
                return False
        return True

    def _is_obstacle_collision(self, point):
        for segment in self.obstacles:
            if self._point_to_segment_distance(point, segment[0], segment[1]) < self.robot_radius:
                return True
        return False

    def _min_obstacle_distance(self, point):
        d = [self._point_to_segment_distance(point, seg[0], seg[1]) for seg in self.obstacles]
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

    # ------------------------------------------------------------------ #
    #  Distance queries                                                   #
    # ------------------------------------------------------------------ #

    def _nearest_exit_distance(self, point):
        d = np.linalg.norm(self.exits - point, axis=1)
        return float(np.min(d))

    def _nearest_visible_exit_distance(self, point):
        visible_distances = []
        for exit_point in self.exits:
            if self._has_clear_line_of_sight(point, exit_point):
                visible_distances.append(float(np.linalg.norm(exit_point - point)))
        if not visible_distances:
            return None
        return min(visible_distances)

    def _nearest_task_distance(self, point):
        pending_idx = np.where(self.task_status == 0)[0]
        if len(pending_idx) == 0:
            return 0.0
        d = np.linalg.norm(self.tasks[pending_idx] - point, axis=1)
        return float(np.min(d))

    def _nearest_visible_task_distance(self, point):
        pending_idx = np.where(self.task_status == 0)[0]
        if len(pending_idx) == 0:
            return None
        visible_distances = []
        for idx in pending_idx:
            task_point = self.tasks[idx]
            if self._has_clear_line_of_sight(point, task_point):
                visible_distances.append(float(np.linalg.norm(task_point - point)))
        if not visible_distances:
            return None
        return min(visible_distances)

    def _nearest_fire_distance(self, point):
        burning = np.argwhere(self.fire_grid > 0)
        if len(burning) == 0:
            return 10.0
        fire_points = np.array([self._grid_to_world_center(gx, gy) for gx, gy in burning], dtype=np.float32)
        d = np.linalg.norm(fire_points - point, axis=1)
        return float(np.min(d))

    # ------------------------------------------------------------------ #
    #  Orientation helpers                                                #
    # ------------------------------------------------------------------ #

    def _task_orientation(self, pos, theta):
        """Robot-relative bearing to the nearest pending task (radians, in [-π, π])."""
        pending_idx = np.where(self.task_status == 0)[0]
        if len(pending_idx) == 0:
            return 0.0
        dists = np.linalg.norm(self.tasks[pending_idx] - pos, axis=1)
        nearest = self.tasks[pending_idx[np.argmin(dists)]]
        angle = np.arctan2(nearest[1] - pos[1], nearest[0] - pos[0])
        return float(((angle - theta + np.pi) % (2 * np.pi)) - np.pi)

    def _exit_orientation(self, pos, theta):
        """Robot-relative bearing to the nearest exit (radians, in [-π, π])."""
        nearest = self.exits[np.argmin(np.linalg.norm(self.exits - pos, axis=1))]
        angle = np.arctan2(nearest[1] - pos[1], nearest[0] - pos[0])
        return float(((angle - theta + np.pi) % (2 * np.pi)) - np.pi)

    def _top3_obs_dist_orient(self, pos, theta):
        """Distance and robot-relative bearing to the 3 nearest obstacle segments.

        Distance is point-to-segment. Returns lists of length 3; shorter obstacle
        lists are impossible given the lab map, but the logic is safe regardless.
        """
        dists = []
        closest_pts = []
        for seg in self.obstacles:
            a = np.array(seg[0], dtype=np.float32)
            b = np.array(seg[1], dtype=np.float32)
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = float(np.clip(np.dot(pos - a, ab) / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0
            closest = a + t * ab
            dists.append(float(np.linalg.norm(pos - closest)))
            closest_pts.append(closest)

        idx3 = np.argsort(dists)[:3]
        top_dists, top_orients = [], []
        for idx in idx3:
            vec = closest_pts[idx] - pos
            angle = np.arctan2(float(vec[1]), float(vec[0]))
            top_dists.append(dists[idx])
            top_orients.append(float(((angle - theta + np.pi) % (2 * np.pi)) - np.pi))
        return top_dists, top_orients

    def _top3_fire_dist_orient(self, pos, theta):
        """Edge distance and robot-relative bearing to the 3 nearest burning cells.

        Edge distance = max(0, centre_distance − cell_size/2).
        Slots with no fire behind them are filled with (10.0, 0.0).
        """
        burning = np.argwhere(self.fire_grid > 0)
        if len(burning) == 0:
            return [10.0, 10.0, 10.0], [0.0, 0.0, 0.0]

        fire_centers = np.array(
            [self._grid_to_world_center(gx, gy) for gx, gy in burning], dtype=np.float32
        )
        center_dists = np.linalg.norm(fire_centers - pos, axis=1)
        edge_dists = np.maximum(0.0, center_dists - self.fire_cell_size / 2.0)

        n = min(3, len(edge_dists))
        idx_sorted = np.argsort(edge_dists)

        top_dists, top_orients = [], []
        for i in range(3):
            if i < n:
                vec = fire_centers[idx_sorted[i]] - pos
                angle = np.arctan2(float(vec[1]), float(vec[0]))
                top_dists.append(float(edge_dists[idx_sorted[i]]))
                top_orients.append(float(((angle - theta + np.pi) % (2 * np.pi)) - np.pi))
            else:
                top_dists.append(10.0)
                top_orients.append(0.0)
        return top_dists, top_orients

    # ------------------------------------------------------------------ #
    #  Fire dynamics                                                      #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Task / robot state updates                                         #
    # ------------------------------------------------------------------ #

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
        obstacle_hits = 0
        boundary_deaths = 0
        obstacle_deaths = 0
        mutual_collision_deaths = 0
        fire_deaths = 0

        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                continue
            p = self.robot_states[i, :2]
            if self._is_out_of_boundary(p):
                self.robot_alive[i] = 0.0
                safety_events += 1
                boundary_deaths += 1
                continue
            if self._is_obstacle_collision(p):
                self.robot_alive[i] = 0.0
                obstacle_hits += 1
                safety_events += 1
                obstacle_deaths += 1

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
                    safety_events += 2
                    mutual_collision_deaths += 2

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
            close_to_fire = any(
                np.linalg.norm(rpos - self._grid_to_world_center(fx, fy)) <= fire_radius
                for fx, fy in affected
            )
            if close_to_fire and self.np_random.random() < self.fire_kill_prob:
                self.robot_alive[i] = 0.0
                safety_events += 1
                fire_deaths += 1

        if safety_events > 0:
            self.global_safety_indicator = 0.0

        for i in range(self.num_robots):
            if self.robot_alive[i] >= 0.5:
                continue
            task_ids = np.where(self.task_carrier == i)[0]
            for tid in task_ids:
                if self.task_status[tid] == 1:
                    self.task_status[tid] = 3
                    self.task_carrier[tid] = -1
                    self.robot_carrying[i] = 0.0

        return safety_events, obstacle_hits, boundary_deaths, obstacle_deaths, mutual_collision_deaths, fire_deaths

    def _update_task_logic(self):
        picked_count = 0
        delivered_count = 0

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

        carrying_tasks = np.where(self.task_status == 1)[0]
        for tid in carrying_tasks:
            rid = self.task_carrier[tid]
            if rid >= 0:
                self.tasks[tid] = self.robot_states[rid, :2]

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

        return picked_count, delivered_count

    def _helper_progress_terms(self):
        progress_task = 0.0
        progress_exit = 0.0

        for i in range(self.num_robots):
            if self.robot_alive[i] < 0.5:
                self.prev_visible_task_dist[i] = np.nan
                self.prev_visible_exit_dist[i] = np.nan
                continue

            p = self.robot_states[i, :2]

            if self.robot_carrying[i] < 0.5:
                self.prev_visible_exit_dist[i] = np.nan
                visible_task_distance = self._nearest_visible_task_distance(p)
                if visible_task_distance is not None and np.isfinite(self.prev_visible_task_dist[i]):
                    progress_task += max(float(self.prev_visible_task_dist[i]) - visible_task_distance, 0.0)
                self.prev_visible_task_dist[i] = (
                    np.float32(visible_task_distance) if visible_task_distance is not None else np.nan
                )
            else:
                self.prev_visible_task_dist[i] = np.nan
                visible_exit_distance = self._nearest_visible_exit_distance(p)
                if visible_exit_distance is not None and np.isfinite(self.prev_visible_exit_dist[i]):
                    progress_exit += max(float(self.prev_visible_exit_dist[i]) - visible_exit_distance, 0.0)
                self.prev_visible_exit_dist[i] = (
                    np.float32(visible_exit_distance) if visible_exit_distance is not None else np.nan
                )

        return progress_task, progress_exit

    # ------------------------------------------------------------------ #
    #  Observation                                                        #
    # ------------------------------------------------------------------ #

    def _build_observation(self):
        # Robot block: 22 features per robot
        #   x, y, sin(θ), cos(θ), alive, carrying,
        #   task_d, task_orient, exit_d, exit_orient,
        #   fire1_d, fire1_o, fire2_d, fire2_o, fire3_d, fire3_o,
        #   obs1_d, obs1_o, obs2_d, obs2_o, obs3_d, obs3_o
        robot_features = []
        for i in range(self.num_robots):
            x, y, th = self.robot_states[i]
            p = np.array([x, y], dtype=np.float32)
            obs_d, obs_o   = self._top3_obs_dist_orient(p, th)
            fire_d, fire_o = self._top3_fire_dist_orient(p, th)
            robot_features += [
                x, y,
                float(np.sin(th)), float(np.cos(th)),
                float(self.robot_alive[i]),
                float(self.robot_carrying[i]),
                self._nearest_task_distance(p),
                self._task_orientation(p, th),
                self._nearest_exit_distance(p),
                self._exit_orientation(p, th),
                fire_d[0], fire_o[0],
                fire_d[1], fire_o[1],
                fire_d[2], fire_o[2],
                obs_d[0],  obs_o[0],
                obs_d[1],  obs_o[1],
                obs_d[2],  obs_o[2],
            ]

        # Task block: x, y, status, carried  (4 per task)
        task_features = []
        for i in range(self.num_tasks):
            x, y = self.tasks[i]
            task_features += [
                x,
                y,
                float(self.task_status[i]),
                1.0 if self.task_status[i] == 1 else 0.0,
            ]

        # Global indicators
        any_carried   = bool(np.any(self.task_status == 1))
        any_delivered = bool(np.any(self.task_status == 2))
        task_indicator = 2.0 if any_delivered else (1.0 if any_carried else 0.0)

        obs = np.concatenate([
            np.array(robot_features, dtype=np.float32),
            np.array(task_features, dtype=np.float32),
            np.array([self.global_safety_indicator, task_indicator], dtype=np.float32),
            self.prev_joint_action,  # action history closes the smoothness-penalty observability gap
        ])
        return np.nan_to_num(obs.astype(np.float32), nan=0.0)

    # ------------------------------------------------------------------ #
    #  Gymnax API                                                         #
    # ------------------------------------------------------------------ #

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        self.step_count += 1

        self._apply_robot_motion(clipped_action)
        self._propagate_fire()

        (
            safety_events,
            obstacle_hits,
            boundary_deaths,
            obstacle_deaths,
            mutual_collision_deaths,
            fire_deaths,
        ) = self._check_robot_failures()
        self._update_task_contamination()
        picked_count, delivered_count = self._update_task_logic()
        progress_task, progress_exit = self._helper_progress_terms()

        # Smoothness penalties
        v_curr = clipped_action[0::2]
        w_curr = clipped_action[1::2]
        v_prev = self.prev_joint_action[0::2]
        w_prev = self.prev_joint_action[1::2]
        dv = v_curr - v_prev
        dw = w_curr - w_prev
        action_smooth_penalty   = self.smooth_action_weight  * float(np.mean(dv * dv))
        turn_smooth_penalty     = self.turn_smooth_weight     * float(np.mean(dw * dw))
        jerk = clipped_action - 2.0 * self.prev_joint_action + self.prev_prev_joint_action
        jerk_penalty            = self.jerk_weight            * float(np.mean(jerk * jerk))
        flips = np.logical_and(v_curr * v_prev < 0.0, np.minimum(np.abs(v_curr), np.abs(v_prev)) > 0.03)
        direction_flip_penalty  = self.direction_flip_weight  * float(np.sum(flips))
        smooth_penalty = -(action_smooth_penalty + turn_smooth_penalty + jerk_penalty + direction_flip_penalty)

        # Reward components
        pickup_reward_component   = self.pickup_reward               * picked_count
        delivery_reward_component = self.delivery_reward             * delivered_count
        progress_reward_component = (
            self.task_progress_reward_weight * progress_task
            + self.exit_progress_reward_weight * progress_exit
        )
        death_penalty_component   = self.death_penalty               * safety_events
        survival_reward_component = self.survival_reward_weight      * float(np.sum(self.robot_alive))

        total_robot_failures = boundary_deaths + obstacle_deaths + mutual_collision_deaths + fire_deaths
        all_robot_failure = total_robot_failures >= self.num_robots

        reward = (
            -death_penalty_component
            + pickup_reward_component
            + delivery_reward_component
            + progress_reward_component
            + smooth_penalty
            + survival_reward_component
            - 0.01
        )

        if all_robot_failure:
            reward = -200.0
            pickup_reward_component   = 0.0
            delivery_reward_component = 0.0
            progress_reward_component = 0.0
            smooth_penalty            = 0.0
            survival_reward_component = 0.0

        self.prev_prev_joint_action = self.prev_joint_action.copy()
        self.prev_joint_action = clipped_action.copy()

        all_robots_destroyed = bool(np.all(self.robot_alive < 0.5))
        no_active_tasks = bool(np.all((self.task_status == 2) | (self.task_status == 3)))

        terminated = bool(all_robots_destroyed or no_active_tasks)
        truncated  = bool(self.step_count >= self.max_steps)

        obs = self._build_observation()
        info = {
            "alive_robots":             int(np.sum(self.robot_alive > 0.5)),
            "pending_tasks":            int(np.sum(self.task_status == 0)),
            "carried_tasks":            int(np.sum(self.task_status == 1)),
            "delivered_tasks":          int(np.sum(self.task_status == 2)),
            "contaminated_tasks":       int(np.sum(self.task_status == 3)),
            "obstacle_hits":            int(obstacle_hits),
            "boundary_deaths":          int(boundary_deaths),
            "obstacle_deaths":          int(obstacle_deaths),
            "mutual_collision_deaths":  int(mutual_collision_deaths),
            "fire_deaths":              int(fire_deaths),
            "smooth_penalty":           float(-smooth_penalty),
            "reward_progress":          float(progress_reward_component),
            "reward_pickup":            float(pickup_reward_component),
            "reward_delivery":          float(delivery_reward_component),
            "reward_survival":          float(survival_reward_component),
            "death_penalty":            float(death_penalty_component),
            "reward_override_all_robot_failure": bool(all_robot_failure),
            "safety_indicator":         float(self.global_safety_indicator),
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.robot_states = self._sample_robot_starts_from_exits()
        self.robot_alive = np.ones(self.num_robots, dtype=np.float32)
        self.robot_carrying = np.zeros(self.num_robots, dtype=np.float32)
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

        self.prev_visible_task_dist = np.full(self.num_robots, np.nan, dtype=np.float32)
        self.prev_visible_exit_dist = np.full(self.num_robots, np.nan, dtype=np.float32)
        self.prev_joint_action = np.zeros(2 * self.num_robots, dtype=np.float32)
        self.prev_prev_joint_action = np.zeros(2 * self.num_robots, dtype=np.float32)
        _ = self._helper_progress_terms()

        return self._build_observation(), {}

    def render(self):
        if self.render_mode != "human":
            return
        self.renderer.render(self)

    def close(self):
        self.renderer.close()

    @staticmethod
    def wrap_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

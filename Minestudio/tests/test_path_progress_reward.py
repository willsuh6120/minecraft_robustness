import unittest

from minestudio.tutorials.inference.evaluate_rocket.path_progress_reward import (
    build_path_progress_reward_config,
    build_zone_label_lookup,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    INTERACTION_BENCHMARK_TASKS,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_posttrain_rewards import (
    InteractionPostTrainRewardTracker,
)


class PathProgressRewardTest(unittest.TestCase):
    def test_left_lane_z1_t6_excludes_dead_right_branch(self):
        reward_config = build_path_progress_reward_config(
            target_local=(0, 6),
            obstacle_positions=((0, 1), (1, 1), (1, 2), (1, 3)),
            arena_settings={
                "funnel_start_z": 4,
                "half_width": 4,
                "back_z": -3,
            },
        )

        self.assertIsNotNone(reward_config)

        zone_lookup = build_zone_label_lookup(reward_config)
        for row_z in (1, 2, 3):
            self.assertNotIn((2, row_z), zone_lookup)
            self.assertNotIn((3, row_z), zone_lookup)
            self.assertNotIn((4, row_z), zone_lookup)

        self.assertEqual(zone_lookup[(-1, 1)], "1")
        self.assertEqual(zone_lookup[(-1, 2)], "2")
        self.assertEqual(zone_lookup[(0, 3)], "3")
        self.assertEqual(zone_lookup[(-1, 4)], "A")
        self.assertEqual(zone_lookup[(0, 4)], "A")
        self.assertEqual(zone_lookup[(1, 4)], "A")

    def test_right_lane_z1_t6_excludes_blocked_left_branch(self):
        reward_config = build_path_progress_reward_config(
            target_local=(0, 6),
            obstacle_positions=((0, 1), (-1, 1), (-1, 2), (-1, 3)),
            arena_settings={
                "funnel_start_z": 4,
                "half_width": 4,
                "back_z": -3,
            },
        )
        zone_lookup = build_zone_label_lookup(reward_config)
        for row_z in (1, 2, 3):
            self.assertNotIn((-2, row_z), zone_lookup)
        self.assertEqual(zone_lookup[(1, 1)], "1")
        self.assertEqual(zone_lookup[(2, 1)], "1")

    def test_left_outer_detour_t6_excludes_dead_end_center_step(self):
        reward_config = build_path_progress_reward_config(
            target_local=(0, 6),
            obstacle_positions=((-1, 1), (0, 1), (1, 1), (1, 2), (0, 3), (1, 4)),
            arena_settings={
                "funnel_start_z": 4,
                "half_width": 4,
                "back_z": -3,
            },
        )
        zone_lookup = build_zone_label_lookup(reward_config)
        self.assertNotIn((0, 2), zone_lookup)
        self.assertEqual(zone_lookup[(-2, 1)], "1")
        self.assertEqual(zone_lookup[(-1, 2)], "2")
        self.assertEqual(zone_lookup[(0, 4)], "A")

    def test_left_chicane_t6_row2_outer_cell_is_pruned(self):
        reward_config = build_path_progress_reward_config(
            target_local=(0, 6),
            obstacle_positions=((0, 1), (1, 1), (-1, 2), (1, 3)),
            arena_settings={
                "funnel_start_z": 4,
                "half_width": 4,
                "back_z": -3,
            },
        )
        zone_lookup = build_zone_label_lookup(reward_config)
        self.assertNotIn((2, 2), zone_lookup)
        self.assertEqual(zone_lookup[(1, 2)], "2")
        self.assertEqual(zone_lookup[(0, 2)], "2")

    def test_right_chicane_t6_row2_outer_cell_is_pruned(self):
        reward_config = build_path_progress_reward_config(
            target_local=(0, 6),
            obstacle_positions=((0, 1), (-1, 1), (1, 2), (-1, 3)),
            arena_settings={
                "funnel_start_z": 4,
                "half_width": 4,
                "back_z": -3,
            },
        )
        zone_lookup = build_zone_label_lookup(reward_config)
        self.assertNotIn((-2, 2), zone_lookup)
        self.assertEqual(zone_lookup[(-1, 2)], "2")
        self.assertEqual(zone_lookup[(0, 2)], "2")

    def test_left_s_curve_t6_removes_left_branch_but_keeps_goal_row_slide(self):
        reward_config = build_path_progress_reward_config(
            target_local=(0, 6),
            obstacle_positions=((0, 1), (1, 1), (1, 2), (0, 3), (-1, 4)),
            arena_settings={
                "funnel_start_z": 4,
                "half_width": 4,
                "back_z": -3,
            },
        )
        zone_lookup = build_zone_label_lookup(reward_config)
        for row_z in (1, 2, 3):
            self.assertNotIn((-2, row_z), zone_lookup)
            self.assertNotIn((-1, row_z), zone_lookup)
        self.assertEqual(zone_lookup[(2, 1)], "1")
        self.assertEqual(zone_lookup[(0, 4)], "A")
        self.assertEqual(zone_lookup[(1, 4)], "A")

    def test_posttrain_tracker_adds_env_reward(self):
        tracker = InteractionPostTrainRewardTracker(
            INTERACTION_BENCHMARK_TASKS["mine_coal"],
            initial_info={},
        )
        event = tracker.evaluate({}, env_reward=0.25)
        self.assertAlmostEqual(event.reward, 0.25)
        self.assertFalse(event.success)
        self.assertFalse(event.failure)


if __name__ == "__main__":
    unittest.main()

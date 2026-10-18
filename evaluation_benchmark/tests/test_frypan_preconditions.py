"""Food must be in the frypan before Tasks8/9/15 can earn pour stages."""
from __future__ import annotations

from types import SimpleNamespace as NS
import unittest

import numpy as np

from test_task22_completion import LOOPS, STAGE_FILES, load_nodes, run_loop, stage_namespace


TASKS = {
    8: ("chocolate_pudding_1", "Pudding", "Tomato_Sauce", 0.10),
    9: ("butter_1", "Butter", "Tomato_Sauce", 0.10),
    15: ("butter_1", "Butter", "Milk", 0.12),
}


def stage_names(task_id):
    _, food_label, source_label, _ = TASKS[task_id]
    return [f"01_Place_{food_label}_Frypan", f"02_Lift_{source_label}",
            "03_Pour_One", "04_Pour_Two"]


class FrypanPreconditionTest(unittest.TestCase):
    def test_stage_maps_reuse_existing_geometry_without_final_placement(self):
        for path in STAGE_FILES:
            for task_id, (food, _, _, xy_limit) in TASKS.items():
                with self.subTest(source=path.name, task=task_id):
                    stages = stage_namespace(path)
                    specs = stages["_task_specs"](task_id)
                    self.assertEqual([s.name for s in specs], stage_names(task_id))
                    stages["_in_container_body"].assert_called_once_with(
                        food, "frypan_1", xy_limit, -0.05, 0.15,
                    )
                    done = dict.fromkeys(stage_names(task_id), True)
                    self.assertTrue(stages["_stage_success_from_stage_done"](task_id, done))
                    self.assertEqual(stages["_stage_score_pct"](task_id, done), 100.0)
                    done[stage_names(task_id)[0]] = False
                    self.assertFalse(stages["_stage_success_from_stage_done"](task_id, done))
                    self.assertEqual(stages["_stage_score_pct"](task_id, done), 75.0)

    def test_existing_body_predicate_rejects_food_outside_pan_or_missing(self):
        for path in STAGE_FILES:
            namespace = load_nodes(path, {"_in_container_body"}, {
                "np": np, "_current_body_pos": lambda env, name: env.positions.get(name),
            })
            for task_id, (food, _, _, xy_limit) in TASKS.items():
                check = namespace["_in_container_body"](food, "frypan_1", xy_limit, -0.05, 0.15)
                for offset, expected in (
                    ([0.0, 0.0, 0.05], True),
                    ([xy_limit - 0.001, 0.0, 0.05], True),
                    ([xy_limit + 0.001, 0.0, 0.05], False),
                    ([0.0, 0.0, 0.151], False),
                    ([0.0, 0.0, -0.051], False),
                    (None, False),
                ):
                    with self.subTest(source=path.name, task=task_id, offset=offset):
                        pan = np.array([0.2, 0.1, 0.5])
                        env = NS(positions={"frypan_1": pan})
                        if offset is not None:
                            env.positions[food] = pan + np.asarray(offset)
                        self.assertEqual(check(env, {}, 0), expected)

    def test_all_runners_cannot_skip_food_in_pan(self):
        for key in LOOPS:
            for task_id in TASKS:
                with self.subTest(loop=key, task=task_id):
                    env, (score, done, goal, diag, _, _) = run_loop(
                        key, dict.fromkeys(stage_names(task_id)[1:], 1),
                        task_id=task_id, max_steps=12,
                    )
                    self.assertEqual(env.steps, 12)
                    self.assertEqual(score, 0.0)
                    self.assertFalse(any(done.values()))
                    self.assertFalse(goal)
                    self.assertFalse(diag["stage_success"])
                    self.assertIsNone(diag["pour_2_step"])

    def test_all_runners_succeed_without_final_bottle_placement(self):
        for key in LOOPS:
            for task_id in TASKS:
                with self.subTest(loop=key, task=task_id):
                    env, (score, done, goal, diag, _, _) = run_loop(
                        key, dict(zip(stage_names(task_id), range(1, 5))), task_id=task_id,
                    )
                    self.assertEqual(env.steps, 6)
                    self.assertEqual(score, 100.0)
                    self.assertEqual(list(done), stage_names(task_id))
                    self.assertTrue(all(done.values()))
                    self.assertTrue(goal)
                    self.assertTrue(diag["stage_success"])
                    self.assertIsNone(diag["failure_reason"])

    def test_task10_remains_three_stages_without_final_placement(self):
        for path in STAGE_FILES:
            with self.subTest(source=path.name):
                stages = stage_namespace(path)
                self.assertEqual([s.name for s in stages["_task_specs"](10)], [
                    "01_Lift_Wine_Bottle", "02_Pour_One", "03_Pour_Two",
                ])
                stages["_in_container_body"].assert_not_called()


if __name__ == "__main__":
    unittest.main()

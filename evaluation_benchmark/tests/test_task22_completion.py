"""CPU regressions for Task22's required stages and extra-pour monitoring.

Execute the real AST functions without importing simulator or model runtimes.
Only physical predicates, observations, and policy/planner calls are mocked.
"""
from __future__ import annotations

import ast
from collections import deque
import logging
from pathlib import Path
import queue
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, call

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE_FILES = (
    ROOT / "scripts/task2_26_reference_stage.py",
    ROOT / "openpi_minimal_runtime/retry_tasks2_26_stage_from_anygrasp.py",
)
LOOPS = {
    "generic": (ROOT / "scripts/eval_tasks2_26.py", "run_episode_with_stateful_stages"),
    "async": (
        ROOT / "async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py",
        "run_episode_async_stateful",
    ),
    "reference": (
        ROOT / "reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py",
        "run_episode_async_stateful",
    ),
}
NAMES = (
    "01_Lift_Tomato_Sauce",
    "02_Pour_One",
    "03_Pour_Two",
    "04_Place_Tomato_Aside",
    "05_Open_Microwave",
    "06_Place_Cookies_Microwave",
    "07_Close_Microwave",
)
STAGE_NODES = {
    "DRAWER_TASK_OPTIONAL_FINAL_STAGE", "COUNTING_POUR_TASKS",
    "_is_drawer_task", "_optional_final_stage_name", "_counted_stage_names",
    "_stage_success_from_stage_done", "_stage_score_pct", "_is_counting_pour_task",
    "_counting_pour_stages", "_task_specs", "_goal_override_check",
    "_near_fixed_position", "_microwave_open", "_microwave_closed", "_in_microwave",
}


def load_nodes(path, names, namespace):
    """Compile unmodified top-level functions/constants, excluding heavy imports."""
    nodes = []
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            matched = {t.id for t in node.targets if isinstance(t, ast.Name)} & names
            if matched:
                nodes.append(node)
                found.update(matched)
    if found != names:
        raise AssertionError(f"Missing AST definitions in {path}: {names - found}")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def stage_namespace(path):
    names = STAGE_NODES
    if path == STAGE_FILES[1]:
        names = names - {"COUNTING_POUR_TASKS", "_is_counting_pour_task"}
    namespace = load_nodes(path, names, {
        "StageSpec": lambda name, check_fn: NS(name=name, check_fn=check_fn), "np": np,
    })
    for name in (
        "_lift_rel", "_counted_pour_stage", "_body_pour_stage", "_near_fixed_position",
        "_microwave_open", "_microwave_closed", "_in_microwave",
    ):
        namespace[name] = Mock(return_value=lambda env, state, start: False)
    return namespace


class RolloutEnv:
    def __init__(self, completion_steps, third_pour_at=None):
        self.completion_steps = completion_steps
        self.third_pour_at = third_pour_at
        self.steps = 0
        self.monitor_observations = []

    def reset(self):
        self.steps = 0
        self.monitor_observations.clear()
        return {}

    def step(self, action):
        self.steps += 1
        return {}, 0.0, False, {}


def run_loop(key, completion_steps, *, max_steps=20, monitor_steps=2, third_pour_at=None):
    env = RolloutEnv(completion_steps, third_pour_at)
    stages = stage_namespace(STAGE_FILES[0])
    specs = stages["_task_specs"](22)
    for spec in specs:
        spec.check_fn = lambda env, state, start, name=spec.name: (
            name in env.completion_steps and env.steps >= env.completion_steps[name]
        )

    def update_state(obs, state):
        state["step_idx"] += 1

    def extra_pour(env, state, start):
        env.monitor_observations.append(env.steps)
        return env.third_pour_at is not None and env.steps >= env.third_pour_at

    stages.update(
        _build_initial_state=lambda env: {"step_idx": 0},
        _update_state=update_state,
        _extra_pour_check=lambda task_id: extra_pour,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    logger = Mock(spec=logging.Logger)
    namespace = {
        **stages, "stage_eval": NS(**stages), "deque": deque, "np": np,
        "threading": threading, "queue": queue, "logging": logger,
        "ec": NS(LIBERO_DUMMY_ACTION=[0.0] * 7),
        "build_eval26_policy_input": lambda **kw: ({}, frame, frame),
        "ensure_action_chunk": lambda actions: actions,
        "obs_to_pi_element": lambda *a, **kw: {
            "observation/image": frame, "observation/wrist_image": frame,
        },
        "_extract_vlm_frame": lambda *a: (frame, frame),
    }
    path, function_name = LOOPS[key]
    function = load_nodes(path, {function_name}, namespace)[function_name]
    actions = np.zeros((1, 7), dtype=np.float32)
    adapter = NS(infer_actions=lambda **kw: actions, infer=lambda element: {"actions": actions})
    common = dict(
        task_id=22, env=env, stage_specs=specs, goal_monitor_dict={},
        goal_check_override=None, fail_on_extra_pour=True,
        extra_pour_monitor_steps=monitor_steps, post_goal_steps=0,
    )
    if key == "generic":
        result = function(
            **common, adapter=adapter, prompt="Task22", resize_size=2,
            replan_steps=1, num_steps_wait=0, max_steps=max_steps,
        )
    else:
        result = function(
            **common, client=adapter,
            planner=NS(default_subtask_prompt="Task22", infer_sync=lambda *a: "Task22"),
            args=NS(n_recent=0, async_vlm=False, vlm_queue_size=1, vlm_interval=1,
                    max_steps=max_steps, num_steps_wait=0, resize_size=2, replan_steps=1),
            vlm_camera_pose=None, logger=logger,
        )
    # The production loops catch exceptions; do not let a harness error look like
    # an expected incomplete episode.
    logger.exception.assert_not_called()
    return env, result


class Task22CompletionTest(unittest.TestCase):
    def test_canonical_and_legacy_seven_stages_six_required(self):
        for path in STAGE_FILES:
            with self.subTest(source=path.name):
                stages = stage_namespace(path)
                self.assertEqual([s.name for s in stages["_task_specs"](22)], list(NAMES))
                self.assertEqual(stages["_optional_final_stage_name"](22), NAMES[-1])
                done = dict.fromkeys(NAMES, False)
                self.assertEqual(stages["_counted_stage_names"](22, done), list(NAMES[:6]))
                if "_is_counting_pour_task" in stages:
                    self.assertTrue(stages["_is_counting_pour_task"](22))
                success = stages["_stage_success_from_stage_done"]
                score = stages["_stage_score_pct"]
                for name in NAMES[:3]:
                    done[name] = True
                self.assertFalse(success(22, done))
                self.assertEqual(score(22, done), 50.0)
                done.update(dict.fromkeys(NAMES[:6], True))
                self.assertTrue(success(22, done))
                self.assertTrue(stages["_goal_override_check"](22)(None, done))
                self.assertEqual(score(22, done), 100.0)
                done[NAMES[-1]] = True
                for name in NAMES[:6]:
                    with self.subTest(missing=name):
                        incomplete = {**done, name: False}
                        self.assertFalse(success(22, incomplete))
                        self.assertAlmostEqual(score(22, incomplete), 100.0 * 5 / 6)

    def test_stage_factories_reuse_historical_targets_and_thresholds(self):
        for path in STAGE_FILES:
            with self.subTest(source=path.name):
                stages = stage_namespace(path)
                stages["_task_specs"](22)
                stages["_lift_rel"].assert_called_once_with("tomato_sauce_1", 0.03)
                if path == STAGE_FILES[0]:
                    self.assertEqual(stages["_counted_pour_stage"].call_args_list, [
                        call("tomato_sauce_1", "body", "cookies_1", 0.20, 1),
                        call("tomato_sauce_1", "body", "cookies_1", 0.20, 2),
                    ])
                else:
                    self.assertEqual(stages["_body_pour_stage"].call_args_list, [
                        call("tomato_sauce_1", "body", "cookies_1", 0.20, 0.15, 0.10),
                    ] * 2)
                aside = stages["_near_fixed_position"]
                aside.assert_called_once()
                obj, target, xy_tol, z_tol = aside.call_args.args
                self.assertEqual(obj, "tomato_sauce_1")
                np.testing.assert_allclose(target, [0.0, -0.2, 0.5])
                self.assertEqual((xy_tol, z_tol), (0.20, 0.20))
                stages["_microwave_open"].assert_called_once_with(0.30)
                stages["_in_microwave"].assert_called_once_with("cookies_1")
                stages["_microwave_closed"].assert_called_once_with(0.05)

    def test_aside_predicate_uses_fixed_target_not_initial_position(self):
        for path in STAGE_FILES:
            namespace = load_nodes(path, {"_near_fixed_position"}, {
                "np": np, "_current_body_pos": lambda env, name: env.pos,
            })
            check = namespace["_near_fixed_position"](
                "tomato_sauce_1", np.array([0.0, -0.2, 0.5]), 0.20, 0.20,
            )
            initial = np.array([1.0, 1.0, 0.5])
            for pos, expected in (
                ([0.0, -0.2, 0.5], True), ([0.19, -0.2, 0.69], True),
                ([0.21, -0.2, 0.5], False), ([0.0, -0.2, 0.71], False),
                (initial, False), (None, False),
            ):
                with self.subTest(source=path.name, position=pos):
                    env = NS(pos=None if pos is None else np.asarray(pos))
                    self.assertEqual(check(env, {"initial_body_pos": {"tomato_sauce_1": initial}}, 0), expected)

    def test_pour_only_is_not_success_and_does_not_stop_at_deadline(self):
        for key in LOOPS:
            with self.subTest(loop=key):
                env, (score, done, goal, diag, main, wrist) = run_loop(
                    key, dict(zip(NAMES[:3], (1, 2, 3))), max_steps=12,
                )
                self.assertEqual(score, 50.0)
                self.assertFalse(goal)
                self.assertFalse(diag["stage_success"])
                self.assertEqual(diag["failure_reason"], "incomplete_stage")
                self.assertFalse(diag["extra_pour_detected"])
                self.assertTrue(all(done[n] for n in NAMES[:3]))
                self.assertFalse(any(done[n] for n in NAMES[3:]))
                self.assertEqual(env.steps, 12)
                self.assertEqual(diag["extra_monitor_end_step"], diag["pour_2_step"] + 2)

    def test_all_required_without_close_succeeds_and_exits_before_budget(self):
        for key in LOOPS:
            with self.subTest(loop=key):
                env, (score, done, goal, diag, main, wrist) = run_loop(
                    key, dict(zip(NAMES[:6], range(1, 7))),
                )
                self.assertEqual(score, 100.0)
                self.assertTrue(goal)
                self.assertTrue(diag["stage_success"])
                self.assertIsNone(diag["failure_reason"])
                self.assertTrue(all(done[n] for n in NAMES[:6]))
                self.assertFalse(done[NAMES[-1]])
                self.assertEqual(env.steps, 6)
                self.assertEqual((len(main), len(wrist)), (6, 6))

    def test_required_stages_still_wait_for_monitor_deadline(self):
        for key in LOOPS:
            with self.subTest(loop=key):
                env, (_, done, goal, diag, _, _) = run_loop(
                    key, dict(zip(NAMES[:6], range(1, 7))), monitor_steps=5,
                )
                self.assertTrue(goal)
                self.assertTrue(diag["stage_success"])
                self.assertIsNone(diag["failure_reason"])
                self.assertFalse(done[NAMES[-1]])
                self.assertEqual(env.steps, 8)
                self.assertEqual(diag["extra_monitor_end_step"], diag["pour_2_step"] + 5)

    def test_late_third_pour_after_deadline_before_cookies_fails(self):
        for key in LOOPS:
            with self.subTest(loop=key):
                steps = dict(zip(NAMES[:6], (1, 2, 3, 4, 5, 9)))
                env, (score, done, goal, diag, _, _) = run_loop(key, steps, third_pour_at=7)
                self.assertEqual(env.steps, 7)
                self.assertIn(7, env.monitor_observations)
                observed_t = env.steps - 1 if key == "generic" else env.steps
                self.assertGreater(observed_t, diag["extra_monitor_end_step"])
                self.assertAlmostEqual(score, 100.0 * 5 / 6)
                self.assertFalse(done[NAMES[5]])
                self.assertFalse(goal)
                self.assertFalse(diag["stage_success"])
                self.assertTrue(diag["extra_pour_detected"])
                self.assertEqual(diag["failure_reason"], "extra_pour")

    def test_monitor_budget_boundary_uses_last_observation_not_next_step(self):
        for key in LOOPS:
            for budget, expected_success in ((6, False), (7, True)):
                with self.subTest(loop=key, budget=budget):
                    env, (score, done, goal, diag, _, _) = run_loop(
                        key, dict(zip(NAMES[:6], range(1, 7))),
                        max_steps=budget, monitor_steps=4,
                    )
                    self.assertEqual(score, 100.0)
                    self.assertTrue(all(done[n] for n in NAMES[:6]))
                    self.assertFalse(done[NAMES[-1]])
                    self.assertEqual(env.steps, budget)
                    self.assertEqual(goal, expected_success)
                    self.assertEqual(diag["stage_success"], expected_success)
                    self.assertFalse(diag["extra_pour_detected"])
                    self.assertEqual(
                        diag["failure_reason"], None if expected_success else "monitor_incomplete",
                    )
                    self.assertEqual(
                        diag["extra_monitor_end_step"],
                        diag["pour_2_step"] + 4 if expected_success else None,
                    )


if __name__ == "__main__":
    unittest.main()

"""Task1 accepts either basket, but both objects must share that basket."""
from collections import deque
import logging
from pathlib import Path
import queue
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from test_task22_completion import LOOPS, load_nodes


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts/task2_26_reference_stage.py"
NAMES = ["01_Place_Cookies_Basket", "02_Place_Tomato_Basket"]


def stages():
    return load_nodes(SCORER, {
        "_in_container_body", "_task1_same_basket", "_any_check", "_task_specs",
        "_goal_override_check", "_stage_score_pct", "_stage_success_from_stage_done",
        "_counted_stage_names", "_optional_final_stage_name", "_is_drawer_task",
        "DRAWER_TASK_OPTIONAL_FINAL_STAGE", "COUNTING_POUR_TASKS", "_is_counting_pour_task",
    }, {
        "np": np, "StageSpec": lambda name, check_fn: NS(name=name, check_fn=check_fn),
        "_current_body_pos": lambda env, name: env.positions.get(name),
    })


def scene(cookies=1, tomato=1):
    positions = {"basket_1": np.array([0.0, 0.0, 0.5]),
                 "basket_2": np.array([0.4, 0.0, 0.5])}
    for obj, basket in (("cookies_1", cookies), ("tomato_sauce_1", tomato)):
        if basket is not None:
            positions[obj] = positions[f"basket_{basket}"] + [0.0, 0.0, 0.05]
    return NS(positions=positions)


def run_positions(key, tomato_basket):
    namespace = stages()
    env = scene(2, None)
    initial = {k: v.copy() for k, v in env.positions.items()}
    def reset():
        env.positions = {k: v.copy() for k, v in initial.items()}
        env.steps = 0
        return {}
    def step(action):
        env.steps += 1
        if env.steps >= 2:
            env.positions["tomato_sauce_1"] = env.positions[f"basket_{tomato_basket}"] + [0.0, 0.0, 0.05]
        return {}, 0.0, False, {}
    env.reset, env.step = reset, step
    namespace.update(
        _build_initial_state=lambda env: {"step_idx": 0},
        _update_state=lambda obs, state: state.update(step_idx=state["step_idx"] + 1),
        _extra_pour_check=lambda task_id: None,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    logger = Mock(spec=logging.Logger)
    namespace.update(
        stage_eval=NS(**namespace), deque=deque, threading=threading, queue=queue,
        logging=logger, ec=NS(LIBERO_DUMMY_ACTION=[0.0] * 7),
        build_eval26_policy_input=lambda **kw: ({}, frame, frame),
        ensure_action_chunk=lambda actions: actions,
        obs_to_pi_element=lambda *a, **kw: {"observation/image": frame, "observation/wrist_image": frame},
        _extract_vlm_frame=lambda *a: (frame, frame),
    )
    actions = np.zeros((1, 7), dtype=np.float32)
    policy = NS(infer_actions=lambda **kw: actions, infer=lambda element: {"actions": actions})
    if key == "public":
        load_nodes(ROOT / "scripts/eval_task1_only.py", {"TASK_ID", "STAGE_CHECKS"}, namespace)
        namespace["LIBERO_DUMMY_ACTION"] = [0.0] * 7
        function = load_nodes(ROOT / "scripts/eval_common.py", {"run_episode_with_stages"}, namespace)["run_episode_with_stages"]
        score, done, goal, main, wrist = function(
            env=env, adapter=policy, prompt="Task1", resize_size=2,
            replan_steps=1, num_steps_wait=0, max_steps=5, post_goal_steps=0,
            stage_checks=namespace["STAGE_CHECKS"], goal_monitor_dict={}, stage_checks_sequential=True,
        )
        logger.exception.assert_not_called()
        return env, (score, done, goal, {"stage_success": bool(goal)}, main, wrist)
    path, name = LOOPS[key]
    function = load_nodes(path, {name}, namespace)[name]
    common = dict(task_id=1, env=env, stage_specs=namespace["_task_specs"](1),
                  goal_monitor_dict={}, goal_check_override=namespace["_goal_override_check"](1),
                  fail_on_extra_pour=True, extra_pour_monitor_steps=2, post_goal_steps=0)
    if key == "generic":
        result = function(**common, adapter=policy, prompt="Task1", resize_size=2,
                          replan_steps=1, num_steps_wait=0, max_steps=5)
    else:
        result = function(**common, client=policy,
                          planner=NS(default_subtask_prompt="Task1", infer_sync=lambda *a: "Task1"),
                          args=NS(n_recent=0, async_vlm=False, vlm_queue_size=1, vlm_interval=1,
                                  max_steps=5, num_steps_wait=0, resize_size=2, replan_steps=1),
                          vlm_camera_pose=None, logger=logger)
    logger.exception.assert_not_called()
    return env, result


class Task1SameBasketTest(unittest.TestCase):
    def test_both_baskets_work_with_fresh_state_per_callback(self):
        ns = stages()
        specs = ns["_task_specs"](1)
        self.assertEqual([s.name for s in specs], NAMES)
        callbacks = [(s.name, lambda env, s=s: s.check_fn(env, {}, 0)) for s in specs]
        for basket in (1, 2, 1):
            env = scene(basket, basket)
            done = {name: fn(env) for name, fn in callbacks}
            self.assertTrue(all(done.values()))
            self.assertEqual(ns["_stage_score_pct"](1, done), 100.0)
            self.assertTrue(ns["_goal_override_check"](1)(env, done))
        self.assertFalse(ns["_goal_override_check"](1)(None, {}))

    def test_different_baskets_do_not_complete_second_stage(self):
        ns = stages()
        specs = ns["_task_specs"](1)
        for cookies, tomato in ((1, 2), (2, 1)):
            env = scene(cookies, tomato)
            done = {s.name: s.check_fn(env, {}, 0) for s in specs}
            self.assertEqual(list(done.values()), [True, False])
            self.assertEqual(ns["_stage_score_pct"](1, done), 50.0)
            self.assertFalse(ns["_goal_override_check"](1)(env, done))

    def test_first_object_must_still_be_in_the_shared_basket(self):
        first, second = stages()["_task_specs"](1)
        env = scene(2, None)
        self.assertTrue(first.check_fn(env, {}, 0))
        env.positions.pop("cookies_1")
        env.positions["tomato_sauce_1"] = env.positions["basket_2"] + [0.0, 0.0, 0.05]
        self.assertFalse(second.check_fn(env, {}, 0))

    def test_existing_geometry_limits_and_missing_bodies(self):
        check = stages()["_task_specs"](1)[1].check_fn
        for offset, expected in (([0.119, 0, 0.05], True), ([0.121, 0, 0.05], False),
                                 ([0, 0, -0.051], False), ([0, 0, 0.201], False)):
            env = scene(2, 2)
            env.positions["tomato_sauce_1"] = env.positions["basket_2"] + offset
            self.assertEqual(check(env, {}, 0), expected)
        for missing in ("cookies_1", "tomato_sauce_1", "basket_2"):
            env = scene(2, 2)
            env.positions.pop(missing)
            self.assertFalse(check(env, {}, 0))

    def test_task2_and_task3_keep_their_fixed_basket_rule(self):
        ns = stages()
        for task, objects in ((2, ("butter_1", "popcorn_1")),
                              (3, ("cream_cheese_1", "chocolate_pudding_1"))):
            env = scene()
            for obj in objects:
                env.positions[obj] = env.positions["basket_2"] + [0.0, 0.0, 0.05]
            self.assertFalse(any(s.check_fn(env, {}, 0) for s in ns["_task_specs"](task)))

    def test_public_and_reference_runners_use_actual_same_basket_predicates(self):
        for key in ("public", *LOOPS):
            for tomato_basket, expected in ((2, True), (1, False)):
                with self.subTest(loop=key, tomato_basket=tomato_basket):
                    env, (score, done, goal, diag, _, _) = run_positions(key, tomato_basket)
                    self.assertEqual(score, 100.0 if expected else 50.0)
                    self.assertEqual(goal, expected)
                    self.assertEqual(diag["stage_success"], expected)
                    self.assertEqual(done[NAMES[1]], expected)
                    self.assertEqual(env.steps, 2 if expected else 5)


if __name__ == "__main__":
    unittest.main()

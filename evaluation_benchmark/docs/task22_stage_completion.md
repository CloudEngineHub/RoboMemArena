# Task22 stage coverage correction

Base: official `OpenHelix-Team/RoboMemArena` main commit
`d09f63b671f362985e1e0d038d2014c2d2519488`, fetched on 2026-10-18.

Task22's VLM task description requests two sauce pours, putting the bottle
aside, opening the microwave, and putting the cookies inside. The previous
active scorer only covered lifting and two pours. This correction restores
the omitted stages using existing predicates.

| Stage | Predicate | Required |
| --- | --- | --- |
| 01_Lift_Tomato_Sauce | Existing counting-pour lift | Yes |
| 02_Pour_One | Existing shared pour counter | Yes |
| 03_Pour_Two | Existing shared pour counter | Yes |
| 04_Place_Tomato_Aside | Existing `_near_fixed_position`, historical Task22 target `(0,-0.2,0.50)` and XY/Z tolerances `0.20` | Yes |
| 05_Open_Microwave | Existing `_microwave_open(0.30)`, as used by Task20 | Yes |
| 06_Place_Cookies_Microwave | Existing `_in_microwave("cookies_1")`, as used by Tasks20/24 | Yes |
| 07_Close_Microwave | Existing `_microwave_closed(0.05)` | No |

The required-stage denominator is six. Completing only lift and two pours
scores 50% and cannot yield stage success. Closing remains an optional audit
stage and does not delay successful termination.

The generic adapter, async VLM reference, and VLM5 reference all consume the
canonical scorer. Their counting-task exit checks use required stages, and
Task22 keeps checking for a third pour throughout the microwave phase. The
configured extra-pour window remains a minimum observation requirement.
The generic runner uses the last observed step at budget exhaustion, avoiding
the previous extra unobserved increment. Monitor-incomplete failures are
classified explicitly in all three runners.

The bundled minimal-runtime stage table is updated to match. Both Task22
BDDL copies remove the unrelated goal of putting the sauce bottle inside
the microwave; their spatial goal is cookies inside the microwave. BDDL alone
cannot certify the required counting and sequence, which remain stage checks.

## Limits of predicate reuse

This patch reuses existing thresholds and physical proxies. The aside check
is a position proxy, not a release/contact test; the existing microwave box
is broad; the shared pour counter still uses its existing rotation metric.
Consequently this is a stage-coverage correction, not a claim that the
physical-predicate issues in the October audit are resolved. No new contact,
stability, collision geometry, or pour-angle implementation is included.

## Validation

```bash
python -m unittest discover -s evaluation_benchmark/tests -p test_task22_completion.py -v
python -m unittest discover -s evaluation_benchmark/tests -p test_shared_pour_counter.py -v
```

Tests exercise exact runner functions with synthetic states. No model rollout
or GPU job is required, and these checks do not establish a real success rate.

Validated on 2026-10-18: eight focused Task22 tests and four shared-counter
tests passed; Python compilation and `git diff --check` passed. The legacy
stage-table tests preserve its existing body-pour implementation rather than
claiming it uses the canonical shared counter.

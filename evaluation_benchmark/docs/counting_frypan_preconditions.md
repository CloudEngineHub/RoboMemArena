# Counting-task frypan preconditions

Base: official `a1099e0b8e329022f671215f2822cf58f22906c4`, checked 2026-10-18.

Tasks8/9/15 now require four ordered stages: food in frypan, lift the pour
source, pour once, pour twice. Task8 checks `chocolate_pudding_1`; Tasks9/15
check `butter_1`. The existing shared pour counter and monitor are unchanged.

The food check reuses `_in_container_body` from the historical generic stage
table: frypan body reference, XY radius 0.10 m for Tasks8/9 or 0.12 m for
Task15, with relative Z strictly between -0.05 and 0.15 m. The legacy minimal
runtime receives the same stage addition, retaining its existing pour logic.

Only stage coverage is corrected. This inherited position proxy does not
certify release, contact, or stable support, nor recheck food retention after
its stage is latched. No final bottle/carton placement is added. Task10,
language prompts, BDDL, policy checkpoints and other task definitions remain
unchanged. Stage numbering shifts for Tasks8/9/15; old three-stage scores
must not be combined with the new four-stage scores as one protocol.

Validation: five focused frypan tests, eight Task22 tests, four shared-counter
tests, and the existing counting-stage-map test passed. Tests exercise actual
runner functions with mocked state/policy inputs; no simulator rollout or
model success-rate claim is made.

```bash
python -m unittest discover -s evaluation_benchmark/tests -p test_frypan_preconditions.py -v
```

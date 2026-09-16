# CareCompass evaluation

- Run: `2026-09-16T20:54:05+00:00`
- Cases: **52**
- Retrieval: `hybrid`
- Model: `retrieval-only` (deterministic paths only)

| Metric | Value |
| --- | --- |
| cases passed (%) | 96.2 |
| urgency exact (%) | 96.2 |
| under triage (%) | 3.8 |
| over triage (%) | 0.0 |
| emergency recall (%) | 100.0 |
| emergency precision (%) | 100.0 |
| red flag recall (%) | 100.0 |
| negation specificity (%) | 100.0 |
| retrieval recall at k (%) | 100.0 |
| restricted intent recall (%) | 100.0 |
| out of scope detected (%) | 100.0 |
| dose leak count | 0 |
| model answers | 0 |
| median latency ms | 7 |

## Failing cases

- `fever_04` — under-triage: predicted INFO, expected SELF_CARE
- `med_01` — under-triage: predicted INFO, expected SELF_CARE

## Reading these numbers

`emergency_recall` is the number to look at first. Missing an emergency is the failure mode with a real cost; over-triage costs a wasted trip. `negation_specificity` guards the other side: it measures how often "I have no chest pain" is correctly *not* escalated, which is what stops a keyword matcher from crying wolf until people stop reading the warnings.

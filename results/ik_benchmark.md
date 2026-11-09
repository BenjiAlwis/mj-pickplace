| Variant | Success | Pos err med / max (mm) | Rot err med / max (deg) | Iters med / p95 | Time med / p95 (ms) |
|---|---|---|---|---|---|
| DLS, single descent (no restarts) | 84.6% | 0.0093 / 0.1000 | 0.00004 / 0.0228 | 18 / 134 | 3.3 / 24.2 |
| DLS + up to 20 random restarts | 98.9% | 0.0107 / 0.1000 | 0.00003 / 0.0344 | 18 / 203 | 3.3 / 38.2 |
| ablation: clip joints instead of wrap, 20 restarts | 96.4% | 0.0123 / 0.1000 | 0.00007 / 0.0516 | 78 / 627 | 14.0 / 112.9 |

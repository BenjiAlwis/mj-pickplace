| Config | N | Success | Grasp | Place err med / p95 (mm) | Yaw err med / p95 (deg) | Plan time med / p95 (s) | Joint path (rad) | TCP path (m) | Cycle (s) | IK fails (eps) | Track err (rad) | Unplanned contact (eps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 200 | 100.0% [98.1, 100.0] | 100.0% | 0.2 / 0.5 | 0.3 / 0.3 | 0.82 / 1.21 | 5.10 | 1.88 | 11.9 | 0 (0) | 0.002 | 0 |
| noisy | 200 | 100.0% [98.1, 100.0] | 100.0% | 0.3 / 0.7 | 0.4 / 1.1 | 0.75 / 1.16 | 5.07 | 1.89 | 11.9 | 0 (0) | 0.002 | 0 |
| noisy_large | 200 | 86.5% [81.1, 90.6] | 87.0% | 0.4 / 0.8 | 0.3 / 1.6 | 0.72 / 1.18 | 5.08 | 1.88 | 11.9 | 0 (0) | 0.002 | 0 |
| noisy_no_inhand | 200 | 100.0% [98.1, 100.0] | 100.0% | 3.5 / 11.1 | 0.5 / 4.4 | 0.79 / 1.23 | 5.05 | 1.89 | 11.9 | 0 (0) | 0.002 | 0 |
| no_feedforward | 200 | 100.0% [98.1, 100.0] | 100.0% | 0.4 / 0.7 | 0.4 / 0.5 | 0.78 / 1.16 | 5.11 | 1.89 | 12.0 | 0 (0) | 0.212 | 0 |

Failure breakdown:
- **nominal**: none
- **noisy**: none
- **noisy_large**: FAIL_GRASP_no_pad_contact x9, FAIL_LIFT_dropped x17, FAIL_PLACE_collision x1
- **noisy_no_inhand**: none
- **no_feedforward**: none

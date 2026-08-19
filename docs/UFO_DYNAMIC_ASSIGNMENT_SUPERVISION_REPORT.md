# UFO Dynamic Assignment Supervision Report

## Scope and isolation

- Independent worktree: `workspace/yx/ufo_dynamic_fix_4090`
- Base: `origin/ufo_h200_reproduction` at `7ff734d`
- Branch: `fix/dynamic-assignment-supervision`
- The active H200 directory was not modified.
- Scope was limited to object-assignment GT construction, tests, and a 500-step A/B diagnostic.

## Root cause

The public-v1-style `predicted_mean` target is self-referential. The target point is
the mean of the 64 Gaussian means decoded for each scene token. Although it is
detached and evaluated under `torch.no_grad()`, its numerical value still depends
on the model's current depth/geometry prediction. At poor initialization, the
predicted point lies outside every tracked box, so the generated class target is
background and object CE reinforces the all-background solution.

The real-data audit and 500-step control confirmed this failure mode: the
`predicted_mean` run produced zero foreground GT tokens at every logged step even
though the sampled scenes contained valid tracked boxes.

## Data flow and status

| Stage | Implementation | Status |
|---|---|---|
| Waymo depth and tracked boxes | Camera-Z depth plus per-frame world-space box corners | PUBLIC_V1 |
| Scene-token assignment | One query per scene token and one object distribution shared by its 64 child Gaussians | PAPER_EXPLICIT |
| Predicted-mean target | Average predicted child-Gaussian means, detached, then point-in-box | PUBLIC_V1 |
| LiDAR-anchor target | Valid observed LiDAR points are back-projected and pooled per 8x8 image patch | REPRODUCTION_DECISION |
| Empty LiDAR patch | Excluded from object CE | REPRODUCTION_DECISION |
| Exact anchor pooling and missing-depth policy | Not specified by the paper | UNKNOWN |

The anchor is ordered as `time -> camera -> patch-row -> patch-column`, matching
the scene-token flattening. Its assignment is expanded spatially so one token's
label is shared by exactly 64 child Gaussians. Invalid/padded boxes cannot be
selected. The anchor is detached and is used only to construct object CE targets;
it does not replace predicted geometry and does not enter rendering or motion.

## Tests

The following tests passed:

- token, camera, timestamp, patch, and 64-child indexing
- synthetic inside/outside/padded box oracle
- independence under a 1000 m perturbation of predicted means
- finite object CE and nonzero bbox-head gradient
- empty-anchor ignore behavior
- existing token-to-child assignment broadcast test

`pytest` is not installed in this environment, so the test functions were invoked
directly. `py_compile` and `git diff --check` also passed.

The real Waymo audit used scene index 13, start frame 27, context frames
`[27, 32, 37, 42]`. It found 40 valid boxes, 5,110 valid patch anchors, and 454
foreground anchors (8.88%). Perturbing predicted means did not change any LiDAR
anchor label, while the perturbed predicted-mean path produced zero foreground.

## 500-step A/B

Both runs used seed 1, the same D50 dynamic-rich pool, initialization, data,
optimizer, LR, network, losses, and 500 optimizer steps. The only changed config
field was `object_assignment_gt_mode`.

Commands:

```bash
python main.py --config configs/dynamic_assignment_ab/predicted_mean_500.json
python main.py --config configs/dynamic_assignment_ab/lidar_anchor_500.json
```

| Metric | predicted_mean | lidar_anchor |
|---|---:|---:|
| Mean dynamic GT ratio | 0.0000% | 6.0963% |
| Final dynamic GT ratio | 0.0000% | 8.3683% |
| Mean dynamic GT count | 0.0 | 264.4 |
| Mean predicted dynamic ratio | 4.1171% | 14.2604% |
| Mean foreground recall | 0.0000% | 30.2367% |
| Mean foreground precision | 0.0000% | 19.5949% |
| Mean bbox-head grad norm | 0.3201 | 1.4700 |
| Final bbox-head grad norm | 0.0197 | 2.1329 |
| Final full PSNR | 17.4382 dB | 17.2039 dB |
| Final dynamic PSNR | 16.9057 dB | 16.5841 dB |
| Mean sec/iter | 1.479 | 1.566 |

The 500-step A/B confirms that `lidar_anchor` removes the self-referential target
collapse and supplies foreground gradients. The short-run RGB difference is not
evidence against the anchor because both runs are still early in the 5,000-step LR
warmup.

## Continuation to 5,000 steps

Both 500-step checkpoints were resumed with their optimizer, loss scaler, RNG, and
sampler state intact. No implementation or loss changes were made. Metrics below
are means over the 100 optimizer steps immediately preceding each milestone.

| Step | Mode | GT dynamic | Pred dynamic | FG recall | FG precision | Class loss | Bbox grad | Dynamic PSNR | Full PSNR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | predicted_mean | 0.00% | 0.00% | N/A | N/A | 0.0000 | 0.0226 | 18.374 | 19.677 |
| 1k | lidar_anchor | 6.70% | 22.17% | 66.41% | 25.07% | 0.5326 | 1.6094 | 17.622 | 18.030 |
| 3k | predicted_mean | 0.00% | 0.00% | N/A | 0.00% | 0.0000 | 0.0197 | 17.433 | 18.808 |
| 3k | lidar_anchor | 3.95% | 14.23% | 81.57% | 27.42% | 0.3809 | 2.0042 | 17.067 | 18.281 |
| 5k | predicted_mean | 0.00% | 0.00% | N/A | 0.00% | 0.0000 | 0.0295 | 17.606 | 19.483 |
| 5k | lidar_anchor | 7.85% | 21.36% | 77.12% | 31.36% | 0.5421 | 2.7751 | 17.646 | 19.164 |

The predicted-mean path did not recover at any milestone. Its target, prediction,
and object loss remained identically background through the end of warmup. The
LiDAR-anchor path kept a healthy, independently observed foreground target and did
not collapse: recall increased from 66.4% at 1k to 77.1% at 5k, precision increased
from 25.1% to 31.4%, and bbox-head gradients remained substantial.

The conditional `lidar_anchor + foreground/background balanced CE` experiment was
not started because its trigger condition was not met. There is no 3k-5k evidence
of learned assignment collapsing toward background under the current D50 stream.

Twelve-scene held-out validation:

| Step | predicted PSNR / SSIM / D-RMSE | anchor PSNR / SSIM / D-RMSE |
|---:|---:|---:|
| 1k | 17.807 / 0.4703 / 11.722 m | 17.149 / 0.4441 / 10.471 m |
| 3k | 18.208 / 0.4916 / 13.358 m | 17.863 / 0.4728 / 12.352 m |
| 5k | 18.877 / 0.4966 / 12.230 m | 18.360 / 0.4765 / 12.629 m |

This validation path returned `NaN` for dynamic-region metrics in both runs, so it
cannot support a learned-vs-control dynamic comparison. Training dynamic-mask
metrics were finite and are reported above. The 5k checkpoint is produced at
optimizer step 4,999; the current training loop would enable LPIPS on the next
optimizer step, so LPIPS was not active in either compared window.

Artifacts:

- `outputs/dynamic_assignment_ab/real_sample_audit.json`
- `outputs/dynamic_assignment_ab/predicted_mean_500/training_metrics.json`
- `outputs/dynamic_assignment_ab/lidar_anchor_500/training_metrics.json`
- continuation logs: each run's `continue_500_to_5000.log`
- checkpoints: each run's `ckpt_000499.pth` through `ckpt_004999.pth`

## Recommendation

The predicted-mean self-reference is a real public-v1 training failure in this
setup and does not self-correct by 5k. The independent LiDAR anchor with an explicit
valid/ignore mask is the supported assignment-GT reproduction decision. The 5k run
does not justify foreground/background balanced CE because the learned assignment
remained healthy through warmup. Before a formal 100k switch, synchronize only
after review and reproduce a short H200 continuation with the same anchor policy;
do not fall back to predicted means for missing LiDAR anchors.

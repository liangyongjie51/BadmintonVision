# Reproducing the Results

This guide maps each reported result to the code that produces it.

## 0. Environment

```bash
pip install -r requirements.txt
# tactical/ and stats/ run with the core deps; ssl/detection/temporal need the DL extras.
```

All randomness is seeded via `project.seed` in `configs/default.yaml`
(`badmintonvision.utils.config.set_seed` seeds Python/NumPy/PyTorch).

## 1. Statistics and figures (no broadcast data required)

```bash
python scripts/06_statistical_analysis.py   # -> outputs/stats_results.json
python scripts/07_make_figures.py           # -> outputs/figures/*.{pdf,png}
```

`06` reproduces, with the manuscript's point estimates:

| Result | Output key |
|---|---|
| Cohen's d + 95% CI for all movement metrics | `movement` |
| Linear mixed-effects β + 95% CI (random intercept for game) | `movement[*].lmm` |
| Markov transition matrices | `markov_matrices` |
| Chi-square homogeneity (χ², df, p) per stroke | `chi_square_tests` |
| Lag-1 sequential adjusted residuals | `lag_sequential_z` |
| Match- vs frame-level split (leakage audit) | `data_split` |
| Temporal-coherence audit (96.7%) | `temporal_coherence` |

`07` regenerates the figures at 190 mm width, 600 DPI, PDF + PNG. Note on figure
numbering: the statistical-validation panels are emitted grouped (as
`Figure_9.{pdf,png}`) for convenience; in the final manuscript these panels were
distributed into the existing figures — data-split leakage and temporal-coherence
→ Figure 6 (f, g); mixed-effects forest plot → Figure 7 (g); lag-sequential
residuals → Figure 8 (g).

## 2. Detection (paper Section 3.1)

```bash
python scripts/03_train_detector.py --config configs/default.yaml
```

Reports AP@0.5 per class and mAP@0.5 at inference settings conf = 0.25,
NMS IoU = 0.70. The **match-level** mAP@0.5 of 0.941 is the primary, conservative
figure; a frame-level split would inflate it (≈0.956) — this contrast is the
leakage audit in `outputs/stats_results.json["data_split"]`.

## 3. Self-supervised pre-training (Section 3.1, +9.8%)

```bash
python scripts/02_ssl_pretrain.py ssl.method=convnextv2_mae   # or ssl.method=simsiam
```

The detector is then trained with `detection.ssl_init=true` to transfer the SSL
backbone (feature fusion). The SSL objective for SimSiam is the negative cosine
similarity with stop-gradient (Eq. 1; denominator = ||z1||·||z2||), implemented
in `badmintonvision/ssl/simsiam.py`.

## 4. Temporal recognition (Section 3.2, +6.3%)

```bash
python scripts/04_train_temporal.py --config configs/default.yaml \
       --train-npz data/temporal/train_windows.npz \
       --val-npz   data/temporal/val_windows.npz
```

MotionFormer (`badmintonvision/temporal/motionformer.py`) consumes 10-frame
windows (400 ms at 25 fps) of frozen detection features.

### Expected feature-window file (`.npz`)

| Array | Shape | Meaning |
|---|---|---|
| `features` | (N, W, D) | per-frame backbone features, W = window (10) |
| `labels` | (N,) | stroke class index per window |
| `rally_id` | (N,) | rally id per window (kept within one split) |

## 5. Tactical analysis (Sections 3.3–3.4)

```bash
python scripts/05_run_tactical_analysis.py --tracks data/tracks.json
# -> outputs/rally_metrics.csv, outputs/tactical_summary.json
```

This applies the per-match homography, Savitzky-Golay smoothing (window 5,
order 2), and computes movement metrics, Markov matrices, and lag-1 adjusted
residuals. The downstream comparisons (Section 3.3) then come from
`scripts/06_statistical_analysis.py`.

## Unit tests

```bash
pytest -q
```

Tests in `tests/` check the tactical metrics (path efficiency, direction changes,
homography round-trip) and the statistics (Cohen's d, lag-sequential residuals)
on small synthetic inputs.

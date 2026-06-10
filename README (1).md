# BadmintonVision

**A Deep Learning Framework for Automated Tactical Analysis in Elite Badminton**

This repository contains the reference implementation, analysis code, and
reproducibility materials for the BadmintonVision framework. It couples
self-supervised pre-training, object detection, a Transformer temporal module
(MotionFormer), and a tactical-analysis stack (homography, movement metrics,
Markov-chain and lag-sequential analysis) into a single, reproducible pipeline.

> **Reproducibility note.** This release is provided to satisfy open-science and
> peer-review requirements. The underlying material is **copyrighted broadcast
> footage**, which we do not redistribute. We release the code, the model and
> training definitions, the annotation protocol (`docs/annotation_guidelines.md`),
> and derived non-identifying analysis data, so the pipeline can be reconstructed
> from publicly broadcast matches. Pre-trained weights are released as described
> in [Pre-trained weights](#pre-trained-weights).

---

## Key results

| Stage | Metric | Value |
|---|---|---|
| Detection (YOLOv12m, match-level split) | mAP@0.5 | **0.941** |
| + Self-supervised pre-training (ConvNeXt V2) | relative gain | **+9.8%** |
| + Temporal modelling (MotionFormer) | additional gain | **+6.3%** |
| Movement: path efficiency (scoring vs conceding) | Cohen's d [95% CI] | 1.77 [1.39, 2.15] |
| Movement: base recovery time | Cohen's d [95% CI] | −1.56 [−1.93, −1.19] |

All numbers are reproduced exactly by `scripts/06_statistical_analysis.py`
(see [Reproducing the statistics and figures](#reproducing-the-statistics-and-figures)).

---

## Repository structure

```
BadmintonVision/
├── badmintonvision/            # importable package
│   ├── data/                   # frame extraction, match-level splitting, datasets
│   ├── ssl/                    # ConvNeXt V2 MAE + SimSiam pre-training
│   ├── detection/              # YOLOv12 detector with SSL-augmented backbone
│   ├── temporal/               # MotionFormer temporal action recognition
│   ├── tactical/               # homography, trajectory, metrics, Markov, lag-sequential
│   ├── stats/                  # effect sizes + CIs, mixed-effects, chi-square
│   └── utils/                  # config loading, seeding, court geometry
├── configs/default.yaml        # all hyper-parameters (paper Sections 2.1-2.10)
├── scripts/                    # numbered, runnable pipeline stages (01-07)
├── docs/                       # annotation guidelines, dataset card, reproduction guide
├── court_keypoints/            # per-match court corner points for homography
├── tests/                      # unit tests for the tactical/statistics code
├── requirements.txt
└── LICENSE                     # MIT
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/<org>/BadmintonVision.git
cd BadmintonVision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: install the package in editable mode
pip install -e .
```

The tactical-analysis and statistics code (`tactical/`, `stats/`) depends only on
NumPy/SciPy/pandas/statsmodels and runs out of the box. The deep-learning stages
(`ssl/`, `detection/`, `temporal/`) additionally require PyTorch, `timm`, and
`ultralytics`; see `requirements.txt`.

---

## Data preparation

We cannot redistribute broadcast footage. To reconstruct the dataset:

1. Obtain the broadcast videos of the matches listed in `docs/dataset_card.md`
   (BWF World Championships, 2019–2025, singles, quarter-finals onward).
2. Annotate following `docs/annotation_guidelines.md` (5 classes: Player,
   Forehand, Backhand, Serve, Jump Smash; inter-annotator agreement κ = 0.89).
3. Mark the four half-court corner points per match in `court_keypoints/`
   (see `court_keypoints/README.md`) for the homography.
4. Place raw videos, frames, labels and the rally index at the paths configured
   in `configs/default.yaml` (`paths:` block).

**Data splitting is performed at the match level** so that no frame, rally clip,
or temporal window from a given match appears in more than one split, and the
self-supervised pre-training images are kept disjoint from the evaluation
rallies at the clip level. This is enforced by `badmintonvision/data/splits.py`.

---

## Reproducing the full pipeline

```bash
# 1. Extract frames + variance-of-Laplacian blur filtering
python scripts/01_extract_frames.py --config configs/default.yaml

# 2. Self-supervised pre-training (ConvNeXt V2 MAE; or ssl.method=simsiam)
python scripts/02_ssl_pretrain.py  --config configs/default.yaml

# 3. Train + evaluate the detector (SSL-initialised backbone, match-level split)
python scripts/03_train_detector.py --config configs/default.yaml

# 4. Train MotionFormer on frozen detection features (10-frame / 400 ms windows)
python scripts/04_train_temporal.py --config configs/default.yaml

# 5. Tactical analysis (homography -> metrics, Markov, lag-sequential)
python scripts/05_run_tactical_analysis.py --config configs/default.yaml \
       --tracks data/tracks.json
```

Any configuration field can be overridden on the command line, e.g.
`python scripts/03_train_detector.py detection.epochs=150 project.seed=0`.

### Reproducing the statistics and figures

These two stages need **no** broadcast data and run immediately:

```bash
python scripts/06_statistical_analysis.py     # -> outputs/stats_results.json
python scripts/07_make_figures.py             # -> outputs/figures/*.pdf, *.png
```

`06` reproduces every reported statistic (effect sizes with 95% CIs, linear
mixed-effects models, full chi-square tests, lag-sequential adjusted residuals,
the match- vs frame-level leakage audit, and the temporal-coherence audit).
`07` regenerates all figures at 190 mm width, 600 DPI, in both PDF and PNG.
Figures use Arial if available and fall back to the metric-compatible Liberation
Sans otherwise (see `assets/fonts/README.md`).

---

## Pre-trained weights

The self-supervised encoder, the detection weights, and the MotionFormer weights
are released on the GitHub *Releases* page of this repository upon publication. A
`scripts/download_weights.sh` helper will fetch them into `weights/`.

---

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```
Liang Y., Zhou Y., Zhang G. BadmintonVision: A Deep Learning Framework for
Automated Tactical Analysis in Elite Badminton.
```

## License

Code is released under the MIT License (`LICENSE`). The broadcast footage used to
train the models is **not** included and remains the property of its respective
rights holders.

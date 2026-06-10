# Dataset Card — BadmintonVision

## Overview

| Field | Value |
|---|---|
| Domain | Elite badminton singles (broadcast video) |
| Source events | BWF World Championships, 2019–2025 (quarter-finals onward) |
| Camera | Standard end-court broadcast angle |
| Annotated images | 30,000 (expert-labelled) |
| Unlabeled images (SSL) | 139,501 |
| Candidate frames extracted | 208,891 |
| Frames after blur filtering | 169,501 (81.1% retained) |
| Rally clips | 5,865 |
| Mean rally duration | 13.27 s (SD 6.4 s; per-subset SD 5.8–7.1 s) |
| Classes | Player, Forehand, Backhand, Serve, Jump Smash |
| Inter-annotator agreement | Cohen's κ = 0.89 |

## Per-championship composition

| Year | Venue | Gender | Rally clips | Avg. duration (s) | SD (s) |
|---|---|---|---|---|---|
| 2025 | Paris | Men | 658 | 12.77 | 6.2 |
| 2025 | Paris | Women | 545 | 13.14 | 6.6 |
| 2023 | Copenhagen | Men | 642 | 12.69 | 5.9 |
| 2023 | Copenhagen | Women | 537 | 13.34 | 6.4 |
| 2022 | Tokyo | Men | 583 | 13.28 | 6.1 |
| 2022 | Tokyo | Women | 615 | 13.87 | 6.8 |
| 2021 | Huelva | Men | 569 | 13.09 | 6.0 |
| 2021 | Huelva | Women | 597 | 14.65 | 7.1 |
| 2019 | Basel | Men | 520 | 13.40 | 5.8 |
| 2019 | Basel | Women | 599 | 14.71 | 7.0 |
| **Total** | | | **5,865** | 13.27 | 6.4 |

## Splits

Splitting is performed at the **match level** (not frame or rally level) and is
stratified by gender. No frame, rally clip, or temporal window from a given match
appears in more than one split. The self-supervised pre-training images are kept
disjoint, at the clip level, from the evaluation rallies. See
`badmintonvision/data/splits.py`.

## Licensing and redistribution

The underlying footage is **copyrighted broadcast material** and is **not**
redistributed. This repository provides:

- the annotation protocol (`docs/annotation_guidelines.md`),
- code to extract frames and labels from videos the user obtains independently,
- derived non-identifying analysis data (court-normalised trajectories and
  stroke-transition counts) sufficient to reproduce the reported statistics,
- the trained model weights (released separately; see the main README).

Schematic figures (e.g. Figure 2a) use pictographs rather than broadcast frames
to avoid reproducing copyrighted or personally identifiable content.

## Intended use and limitations

Intended for research on automated tactical analysis of **elite singles** play
from the standard broadcast angle. The dataset/models are **not** validated for
doubles, lower-level play, or other camera configurations, and the tactical
analyses are exploratory and correlational (no causal claims).

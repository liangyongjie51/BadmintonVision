# Annotation Guidelines — BadmintonVision

These are the annotation guidelines used to label the BadmintonVision detection
dataset (30,000 expert-annotated images; inter-annotator agreement Cohen's
κ = 0.89). They are released so the dataset can be reconstructed from publicly
broadcast matches.

## 1. Scope and source material

- **Source**: broadcast video of BWF World Championships singles matches
  (2019–2025), quarter-finals onward, using the standard end-court (behind the
  baseline) broadcast camera angle.
- **Unit of annotation**: individual video frames sampled at a fixed stride
  (every 10th frame) after a variance-of-Laplacian blur filter (threshold 100).
- Annotators label the **near-court player** (the player closest to the camera)
  for stroke classes; the far player is labelled only with the `Player` box.

## 2. Object classes

Bounding boxes are drawn in YOLO format. Five classes (index order matters):

| Idx | Class | Definition |
|---|---|---|
| 0 | `Player` | The full body of a player (either player), tight bounding box including racket-side arm but excluding the racket. |
| 1 | `Forehand` | Near player executing a forehand stroke (racket on the dominant-hand side, palm facing the shuttle). Box covers the player at the moment of the stroke. |
| 2 | `Backhand` | Near player executing a backhand stroke (racket crossing to the non-dominant side, back of the hand toward the shuttle). |
| 3 | `Serve` | Either player performing a service action, from shuttle release to contact. |
| 4 | `Jump_Smash` | Near player performing a jumping overhead smash (both feet off the ground during a downward overhead strike). |

A frame may contain both a `Player` box (far player) and a stroke-class box
(near player). Stroke classes are **mutually exclusive** for a given player in a
given frame.

## 3. Temporal extent of a stroke

Strokes are dynamic. For temporal recognition the stroke label is assigned to the
**stroke window** spanning preparation → execution → recovery (≈10 frames /
400 ms at 25 fps). The single-frame detection label is assigned to the frame at
or nearest the **moment of racket–shuttle contact** (the most discriminative
instant).

## 4. Bounding-box conventions

- Draw the **tightest** box that fully contains the player's body.
- Include extended limbs that are part of the stroke (the hitting arm); exclude
  the racket head and the shuttle.
- If the player is partially occluded (e.g. by the net post or an on-screen
  graphic), box the **visible** extent; if < 25% of the body is visible, skip.
- One box per player instance; do not merge two players into one box.

## 5. Difficult / edge cases

- **Ambiguous forehand/backhand**: classify by the racket-face orientation at
  contact, not by court side. When genuinely undecidable, use `Player` and flag
  the frame for adjudication.
- **Net shots / drives** that are not jump smashes are labelled by their
  forehand/backhand character.
- **Motion blur**: if contact is too blurred to determine the stroke, do not
  guess — label `Player` and flag.
- **Replays / slow-motion / split-screen / score overlays covering the player**:
  exclude these frames.
- **Between-rally footage** (celebrations, towel breaks, crowd shots): exclude.

## 6. Quality control and agreement

- Each frame is independently labelled by two trained annotators.
- Disagreements (class mismatch, or IoU < 0.5 between the two boxes) are
  adjudicated by a third expert.
- Agreement is monitored with Cohen's κ on the stroke class; the released
  dataset achieved κ = 0.89 ("almost perfect").
- A reference set of ~200 adjudicated frames is used to onboard and periodically
  re-check every annotator.

## 7. Court key-points (for homography)

In addition to player boxes, each match requires the four **half-court** corner
points (near-left, near-right, far-right, far-left), recorded once per match in
`court_keypoints/<match_id>.json` (see `court_keypoints/README.md`). These map
the image to the BWF singles half-court (5.18 m width × 6.70 m net-to-baseline).

## 8. File layout

```
data/
├── frames/<match_id>/<match_id>_<frame>.jpg
├── annotations/<match_id>/<match_id>_<frame>.txt   # YOLO: cls cx cy w h (normalised)
├── annotations/classes.txt                          # one class name per line, index order
└── rally_index.csv                                  # rally_id,match_id,gender,start_f,end_f,scorer
```

# Fonts

The figures in the paper use **Arial** (190 mm width, 600 DPI). Arial is a
proprietary Monotype typeface and is therefore **not** bundled in this
repository.

`scripts/07_make_figures.py` will:
1. use Arial if it is installed/registered with Matplotlib; otherwise
2. fall back to **Liberation Sans** (a metric-compatible, freely licensed Arial
   substitute) or, failing that, Matplotlib's bundled DejaVu Sans.

To reproduce the exact typography, install Arial locally (e.g. place
`Arial.ttf`, `Arial_Bold.ttf`, `Arial_Italic.ttf` in this folder) and rerun the
figure script.

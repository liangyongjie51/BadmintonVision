"""BadmintonVision: a deep-learning framework for automated tactical analysis
in elite badminton.

Modules
-------
- data      : frame extraction, match-level splitting, datasets
- ssl       : self-supervised pre-training (ConvNeXt V2 MAE, SimSiam)
- detection : YOLOv12 detector with SSL-augmented backbone
- temporal  : MotionFormer temporal action recognition
- tactical  : homography, trajectory metrics, Markov & lag-sequential analysis
- stats     : effect sizes, mixed-effects models, chi-square tests
- utils     : config loading, seeding
"""
__version__ = "1.0.0"

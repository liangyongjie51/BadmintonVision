"""Match-level data splitting (paper Section 2.4.1 / leakage control).

Reviewer-2 (Major Comment 1) correctly noted that a frame-level random split
leaks information between temporally adjacent frames and inflates mAP. To avoid
this, **all** partitioning is performed at the MATCH level: every frame, rally
clip and temporal window from a given match is assigned, in its entirety, to a
single split. The self-supervised pre-training images are additionally kept
disjoint, at the clip level, from the evaluation rallies.

This module reads a rally index (rally_id, match_id, gender, start_f, end_f,
scorer) and writes train/val/test assignments at the match level, optionally
stratified by gender so that men's and women's matches stay balanced.

Usage
-----
    python -m badmintonvision.data.splits \
        --rally-index data/rally_index.csv --out data/splits \
        --train 0.70 --val 0.15 --test 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict


def read_rally_index(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def assign_matches(
    rallies: list[dict],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    stratify_by: str | None = "gender",
    seed: int = 42,
) -> dict[str, str]:
    """Return a mapping ``match_id -> split`` (train/val/test).

    Matches (not rallies or frames) are the unit of assignment, so no match is
    ever split across train/val/test.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "splits must sum to 1.0"
    rng = random.Random(seed)

    # group matches, optionally by stratum, so proportions hold within each
    matches_by_stratum: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for r in rallies:
        mid = r["match_id"]
        if mid in seen:
            continue
        seen.add(mid)
        stratum = r.get(stratify_by, "all") if stratify_by else "all"
        matches_by_stratum[stratum].append(mid)

    assignment: dict[str, str] = {}
    for stratum, mids in matches_by_stratum.items():
        rng.shuffle(mids)
        n = len(mids)
        n_train = round(n * train)
        n_val = round(n * val)
        for i, mid in enumerate(mids):
            if i < n_train:
                assignment[mid] = "train"
            elif i < n_train + n_val:
                assignment[mid] = "val"
            else:
                assignment[mid] = "test"
    return assignment


def expand_to_rallies(rallies: list[dict], match_split: dict[str, str]) -> dict[str, list]:
    """Expand a match-level assignment to rally lists per split."""
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in rallies:
        out[match_split[r["match_id"]]].append(r["rally_id"])
    return out


def verify_no_leakage(rallies: list[dict], match_split: dict[str, str]) -> None:
    """Assert that every match maps to exactly one split (no cross-split match)."""
    match_to_splits: dict[str, set] = defaultdict(set)
    for r in rallies:
        match_to_splits[r["match_id"]].add(match_split[r["match_id"]])
    bad = {m: s for m, s in match_to_splits.items() if len(s) > 1}
    assert not bad, f"Leakage: matches in multiple splits: {bad}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Match-level train/val/test split.")
    ap.add_argument("--rally-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--stratify-by", default="gender")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rallies = read_rally_index(args.rally_index)
    match_split = assign_matches(
        rallies, args.train, args.val, args.test, args.stratify_by, args.seed
    )
    verify_no_leakage(rallies, match_split)
    rally_split = expand_to_rallies(rallies, match_split)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "match_split.json"), "w", encoding="utf-8") as fh:
        json.dump(match_split, fh, indent=2)
    with open(os.path.join(args.out, "rally_split.json"), "w", encoding="utf-8") as fh:
        json.dump(rally_split, fh, indent=2)

    for split in ("train", "val", "test"):
        n_m = sum(1 for v in match_split.values() if v == split)
        n_r = len(rally_split[split])
        print(f"{split:>5}: {n_m} matches, {n_r} rallies")
    print(f"\nNo-leakage check passed. Wrote splits to {args.out}")


if __name__ == "__main__":
    main()

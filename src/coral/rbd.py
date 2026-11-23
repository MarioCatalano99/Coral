"""RBD interval helpers shared across analyses.

The RBD annotation pipeline (parse_rbd_domtbl.py) emits a long-format TSV with
1-based inclusive `merged_start`/`merged_end` per protein. This module loads
that TSV into the `{protein_id: intervals}` dict that `DataCollatorForRBDPrediction`
consumes, and converts intervals to a 0-based residue boolean mask.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_rbd_intervals(tsv_path: str | Path) -> Dict[str, List[Tuple[int, int]]]:
    """Load `<prefix>.rbd_intervals.tsv` into {protein_id: [(start, end), ...]}.

    Coordinates are 1-based inclusive, matching the file. Intervals are
    sorted by start (the parser already sorts them and merges overlaps, so
    no additional cleaning is needed).
    """
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["protein_id", "merged_start", "merged_end"])
    out: Dict[str, List[Tuple[int, int]]] = {}
    for pid, group in df.groupby("protein_id", sort=False):
        out[pid] = [(int(s), int(e))
                    for s, e in zip(group["merged_start"], group["merged_end"])]
    return out


def intervals_to_residue_mask(intervals: List[Tuple[int, int]],
                              length: int) -> np.ndarray:
    """Expand 1-based inclusive intervals to a 0-based boolean residue mask.

    The collator layers special-token offsets on top of this; this helper
    stays tokenizer-agnostic so it can be used in scripts and tests too.
    """
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        # 1-based inclusive -> 0-based half-open
        s = max(0, start - 1)
        e = min(length, end)
        if s < e:
            mask[s:e] = True
    return mask

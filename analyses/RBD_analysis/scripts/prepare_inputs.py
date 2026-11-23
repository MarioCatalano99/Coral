#!/usr/bin/env python3
"""Inputs of the RBD analysis (run.sh steps 1 and 2).

Two steps, run before and after scoring with scripts/predict.py:

  pairs     write the fold-1596 curated positives (train + val, labels == 1) as
            N shard CSVs for scripts/predict.py: the 25,202 pairs of the paper.
  universe  keep the pairs the checkpoint predicts positive (p > 0.5), write
            all_predicted_interactions.csv and the protein FASTA, and check that
            every protein is in the Pfam-scanned universe (data/rbd/scanned_proteins.fa).

The universe check matters because hmmscan output only lists proteins with hits:
a protein missing from the scanned FASTA would silently look RBD-free.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from coral import paths  # noqa: E402

LOGGER = logging.getLogger("rbd_inputs")
FOLD_DIR = paths.DATASETS_DIR / "redundant_split" / "fold_1596"
PAIR_COLUMNS = ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs", "labels"]
EXPECTED_POSITIVES = 25_202


def write_pairs(out_dir: Path, shards: int) -> None:
    """Write the fold's curated positives as `shards` CSVs for scripts/predict.py."""
    frames = [pd.read_csv(FOLD_DIR / f"{split}.csv") for split in ("train", "val")]
    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.loc[pairs["labels"] == 1, PAIR_COLUMNS].reset_index(drop=True)
    if len(pairs) != EXPECTED_POSITIVES:
        raise ValueError(f"expected {EXPECTED_POSITIVES} positives, found {len(pairs)}")
    if pairs.duplicated(["RNA_id", "Protein_id"]).any():
        raise ValueError("duplicate (RNA_id, Protein_id) among positives")

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(np.array_split(np.arange(len(pairs)), shards)):
        path = out_dir / f"fold_1596_positives.part{k}.csv"
        pairs.iloc[idx].to_csv(path, index=False)
        LOGGER.info("wrote %s (%d pairs)", path, len(idx))


def read_fasta_ids(path: Path) -> set[str]:
    """Return the record ids (first token after '>') of a FASTA file."""
    with open(path) as fh:
        return {line[1:].split()[0] for line in fh if line.startswith(">")}


def write_fasta(proteins: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for pid, seq in proteins.itertuples(index=False):
            fh.write(f">{pid}\n{seq}\n")
    LOGGER.info("wrote %s (%d proteins)", path, len(proteins))


def write_universe(pred_dir: Path, fasta_out: Path, scanned_fasta: Path,
                   allow_unscanned: bool) -> None:
    """Keep predicted positives, write the analysis CSV and FASTA, check Pfam coverage.

    With allow_unscanned, proteins outside the scanned universe are written to
    <fasta_out>.unscanned.fa for hmmscan (run.sh step 3) instead of stopping the run.
    That file exists only when such proteins exist, so a stale one is removed first.
    """
    parts = sorted(pred_dir.glob("fold_1596_positives.part*.predicted.csv"))
    if not parts:
        raise FileNotFoundError(f"no shard predictions in {pred_dir}")
    pred = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if len(pred) != EXPECTED_POSITIVES:
        raise ValueError(f"expected {EXPECTED_POSITIVES} scored pairs, found {len(pred)}")
    if not (pred["true_label"] == 1).all():
        raise ValueError("scored pairs include non-positives")
    if not ((pred["interaction_probability"] > 0.5) == (pred["prediction"] == 1)).all():
        raise ValueError("prediction column disagrees with p > 0.5")
    pred.to_csv(pred_dir / "fold_1596_positives_predicted.csv", index=False)

    tp = pred.loc[pred["prediction"] == 1,
                  ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs", "prediction"]]
    tp.to_csv(pred_dir / "all_predicted_interactions.csv", index=False)
    LOGGER.info("true positives: %d / %d pairs (recall %.4f), %d proteins",
                len(tp), len(pred), len(tp) / len(pred), tp["Protein_id"].nunique())

    proteins = tp[["Protein_id", "Prot_seqs"]].drop_duplicates()
    if proteins["Protein_id"].duplicated().any():
        raise ValueError("a Protein_id maps to more than one sequence")
    write_fasta(proteins, fasta_out)

    # --- Every protein must have been scanned, or its RBD mask would be wrong ---
    unscanned_fasta = fasta_out.with_suffix(".unscanned.fa")
    unscanned_fasta.unlink(missing_ok=True)
    is_scanned = proteins["Protein_id"].isin(read_fasta_ids(scanned_fasta))
    if is_scanned.all():
        LOGGER.info("all %d proteins are in the scanned universe %s", len(proteins), scanned_fasta)
        return
    n_unscanned = int((~is_scanned).sum())
    if not allow_unscanned:
        raise SystemExit(f"{n_unscanned} proteins were not in {scanned_fasta}; run hmmscan on "
                         f"them first, or pass --allow-unscanned to write them out for it.")
    LOGGER.warning("%d proteins were not in %s; writing them for hmmscan", n_unscanned, scanned_fasta)
    write_fasta(proteins.loc[~is_scanned], unscanned_fasta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shards", type=int, default=2)
    u = sub.add_parser("universe")
    u.add_argument("--pred-dir", type=Path, required=True)
    u.add_argument("--fasta-out", type=Path, required=True)
    u.add_argument("--scanned-fasta", type=Path,
                   default=paths.RBD_DIR / "scanned_proteins.fa",
                   help="Every protein in data/rbd/scanned_proteins.domtbl's hmmscan run.")
    u.add_argument("--allow-unscanned", action="store_true",
                   help="Write unscanned proteins to <fasta-out>.unscanned.fa instead of stopping.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.step == "pairs":
        write_pairs(args.out_dir, args.shards)
    else:
        write_universe(args.pred_dir, args.fasta_out, args.scanned_fasta, args.allow_unscanned)


if __name__ == "__main__":
    main()

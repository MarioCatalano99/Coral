#!/usr/bin/env python3
"""
RBD analysis — Step 4 head screening.

The unit of analysis is the protein: inference runs per (RNA, protein) pair with
all 12 cross-attention heads, attention is marginalized over the RNA axis, and a
protein's per-pair per-head vectors are averaged into one consensus profile
before computing the per-head difference

    Δ_h = mean(attn_h | residue ∈ RBD) − mean(attn_h | residue ∉ RBD).

Then run Friedman across heads on the per-protein Δ_h and a Wilcoxon
best-vs-rest post-hoc with Benjamini–Hochberg FDR (delegated to
`coral.head_selection.select_heads_by_friedman_wilcoxon`).

Outputs (under results/<model>/):
    rbd_attention_heads_stats.csv   one row per (protein, head)
    selected_heads.json             {best_head, retained_heads,
                                     friedman_p, wilcoxon_padj}

Re-run head selection without re-running inference:
    python ... screen_attention_heads.py --from-csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run

from coral import paths
from coral.attention import AttentionAnalyzer
from coral.head_selection import select_heads_by_friedman_wilcoxon
from coral.rbd import load_rbd_intervals

import _pipeline as pipe  # sibling module, on sys.path[0] when run as script


def per_unit_head_stats(attentions_per_head: np.ndarray,
                        residue_mask: np.ndarray) -> List[dict]:
    """attentions_per_head: (n_heads, protein_len). residue_mask: (protein_len,) 0/1."""
    rbd = residue_mask == 1
    non_rbd = residue_mask == 0
    if rbd.sum() == 0 or non_rbd.sum() == 0:
        return []

    rows = []
    for h in range(attentions_per_head.shape[0]):
        attn = attentions_per_head[h]
        rows.append({
            "head_id": h,
            "n_rbd": int(rbd.sum()),
            "n_non_rbd": int(non_rbd.sum()),
            "rbd_mean": float(attn[rbd].mean()),
            "non_rbd_mean": float(attn[non_rbd].mean()),
            "mean_diff": float(attn[rbd].mean() - attn[non_rbd].mean()),
        })
    return rows


def collect_pair_head_tensors(predictions_csv: Path,
                              rbd_intervals: dict,
                              protein_summary: pd.DataFrame,
                              checkpoint_dir: str,
                              device: torch.device,
                              batch_size: int,
                              rna_max_length: int,
                              protein_max_length: int):
    """Inference loop. Returns (per_pair_attn_per_head, per_pair_masks,
    pair_ids, protein_ids), where each per-pair attn array has shape
    (n_heads, n_residues)."""
    pairs_df = pipe.filter_pairs(predictions_csv, rbd_intervals, protein_summary)
    print(f"Pairs after filtering: {len(pairs_df)} "
          f"(unique proteins: {pairs_df['Protein_id'].nunique()})")
    if pairs_df.empty:
        raise RuntimeError("No pairs left after filtering — check inputs.")

    rna_tok = AutoTokenizer.from_pretrained(str(paths.DNABERT_DIR), trust_remote_code=True)
    prot_tok = AutoTokenizer.from_pretrained(pipe.PROTEIN_ENCODER, trust_remote_code=True)

    dataloader = pipe.prepare_dataloader(
        pairs_df, rna_tok, prot_tok, rbd_intervals,
        batch_size=batch_size,
        max_rna_length=rna_max_length,
        max_protein_length=protein_max_length,
        device=str(device),
    )

    model = pipe.load_pretrained_model(checkpoint_dir, str(paths.DNABERT_DIR),
                                       pipe.PROTEIN_ENCODER, device)
    analyzer = AttentionAnalyzer(model, rna_tok, prot_tok, device=device)

    attentions: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    pair_ids: List[str] = []
    protein_ids: List[str] = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        rna_in = {"input_ids": batch["rna_input_ids"].to(device),
                  "attention_mask": batch["rna_attention_mask"].to(device)}
        prot_in = {"input_ids": batch["protein_input_ids"].to(device),
                   "attention_mask": batch["protein_attention_mask"].to(device)}

        out = analyzer.extract_cross_attention_scores(rna_in, prot_in, subset=None)
        rbd_labels_batch = batch["rbd_labels"].cpu().numpy()
        batch_pair_keys = batch["pair_keys"]
        batch_protein_keys = batch["protein_keys"]

        # AttentionAnalyzer returns (n_heads, rna_residues, protein_residues) when subset=None.
        for i, attn_full in enumerate(out["rna_to_protein_attentions"]):
            mask_full = rbd_labels_batch[i]
            mask = mask_full[mask_full != -100]
            n_res = min(attn_full.shape[-1], mask.shape[0])
            if n_res == 0:
                continue
            attn_per_head = attn_full[:, :, :n_res].mean(axis=1).astype(np.float32)  # (n_heads, n_res)
            attentions.append(attn_per_head)
            masks.append(mask[:n_res].astype(np.int8))
            pair_ids.append(batch_pair_keys[i])
            protein_ids.append(batch_protein_keys[i])

    return attentions, masks, pair_ids, protein_ids


def build_head_stats_df(attentions: List[np.ndarray],
                        masks: List[np.ndarray],
                        protein_ids: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for i, (attn_per_head, mask) in enumerate(zip(attentions, masks)):
        for row in per_unit_head_stats(attn_per_head, mask):
            row["protein_id"] = protein_ids[i]
            rows.append(row)

    cols = ["protein_id", "head_id", "n_rbd", "n_non_rbd",
            "rbd_mean", "non_rbd_mean", "mean_diff"]
    return pd.DataFrame(rows, columns=cols)


def run_head_selection(stats_df: pd.DataFrame, fdr_alpha: float) -> dict:
    delta = stats_df.pivot(index="protein_id", columns="head_id", values="mean_diff")
    n_units, n_heads = delta.shape
    print(f"\nDelta matrix: {n_units} proteins × {n_heads} heads")

    res = select_heads_by_friedman_wilcoxon(delta, fdr_alpha=fdr_alpha)

    print("\n" + "-" * 72)
    print(f"{'Head':>6}  {'Mean':>12}  {'Median':>12}  {'Std':>12}  {'Cohen d':>10}")
    print("-" * 72)
    for _, row in res["head_summary"].iterrows():
        print(f"{int(row['head']):>6}  {row['mean']:>12.6f}  {row['median']:>12.6f}  "
              f"{row['std']:>12.6f}  {row['cohens_d']:>10.6f}")

    print(f"\nFriedman χ²={res['friedman_chi2']:.1f}  p={res['friedman_p']:.3e}")
    if res["friedman_p"] < fdr_alpha and not res["posthoc"].empty:
        print(f"Best head (by median Δ): {int(res['best_head'])}")
        print("Best-vs-rest Wilcoxon (BH-adjusted):")
        for _, row in res["posthoc"].iterrows():
            sig = "***" if row["reject_H0"] else "n.s."
            print(f"  {int(row['head_A']):>3} vs {int(row['head_B']):>3}  "
                  f"W={row['W_stat']:>10.1f}  p_raw={row['p_raw']:.3e}  "
                  f"p_BH={row['p_adj_BH']:.3e}  {sig}")

    retained = sorted(int(h) for h in res["retained_heads"])
    print(f"\n✓ Selected heads: {retained}")
    return res


def write_selected_heads(res: dict, out_path: Path) -> None:
    payload = {
        "best_head": int(res["best_head"]),
        "selected": sorted(int(h) for h in res["retained_heads"]),
        "friedman_chi2": res["friedman_chi2"],
        "friedman_p": res["friedman_p"],
        "wilcoxon_padj": {
            int(row["head_B"]): float(row["p_adj_BH"])
            for _, row in res["posthoc"].iterrows()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path,
                    default=paths.PREDICTIONS_DIR / "val_1596_predicted.csv",
                    help="CSV with [RNA_id, Protein_id, RNA_seqs, Prot_seqs, prediction]")
    ap.add_argument("--min-partners", type=int, default=1,
                    help="Drop proteins backed by fewer TP partners than this.")
    ap.add_argument("--from-csv", action="store_true",
                    help="Skip inference and re-run head selection from cached stats.")
    ap.add_argument("--epoch", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rna-max-length", type=int, default=6000)
    ap.add_argument("--protein-max-length", type=int, default=1250)
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    args = ap.parse_args()

    pipe.seed_everything(42)

    results_dir = paths.analysis_results("RBD_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = results_dir / "rbd_attention_heads_stats.csv"
    selected_json = results_dir / "selected_heads.json"

    if args.from_csv:
        if not stats_csv.exists():
            raise FileNotFoundError(f"--from-csv set but {stats_csv} does not exist.")
        stats_df = pd.read_csv(stats_csv)
        print(f"Loaded cached stats: {len(stats_df)} rows from {stats_csv}")
    else:
        intervals_path = results_dir / "rbd_annotations.rbd_intervals.tsv"
        summary_path = results_dir / "rbd_annotations.protein_summary.tsv"
        rbd_intervals = load_rbd_intervals(intervals_path)
        protein_summary = pd.read_csv(summary_path, sep="\t")
        print(f"Loaded {len(rbd_intervals)} proteins with RBD intervals; "
              f"{int((protein_summary['passes_filters'] == 1).sum())} pass Step-2 filters.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Inference: always per-pair, all 12 heads ---
        pair_attn, pair_masks, pair_ids, pair_protein_ids = collect_pair_head_tensors(
            predictions_csv=args.predictions,
            rbd_intervals=rbd_intervals,
            protein_summary=protein_summary,
            checkpoint_dir=str(paths.checkpoint(args.epoch)),
            device=device,
            batch_size=args.batch_size,
            rna_max_length=args.rna_max_length,
            protein_max_length=args.protein_max_length,
        )

        # --- Aggregate pairs → per-protein consensus (helper keeps the head axis) ---
        attentions, masks, protein_ids, n_partners = pipe.aggregate_to_protein(
            pair_attn, pair_masks, pair_protein_ids,
        )
        attentions, masks, protein_ids, _ = pipe.drop_low_partner_proteins(
            attentions, masks, protein_ids, n_partners, args.min_partners,
        )
        print(f"Aggregated {len(pair_attn)} pairs → {len(protein_ids)} proteins.")

        stats_df = build_head_stats_df(attentions, masks, protein_ids)
        stats_df.to_csv(stats_csv, index=False)
        print(f"Wrote {stats_csv} ({len(stats_df)} rows)")

    res = run_head_selection(stats_df, fdr_alpha=args.fdr_alpha)
    write_selected_heads(res, selected_json)


if __name__ == "__main__":
    main()

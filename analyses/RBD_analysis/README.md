# RBD analysis (paper Figure 4)

Tests whether CORAL's protein-side cross-attention is higher inside Pfam-annotated RNA-binding domains (RBDs)
than outside them. The unit of analysis is the protein: for each pair, the RNA→protein attention of the last
cross-attention layer is averaged over the RNA axis, and each protein's pairs are averaged into one consensus
profile.

## Running

From the repository root, in the `coral` environment:

```bash
bash analyses/RBD_analysis/run.sh
```

It uses `checkpoints/coral-epoch-9`, the paper's final model as `scripts/download_weights.sh` installs it (Zenodo
weights v3); set `CKPT` to another checkpoint directory to use that instead. The run needs the benchmark splits in
`data/datasets/` (`scripts/download_data.sh`) and takes about 30 min on two GPUs.

| Step | Script | Does |
| --- | --- | --- |
| 1 | `scripts/prepare_inputs.py pairs`, `scripts/predict.py` | Scores the 25,202 positives of redundant fold 1596 (train + test). Skipped if done. |
| 2 | `scripts/prepare_inputs.py universe` | Keeps the pairs called interacting (p > 0.5) and writes their proteins' FASTA. Proteins outside `data/rbd/scanned_proteins.fa` are written to a separate FASTA for step 3. |
| 3 | `scripts/run_hmmscan.sh`, `scripts/parse_rbd_domtbl.py` | Maps Pfam hits of the 85 whitelisted families to merged RBD intervals and applies the protein filters. hmmscan runs only for proteins outside the scanned set. |
| 4 | `scripts/screen_attention_heads.py` | Friedman test across the 12 heads, then Wilcoxon best-vs-rest with BH; writes `selected_heads.json`. |
| 5 | `scripts/attention_analysis.py` | Δ per protein with the selected heads; one-sample t, sign test, bootstrap CI, per-protein Cohen's d. |
| 6 | `scripts/controls.py --heads best` | Position-shuffle null (1,000 rotations) and per-family effect sizes with BH. |
| 7 | `scripts/plot_figure6.py` | The four panels and the assembled figure. |

`scripts/_pipeline.py` holds the inference code shared by steps 4 and 5.

## Inputs

Committed in `data/rbd/`:

- `semiconservative_rbd2pfam.json`: the 85-family RBD whitelist (family name → Pfam accession). It keeps the
  Pfam domains of the Gerstberger et al. (2014) census (`Gerstberger_RBD_list.xlsx`) found in at least four
  RNA-binding proteins and present in `Gerstberger_RBP_list.xls`. Names were mapped to accessions with Pfam's
  `Pfam-A.clans.tsv`; seven renamed families were mapped by hand: Nol1_Nop2_Fmu → Methyltr_RsmB-F (PF01189),
  APOBEC_N → AID (PF08210), DZF → DZF_N (PF07528), HA2 → WHD_HA2 (PF04408) plus HA2_C (PF21010),
  PABP → MLLE (PF00658) and zf-RanBP → Zn_ribbon_RanBP (PF00641); RRM_6 is now part of RRM_1 (PF00076).
- `scanned_proteins.fa` and `scanned_proteins.domtbl`: 4,222 proteins and their `hmmscan --cut_ga` hits against
  Pfam-A (HMMER 3.3.2), made by `scripts/run_hmmscan.sh` in two runs that were concatenated: 3,814 proteins
  scanned for the preprint, and 408 that only the final model calls interacting. A rescan of 20 of the first set
  with the second setup reproduced their rows byte for byte. Every one of the 4,141 proteins in the analysis is
  in this set, so the paper's run needs neither Pfam nor HMMER. Pfam-A is needed only for a checkpoint that
  calls other proteins; set `PFAM` to a pressed `Pfam-A.hmm` (default `data/external/Pfam/Pfam-A.hmm`).

Written by the run (gitignored): `data/predictions/<checkpoint>/` (scores, `all_predicted_interactions.csv`) and
`data/sequences/<checkpoint>/` (the analysis FASTA).

## Outputs

In `results/` (gitignored; `CORAL_RESULTS_DIR` redirects it):

- `rbd_annotations.{rbd_hits,rbd_intervals,protein_summary}.tsv`: domain hits, merged intervals, protein filters.
- `rbd_attention_heads_stats.csv`, `selected_heads.json`, `results_head_analysis.txt`: the head screen.
- `heads_best/`: per-protein statistics, cached consensus vectors, the meta-analysis printout, the null
  distribution and the per-family table.
- `figures/figure6.{svg,jpg}`: the figure in the paper's layout (Figure 4 of the paper). The SVG nests the four
  panel SVGs, so text stays editable and only the heatmap tracks are raster. The panels are also saved alone in
  `figures/{svg,jpg}/`: `fig6a_heatmap`, `fig6b_delta`, `fig6c_ecdf`, `fig6d_forest`.

Panel d plots each family's effect size across proteins (mean Δ / SD of Δ), a different quantity from the
within-protein Cohen's d of panel c.

## Expected results

- 24,708 of the 25,202 pairs are called interacting, across 4,141 proteins; 669 carry a whitelisted RBD and 496
  pass the filters (12,760 pairs).
- The screen selects heads 5, 10 and 11 (Friedman χ²(11) = 906.0).
- 467/496 proteins have Δ > 0; mean Δ = 2.65 × 10⁻³ (bootstrap 95% CI 2.46–2.85 × 10⁻³), t(495) = 26.2.
  Median per-protein Cohen's d = 0.88.
- The observed mean Δ exceeds all 1,000 rotations of the domain annotation (33 null SDs above the null mean).
- 13 of the 14 families with at least ten proteins are positive after BH correction; PUA is the exception.

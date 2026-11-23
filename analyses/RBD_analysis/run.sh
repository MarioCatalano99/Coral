#!/usr/bin/env bash
# RBD analysis (paper Figure 4): protein-side cross-attention at Pfam-annotated RNA-binding domains.
#
# Scores the 25,202 positives of redundant fold 1596 with the final model, keeps the pairs it calls
# interacting (p > 0.5), annotates their proteins with the 85 whitelisted RBD families, screens the
# 12 heads, tests attention inside vs outside RBDs, runs the controls and draws the figure.
# Outputs: data/predictions/<checkpoint>/, data/sequences/<checkpoint>/ and results/ next to this script.
#
# Needs data/datasets/ (scripts/download_data.sh), the checkpoint, and GPUs: two for scoring (skipped
# if done), one (GPU, default 0) for the attention steps. hmmscan and Pfam-A are needed only if the
# checkpoint calls a protein outside data/rbd/scanned_proteins.fa, which the paper's model never does.
#
#   conda run --no-capture-output -n coral bash analyses/RBD_analysis/run.sh
#
# CKPT (default checkpoints/coral-epoch-9) selects another checkpoint directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
CKPT="${CKPT:-checkpoints/coral-epoch-9}"
NAME="$(basename "$CKPT")"
PRED=data/predictions/$NAME
FASTA=data/sequences/$NAME/predicted_interacting_proteins.fa
SCRIPTS=analyses/RBD_analysis/scripts
# Same rule as coral.paths.analysis_results("RBD_analysis"), which the Python steps use.
RES="${CORAL_RESULTS_DIR:+$CORAL_RESULTS_DIR/RBD_analysis}"
RES="${RES:-$HERE/results}"
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-0}"
mkdir -p "$RES/heads_best"

# --- 1. Score the fold-1596 positives, one shard per GPU (fp32); skipped if done ---
if [ ! -f "$PRED/fold_1596_positives.part1.predicted.csv" ]; then
  python $SCRIPTS/prepare_inputs.py pairs --out-dir "$PRED" --shards 2
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$k python scripts/predict.py --weights "$CKPT" --batch-size 4 \
      --input-csv "$PRED/fold_1596_positives.part$k.csv" \
      --output "$PRED/fold_1596_positives.part$k.predicted.csv" > "$PRED/predict.part$k.log" 2>&1 &
  done
  wait
fi

# --- 2. Predicted positives -> analysis pairs + protein FASTA ---
python $SCRIPTS/prepare_inputs.py universe --pred-dir "$PRED" --fasta-out "$FASTA" --allow-unscanned

# --- 3. RBD annotations; proteins outside the scanned set are scanned first ---
DOMTBL=data/rbd/scanned_proteins.domtbl
UNSCANNED="${FASTA%.fa}.unscanned.fa"
if [ -f "$UNSCANNED" ]; then
  bash $SCRIPTS/run_hmmscan.sh "$UNSCANNED" "$PRED/unscanned"
  cat "$DOMTBL" "$PRED/unscanned.domtbl" > "$PRED/combined.domtbl"
  DOMTBL=$PRED/combined.domtbl
fi
python $SCRIPTS/parse_rbd_domtbl.py --domtbl "$DOMTBL" --fasta "$FASTA" \
  --rbd-json data/rbd/semiconservative_rbd2pfam.json --out "$RES/rbd_annotations" 2> "$RES/rbd_annotations.log"

# --- 4. Head screen on the full cohort (Friedman + Wilcoxon/BH) ---
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPTS/screen_attention_heads.py --checkpoint "$CKPT" \
  --predictions "$PRED/all_predicted_interactions.csv" > "$RES/results_head_analysis.txt" 2>&1

# --- 5. Main test with the screened heads ---
CUDA_VISIBLE_DEVICES=$GPU python $SCRIPTS/attention_analysis.py --checkpoint "$CKPT" \
  --predictions "$PRED/all_predicted_interactions.csv" --selected-heads "$RES/selected_heads.json" \
  > "$RES/attention_analysis.log" 2>&1

# --- 6. Position-shuffle null and family stratification ---
python $SCRIPTS/controls.py --heads best > "$RES/heads_best/control_results.txt"

# --- 7. Figure: four panels and the assembled figure6.{svg,jpg} ---
python $SCRIPTS/plot_figure6.py

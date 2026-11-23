#!/usr/bin/env bash
# Score every LUAD inference RNA against IGF2BP1, IGF2BP3 and LIN28B with the
# checkpoint named in config.json (GPU; GPUS defaults to 2).
# Writes data/luad/luad_predictions/<checkpoint-name>/; an interrupted run resumes.
# Run from anywhere, inside the analysis environment:
#   conda run --no-capture-output -n coral \
#     bash analyses/luad_analysis/scripts/inputs/run_inference.sh
set -euo pipefail

ANALYSIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "$ANALYSIS/../.." && pwd)"
cd "$REPO"
CHECKPOINT="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$ANALYSIS/config.json")"
OUTPUT="data/luad/luad_predictions/$(basename "$CHECKPOINT")"
echo "checkpoint: $CHECKPOINT -> $OUTPUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python -B -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-2}" scripts/predict_all_vs_all.py \
    --weights "$CHECKPOINT" \
    --rnas data/luad/luad_rna_sequences.parquet \
    --proteins data/luad/luad_proteins.tsv \
    --output-dir "$OUTPUT" \
    --batch-size auto

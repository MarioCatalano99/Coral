#!/usr/bin/env bash
# Step 2 of the RBD analysis: scan predicted_interacting_proteins.fa against Pfam-A with hmmscan.
# Output is consumed by parse_rbd_domtbl.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/../../.."
DATA="$ROOT/analyses/RBD_analysis/data"
SEQS="$ROOT/data/sequences"
PFAM="$ROOT/data/external/Pfam/Pfam-A.hmm"

hmmscan --cut_ga --cpu 8 \
    --domtblout "$DATA/predicted_interacting_proteins.domtbl" \
    "$PFAM" "$SEQS/predicted_interacting_proteins.fa" \
    > "$DATA/predicted_interacting_proteins.hmmscan.txt"

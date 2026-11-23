#!/usr/bin/env bash
# Scan a protein FASTA against Pfam-A with hmmscan at the curated gathering thresholds.
# data/rbd/scanned_proteins.domtbl was made with this command (HMMER 3.3.2) from
# data/rbd/scanned_proteins.fa; run.sh calls it only for proteins outside that set.
#
#   PFAM=/path/to/Pfam-A.hmm bash analyses/RBD_analysis/scripts/run_hmmscan.sh <proteins.fa> <out-prefix>
#
# Writes <out-prefix>.domtbl, read by parse_rbd_domtbl.py, and <out-prefix>.hmmscan.txt.
# PFAM defaults to data/external/Pfam/Pfam-A.hmm, which must be hmmpress-ed.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <proteins.fa> <out-prefix>" >&2
    exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PFAM="${PFAM:-$REPO/data/external/Pfam/Pfam-A.hmm}"

hmmscan --cut_ga --cpu "${CPUS:-8}" --domtblout "$2.domtbl" "$PFAM" "$1" > "$2.hmmscan.txt"

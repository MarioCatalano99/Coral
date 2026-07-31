#!/usr/bin/env bash
# Download the trained CORAL weights from Zenodo into checkpoints/coral-epoch-9/.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/checkpoints/coral-epoch-9"
# Concept DOI: always resolves to the newest version of the record.
# Versions before v2 hold a checkpoint that does not match the published
# architecture (LoRA r=8, no reverse_projection) — do not pin to them.
CONCEPT_DOI="10.5281/zenodo.19729892"
URL="https://zenodo.org/doi/${CONCEPT_DOI}/files/coral-epoch-9.tar.gz"

mkdir -p "${TARGET}"
echo "Downloading CORAL weights from ${URL}..."
curl -L -o "${TARGET}.tar.gz" "${URL}"
tar -xzf "${TARGET}.tar.gz" -C "${ROOT}/checkpoints"
rm "${TARGET}.tar.gz"
echo "Done. Weights extracted to ${TARGET}"

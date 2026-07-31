#!/usr/bin/env bash
# Download the CORAL benchmark splits from Zenodo into data/datasets/.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/data/datasets"
# Concept DOI: always resolves to the newest version of the record.
CONCEPT_DOI="10.5281/zenodo.19730809"
URL="https://zenodo.org/doi/${CONCEPT_DOI}/files/coral-benchmarks.tar.gz"

mkdir -p "${TARGET}"
echo "Downloading CORAL benchmark splits from ${URL}..."
curl -L -o "${TARGET}.tar.gz" "${URL}"
tar -xzf "${TARGET}.tar.gz" -C "${ROOT}/data"
rm "${TARGET}.tar.gz"
echo "Done. Datasets extracted to ${TARGET}"

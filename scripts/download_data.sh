#!/usr/bin/env bash
# Download the CORAL benchmark splits from Zenodo into Data/datasets/.
#
# TODO: <zenodo-data-url>  (replace the URL below when the Zenodo record is minted)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/Data/datasets"
URL="https://zenodo.org/record/TODO/files/coral-benchmarks.tar.gz"

mkdir -p "${TARGET}"
echo "Downloading CORAL benchmark splits from ${URL}..."
curl -L -o "${TARGET}.tar.gz" "${URL}"
tar -xzf "${TARGET}.tar.gz" -C "${ROOT}/Data"
rm "${TARGET}.tar.gz"
echo "Done. Datasets extracted to ${TARGET}"

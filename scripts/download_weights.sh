#!/usr/bin/env bash
# Download the trained CORAL weights from Zenodo into weights/coral-epoch-9/.
#
# TODO: <zenodo-weights-url>  (replace the URL below when the Zenodo record is minted)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/weights/coral-epoch-9"
URL="https://zenodo.org/record/TODO/files/coral-epoch-9.tar.gz"

mkdir -p "${TARGET}"
echo "Downloading CORAL weights from ${URL}..."
curl -L -o "${TARGET}.tar.gz" "${URL}"
tar -xzf "${TARGET}.tar.gz" -C "${ROOT}/weights"
rm "${TARGET}.tar.gz"
echo "Done. Weights extracted to ${TARGET}"

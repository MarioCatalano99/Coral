#!/usr/bin/env bash
# Download the trained CORAL weights from Zenodo into checkpoints/coral-epoch-9/.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/checkpoints/coral-epoch-9"
ARCHIVE="coral-epoch-9.tar.gz"
# Concept DOI: always resolves to the newest version of the record.
# Versions before v2 hold a checkpoint that does not match the published
# architecture (LoRA r=8, no reverse_projection) — do not pin to them.
CONCEPT_DOI="10.5281/zenodo.19729892"

# --- Resolve the concept DOI to the newest version's record --- #
# Zenodo redirects /doi/<doi> to /records/<id> but does not serve files under
# the /doi/ path, so the file URL has to be built from the resolved record.
RECORD="$(curl -fsSL "https://zenodo.org/doi/${CONCEPT_DOI}" -o /dev/null -w '%{url_effective}')"
case "${RECORD}" in
    https://zenodo.org/records/*) ;;
    *) echo "Unexpected redirect for ${CONCEPT_DOI}: ${RECORD}" >&2; exit 1 ;;
esac
URL="${RECORD}/files/${ARCHIVE}"

mkdir -p "${ROOT}/checkpoints"
echo "Downloading CORAL weights from ${URL}..."
curl -fL -o "${TARGET}.tar.gz" "${URL}"
tar -xzf "${TARGET}.tar.gz" -C "${ROOT}/checkpoints"
rm "${TARGET}.tar.gz"
echo "Done. Weights extracted to ${TARGET}"

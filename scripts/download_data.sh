#!/usr/bin/env bash
# Download the CORAL benchmark splits from Zenodo into data/datasets/.
#
# The newest version holds the degree-corrected splits of the paper (L = 50):
# redundant_split, pairwise_splits_60_80 and componentwise_splits_60_80, five
# folds each (v3, 10.5281/zenodo.24466337). The splits with uniformly sampled
# negatives (Supplementary Figure 2) stay in v1, 10.5281/zenodo.19730810.
# v2 holds a truncated archive; do not pin to it.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/data/datasets"
ARCHIVE="coral-benchmarks.tar.gz"
# Concept DOI: always resolves to the newest version of the record.
CONCEPT_DOI="10.5281/zenodo.19730809"

# --- Refuse to extract over an existing copy --- #
# Versions differ in which splits they contain, so extracting over an older
# copy would overwrite some folds and silently keep others.
if [ -d "${TARGET}" ] && [ -n "$(ls -A "${TARGET}")" ]; then
    echo "${TARGET} already exists. Move it aside and rerun to fetch the current splits." >&2
    exit 1
fi

# --- Resolve the concept DOI to the newest version's record --- #
# Zenodo redirects /doi/<doi> to /records/<id> but does not serve files under
# the /doi/ path, so the file URL has to be built from the resolved record.
RECORD="$(curl -fsSL "https://zenodo.org/doi/${CONCEPT_DOI}" -o /dev/null -w '%{url_effective}')"
case "${RECORD}" in
    https://zenodo.org/records/*) ;;
    *) echo "Unexpected redirect for ${CONCEPT_DOI}: ${RECORD}" >&2; exit 1 ;;
esac
URL="${RECORD}/files/${ARCHIVE}"

mkdir -p "${ROOT}/data"
echo "Downloading CORAL benchmark splits from ${URL}..."
curl -fL -o "${TARGET}.tar.gz" "${URL}"
tar -xzf "${TARGET}.tar.gz" -C "${ROOT}/data"
rm "${TARGET}.tar.gz"
echo "Done. Datasets extracted to ${TARGET}"

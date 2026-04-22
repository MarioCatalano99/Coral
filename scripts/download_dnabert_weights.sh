#!/usr/bin/env bash
# Download DNABERT-2-117M pytorch_model.bin (~447 MB) from HuggingFace into
# DNABERT-2-117M/. The model *code* (bert_layers.py, bert_padding.py,
# flash_attn_triton.py, configuration_bert.py, tokenizer files, ...) is
# committed to this repository with patches required for newer PyTorch
# releases, so only the weights file is fetched here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT}/DNABERT-2-117M"
TARGET_FILE="${TARGET_DIR}/pytorch_model.bin"
URL="https://huggingface.co/zhihan1996/DNABERT-2-117M/resolve/main/pytorch_model.bin"

if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "error: ${TARGET_DIR} does not exist — did you clone the CORAL repository?" >&2
    exit 1
fi

if [[ -f "${TARGET_FILE}" ]]; then
    echo "DNABERT-2 weights already present at ${TARGET_FILE} — skipping download."
    exit 0
fi

echo "Downloading DNABERT-2 weights from ${URL}..."
curl -L --fail -o "${TARGET_FILE}" "${URL}"
echo "Done. Weights saved to ${TARGET_FILE}"

"""Shared paths, configuration and transactional output for LUAD commands."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from coral import paths
from coral.prediction_io import read_predictions, sha256_file

REPO_ROOT = paths.REPO_ROOT
ANALYSIS_ROOT = paths.ANALYSES_DIR / "luad_analysis"
RESULTS_ROOT = Path(os.environ["CORAL_RESULTS_DIR"]) / "luad_analysis" if os.environ.get("CORAL_RESULTS_DIR") else ANALYSIS_ROOT
DATA_ROOT = RESULTS_ROOT / "data" if os.environ.get("CORAL_RESULTS_DIR") else paths.DATA_DIR / "TCGA"
CONFIG_PATH = ANALYSIS_ROOT / "config/expression_context.json"
LOGGER = logging.getLogger(__name__)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the single authoritative analysis configuration."""
    config = json.loads(path.read_text())
    for key in ("rbp_tpm_cutoff", "target_tpm_cutoff", "sensitivity_tpm_cutoff", "expression_pseudocount", "active_count_pseudocount"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("prevalence_fraction", "top_fraction", "bh_alpha"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if config["sensitivity_tpm_cutoff"] > config["target_tpm_cutoff"]:
        raise ValueError("sensitivity cutoff must not exceed primary cutoff")
    if config["schema_version"] != 3 or config["wilcoxon"]["permutation_resamples"] < 999:
        raise ValueError("unsupported schema or insufficient permutation resamples")
    if config["wilcoxon"]["zero_method"] != "pratt" or config["wilcoxon"]["alternative"] != "two-sided":
        raise ValueError("this analysis uses two-sided Pratt signed-rank inference")
    return config


def git_context() -> dict[str, Any]:
    """Record current git state without silently hiding unavailable provenance."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        LOGGER.warning("Cannot record git state: %s", error)
        return {"commit": None, "dirty": None}


def provenance(inputs: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    """Record inputs and all local analysis code, including shared dependencies."""
    code = sorted((ANALYSIS_ROOT / "scripts").glob("*.py")) + [REPO_ROOT / "src/coral/prediction_io.py", REPO_ROOT / "src/coral/paths.py"]
    return {"schema_version": 3, "config": config, "git": git_context(),
            "inputs": {str(p.resolve()): sha256_file(p) for p in inputs},
            "code": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in code},
            "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scipy", "statsmodels")}}}


def verify_manifest(path: Path, root: Path) -> dict[str, Any]:
    """Validate every output in a relocatable completed-stage manifest."""
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"stage is incomplete: {path}")
    for name, digest in manifest.get("inputs", {}).items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"stage input changed since publication: {name}")
    for name, item in manifest["outputs"].items():
        if sha256_file(root / name) != item["sha256"]:
            raise ValueError(f"output hash mismatch: {root / name}")
    return manifest


def read_exposure(data_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Validate the complete audit and its current sequence and eligibility inputs."""
    directory = data_dir / "luad_training_exposure"
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("training audit is incomplete")
    for name, digest in manifest["outputs"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"training audit hash mismatch: {name}")
    for name, field in (("luad_rna_sequences.parquet", "rna_sha256"),
                        ("luad_proteins.tsv", "protein_sha256"),
                        ("luad_eligibility.parquet", "eligibility_sha256")):
        if manifest["configuration"][field] != sha256_file(data_dir / name):
            raise ValueError("training audit refers to different inputs")
    frame = pd.read_parquet(directory / "exposure.parquet")
    if frame[["RNA_id", "Protein_id"]].duplicated().any() or frame.exposure_class.isna().any():
        raise ValueError("training audit has duplicate or unclassified pairs")
    return frame, path


@contextmanager
def staged_outputs(root: Path, manifest_path: Path, metadata: dict[str, Any]) -> Iterator[Path]:
    """Stage and validate outputs, then publish with the manifest as commit marker.

    An incomplete marker replaces any old manifest before publication. A failure
    during publication therefore cannot leave a falsely complete mixed run.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock = manifest_path.with_suffix(".lock")
    with lock.open("x"):
        try:
            with tempfile.TemporaryDirectory(prefix=".luad-stage-", dir=root) as temporary:
                stage = Path(temporary)
                yield stage
                outputs = {}
                for file in sorted(stage.rglob("*")):
                    if not file.is_file():
                        continue
                    rows = None
                    if file.suffix in {".tsv", ".parquet"}:
                        frame = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file, sep="\t")
                        rows = len(frame)
                    outputs[str(file.relative_to(stage))] = {"sha256": sha256_file(file), "rows": rows}
                marker = stage / ".manifest.json"
                marker.write_text(json.dumps({**metadata, "status": "publishing"}, indent=2, allow_nan=False) + "\n")
                os.replace(marker, manifest_path)
                for name in outputs:
                    destination = root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(stage / name, destination)
                marker.write_text(json.dumps({**metadata, "status": "complete", "outputs": outputs}, indent=2, allow_nan=False) + "\n")
                os.replace(marker, manifest_path)
        finally:
            lock.unlink()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write and round-trip a staged table without reducing numerical precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False, compression="zstd")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_parquet(path))
    else:
        frame.to_csv(path, sep="\t", index=False)
        reopened = pd.read_csv(path, sep="\t", float_precision="round_trip")
        if list(reopened.columns) != list(frame.columns) or len(reopened) != len(frame):
            raise ValueError(f"staged table does not round-trip: {path}")
        numeric = frame.select_dtypes(include="number").columns
        if len(numeric) and not np.allclose(frame[numeric].to_numpy(dtype=float, na_value=np.nan), reopened[numeric].to_numpy(float), equal_nan=True):
            raise ValueError(f"staged numerical values changed: {path}")

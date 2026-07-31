"""Build Figure-5 panel-b PyMOL sessions: attention-colored RNA-protein complexes.

For each exemplar complex we map the saved cross-attention
(`results/<model>/<heads>/contact_vectors.npz`) onto the PDB structure, colour
each residue by its attention, draw thin lines for the top-N% attention cells,
and write one `.pse` session. A companion notebook then picks a camera angle and
renders JPEGs.

The score->structure mapping (validated during planning):

  * Attention matrix is (n_rna_tokens, n_protein_residues), 0-indexed, with
    CLS/SEP already stripped.
  * Protein column j  <->  the j-th protein residue in chain order. The model
    protein sequence (CSV `Prot_seqs`) equals the chain's ordered ATOM residues
    exactly, so we walk residues in order (author numbering has gaps -- never add
    a fixed offset) and assert the reconstructed sequence matches.
  * RNA token t  <->  a contiguous nucleotide span from the DNABERT-2 tokenizer
    (token strings tile the RNA). We broadcast a token's score to its
    nucleotides; the line endpoint for a token is its central nucleotide.

Run from repo root with the `encodersenv` conda env active:
    python analyses/contact_analysis/scripts/build_panelb_sessions.py [--top-percent 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
import pymol2

from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable so we can use `coral.*` regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run

from coral.paths import (  # noqa: E402
    DNABERT_DIR,
    PDB_DIR,
    PREDICTIONS_DIR,
    analysis_results,
)

log = logging.getLogger("build_panelb")

# --- Defaults + style ------------------------------------------------------ #

DEFAULT_HEADS = "heads_best"
DEFAULT_EXEMPLARS = ["4QIL|C_A", "3MOJ|A_B", "2UWM|C_A"]

# Sequential heat ramp: low attention -> white, high -> deepblue.
COLOR_RAMP = "white palecyan lightblue marine blue deepblue"
LINE_COLOR = "grey40"

RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
# RNA residue names -> one-letter, in DNA alphabet (U->T) to match model input.
RNA_TO_DNA = {"A": "A", "U": "T", "G": "G", "C": "C"}
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}


# --- Inputs ---------------------------------------------------------------- #

def parse_complex_id(cid: str) -> tuple[str, str, str]:
    """'4QIL|C_A' -> ('4QIL', 'C', 'A')  (pdb, rna_chain, prot_chain)."""
    pdb, rest = cid.split("|")
    rna_ch, prot_ch = rest.split("_")
    return pdb, rna_ch, prot_ch


def load_attention(npz_path: Path, cid: str) -> np.ndarray:
    """Return the (n_rna_tokens, n_protein_residues) attention matrix for `cid`."""
    data = np.load(npz_path, allow_pickle=True)
    ids = list(data["complex_ids"])
    if cid not in ids:
        raise KeyError(f"{cid} not in {npz_path.name} (has {len(ids)} complexes)")
    return np.asarray(data["attentions"][ids.index(cid)], dtype=np.float64)


def get_model_sequences(df: pd.DataFrame, pdb: str, rna_ch: str, prot_ch: str
                        ) -> tuple[str, str]:
    """Look up the exact (RNA in T-space, protein) sequences fed to the model."""
    rna_id, prot_id = f"{pdb}|{rna_ch}", f"{pdb}|{prot_ch}"
    row = df[(df["RNA_id"] == rna_id) & (df["Protein_id"] == prot_id)]
    if row.empty:
        raise KeyError(f"no prediction row for RNA_id={rna_id} Protein_id={prot_id}")
    r = row.iloc[0]
    return str(r["RNA_seqs"]).upper(), str(r["Prot_seqs"]).upper()


def tokenize_spans(tokenizer, rna_seq: str) -> list[tuple[int, int]]:
    """DNABERT-2 BPE spans tiling `rna_seq`: [(start, end), ...] per token."""
    ids = tokenizer(rna_seq)["input_ids"]
    toks = [t for t in tokenizer.convert_ids_to_tokens(ids)
            if t not in SPECIAL_TOKENS]
    spans, pos = [], 0
    for t in toks:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    if pos != len(rna_seq):
        raise ValueError(f"token spans cover {pos} nt but RNA is {len(rna_seq)} nt")
    return spans


def ensure_pdb(pdb: str) -> Path:
    """Return the local PDB path, downloading from RCSB if absent."""
    path = PDB_DIR / "single_pairs_pdb_files" / f"{pdb}.pdb"
    if not path.exists():
        log.info("  %s not on disk -> fetching from RCSB", pdb)
        urllib.request.urlretrieve(RCSB_URL.format(pdb=pdb), path)
    return path


# --- PyMOL helpers --------------------------------------------------------- #

def ordered_residue_atoms(cmd, sel: str) -> tuple[list[int], str]:
    """Atom `index` per residue (in chain order) + reconstructed resname string.

    `sel` should pick one anchor atom per residue (CA for protein, C1' for RNA).
    Returns (indices, resnames) where resnames is the raw 3/1-letter codes joined
    with '|' for diagnostics.
    """
    rows: list[tuple[int, str]] = []
    cmd.iterate(sel, "rows.append((index, resn))", space={"rows": rows})
    return [r[0] for r in rows], "|".join(r[1] for r in rows)


def reconstruct_protein(resnames: str) -> str:
    return "".join(AA3TO1.get(r, "?") for r in resnames.split("|") if r)


def reconstruct_rna(resnames: str) -> str:
    return "".join(RNA_TO_DNA.get(r, "?") for r in resnames.split("|") if r)


def set_bfactors(cmd, anchor_indices: list[int], scores: np.ndarray) -> None:
    """Write per-residue `scores` into the B-factor of each whole residue."""
    for k, idx in enumerate(anchor_indices):
        cmd.alter(f"byres (index {idx})", f"b={float(scores[k])}")


# --- Per-complex builder --------------------------------------------------- #

def build_session(cid: str, att: np.ndarray, rna_seq: str, prot_seq: str,
                  spans: list[tuple[int, int]], pdb_path: Path,
                  top_percent: float, out_dir: Path) -> dict:
    """Build one attention-coloured PyMOL session; return a sidecar score dict."""

    pdb, rna_ch, prot_ch = parse_complex_id(cid)
    n_tok, n_res = att.shape
    if n_tok != len(spans):
        raise ValueError(f"{cid}: {n_tok} attention tokens vs {len(spans)} spans")

    # --- Per-residue scores: max over the partner axis (per-molecule scale). ---
    prot_score = att.max(axis=0)                      # (n_res,)
    rna_tok_score = att.max(axis=1)                   # (n_tok,)
    token_of = np.empty(len(rna_seq), dtype=int)
    for ti, (s, e) in enumerate(spans):
        token_of[s:e] = ti
    rna_nt_score = rna_tok_score[token_of]            # (len(rna_seq),)

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(str(pdb_path), "cplx")

        # --- Keep only the analysed RNA + protein chains. ---
        cmd.remove("solvent")
        cmd.remove(f"cplx and not (chain {rna_ch} or chain {prot_ch})")

        # --- Ordered residue anchors (one altloc only) + sequence checks. ---
        # `elem C` on CA excludes calcium ions (resn CA, atom name CA, elem Ca);
        # `resn A+U+G+C` keeps only standard ribonucleotides (drops modified 5'
        # caps like GTP that the model's sequence omits).
        prot_idx, prot_res = ordered_residue_atoms(
            cmd, f"chain {prot_ch} and name CA and elem C and (alt ''+A)")
        rna_idx, rna_res = ordered_residue_atoms(
            cmd, f"chain {rna_ch} and resn A+U+G+C and name C1' and (alt ''+A)")

        got_prot, got_rna = reconstruct_protein(prot_res), reconstruct_rna(rna_res)
        if got_prot != prot_seq:
            raise AssertionError(
                f"{cid}: protein seq mismatch\n  PDB  ({len(got_prot)}): {got_prot}\n"
                f"  CSV  ({len(prot_seq)}): {prot_seq}")
        if got_rna != rna_seq:
            raise AssertionError(
                f"{cid}: RNA seq mismatch\n  PDB  ({len(got_rna)}): {got_rna}\n"
                f"  CSV  ({len(rna_seq)}): {rna_seq}")
        if len(prot_idx) != n_res:
            raise AssertionError(f"{cid}: {len(prot_idx)} residues vs {n_res} cols")
        log.info("  %s: sequence match OK (prot=%d, rna=%d, tokens=%d)",
                 cid, n_res, len(rna_seq), n_tok)

        # --- Attention -> B-factors. ---
        cmd.alter("all", "b=0.0")
        set_bfactors(cmd, prot_idx, prot_score)
        set_bfactors(cmd, rna_idx, rna_nt_score)
        cmd.sort()

        # --- Colour each molecule on its own scale (max already applied). ---
        def _rng(v: np.ndarray) -> tuple[float, float]:
            lo, hi = float(v.min()), float(v.max())
            return lo, (hi if hi > lo else lo + 1e-9)

        lo_p, hi_p = _rng(prot_score)
        lo_r, hi_r = _rng(rna_nt_score)
        cmd.spectrum("b", COLOR_RAMP, f"chain {prot_ch}", minimum=lo_p, maximum=hi_p)
        cmd.spectrum("b", COLOR_RAMP, f"chain {rna_ch}", minimum=lo_r, maximum=hi_r)

        # --- Representation + scene. ---
        cmd.hide("everything")
        cmd.show("cartoon", "polymer")
        #cmd.set("cartoon_ring_mode", 3)        # filled base rings for RNA
        cmd.set("cartoon_ring_finder", 1)
        cmd.set("cartoon_transparency", 0.0)
        cmd.bg_color("white")
        cmd.set("ray_shadows", 0)
        cmd.set("antialias", 2)
        cmd.set("ray_opaque_background", 1)

        # --- Top-N% attention lines: protein CA <-> RNA backbone of central nt. ---
        # End the RNA side on the sugar-phosphate backbone (the cartoon "main
        # frame"), not the C1'/base junction: prefer the phosphate, falling back
        # toward the 5'-sugar atoms for the 5'-terminal residue that lacks a P.
        # `first (...)` over the priority-ordered names yields exactly one atom.
        thr = float(np.percentile(att, 100.0 - top_percent))
        cells = np.argwhere(att >= thr)            # (k, 2): [token, residue]
        for k, (t, j) in enumerate(cells):
            s, e = spans[t]
            n_mid = (s + e - 1) // 2
            rna_bb = f"first ((byres index {rna_idx[n_mid]}) and name P+O5'+C5'+C4'+C3')"
            cmd.distance(f"att_line_{k}", rna_bb, f"index {prot_idx[j]}")
        cmd.group("attention_lines", "att_line_*")
        cmd.set("dash_gap", 0.0)
        cmd.set("dash_radius", 0.06)
        cmd.set("dash_color", LINE_COLOR)
        cmd.hide("labels")

        cmd.orient()
        out_dir.mkdir(parents=True, exist_ok=True)
        pse_path = out_dir / f"{cid.replace('|', '_')}.pse"
        cmd.save(str(pse_path))
        log.info("  %s: %d top-%.1f%% lines (thr=%.4g) -> %s",
                 cid, len(cells), top_percent, thr, pse_path.name)

    return {
        "complex_id": cid, "pdb": pdb, "rna_chain": rna_ch, "prot_chain": prot_ch,
        "n_rna_tokens": int(n_tok), "n_protein_residues": int(n_res),
        "top_percent": top_percent, "threshold": thr, "n_lines": int(len(cells)),
        "prot_score": prot_score.round(6).tolist(),
        "rna_token_score": rna_tok_score.round(6).tolist(),
        "rna_nt_score": rna_nt_score.round(6).tolist(),
        "prot_range": [lo_p, hi_p], "rna_range": [lo_r, hi_r],
    }


# --- Entry point ----------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", default=DEFAULT_HEADS,
                    help="contact_vectors subdir (heads_best | heads_all)")
    ap.add_argument("--exemplars", nargs="+", default=DEFAULT_EXEMPLARS)
    ap.add_argument("--top-percent", type=float, default=1.0,
                    help="draw lines for the top N%% of attention cells")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default results/<model>/panel_b/sessions)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = analysis_results("contact_analysis")
    npz_path = results_dir / args.heads / "contact_vectors.npz"
    out_dir = args.out or (results_dir / "panel_b" / "sessions")

    # --- Shared inputs loaded once. ---
    from transformers import AutoTokenizer
    log.info("Loading DNABERT-2 tokenizer + predictions ...")
    tokenizer = AutoTokenizer.from_pretrained(str(DNABERT_DIR), trust_remote_code=True)
    df = pd.read_csv(PREDICTIONS_DIR / "pdb_predicted_interactions.csv")

    sidecars = {}
    for cid in args.exemplars:
        log.info("=== %s ===", cid)
        pdb, rna_ch, prot_ch = parse_complex_id(cid)
        att = load_attention(npz_path, cid)
        rna_seq, prot_seq = get_model_sequences(df, pdb, rna_ch, prot_ch)
        spans = tokenize_spans(tokenizer, rna_seq)
        pdb_path = ensure_pdb(pdb)
        sidecars[cid] = build_session(
            cid, att, rna_seq, prot_seq, spans, pdb_path, args.top_percent, out_dir)
        sidecar_path = out_dir / f"{cid.replace('|', '_')}.scores.json"
        sidecar_path.write_text(json.dumps(sidecars[cid], indent=2))

    log.info("Done. %d sessions in %s", len(sidecars), out_dir)


if __name__ == "__main__":
    main()

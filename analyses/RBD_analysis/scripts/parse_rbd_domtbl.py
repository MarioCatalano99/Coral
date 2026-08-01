#!/usr/bin/env python3
"""
Parse HMMER hmmscan --domtblout against Pfam-A, filter to a curated RBD list,
merge overlapping intervals per protein, and emit three tables.

Outputs (TSV, 1-based inclusive coordinates throughout):

  <out>.rbd_hits.tsv        long format, one row per filtered domain hit
                            -> use this for per-domain analyses, QC plots,
                            head-screening sanity checks

  <out>.rbd_intervals.tsv   merged intervals per protein (overlapping/touching
                            hits collapsed into a single residue span)
                            -> use this as the RBD residue mask for Step 5

  <out>.protein_summary.tsv one row per FASTA protein with length, RBD residue
                            count, coverage fraction, dominant RBD family, and
                            Step-2 filter flags (length, coverage)
                            -> use this to apply the Step-2 protein filters

Note: the third Step-2 filter ("≥1 interaction with predicted prob > 0.5")
needs your CORAL prediction table and is intentionally NOT applied here as
predicted_interacting_proteins.fa already contains only proteins that satisfy
the threshold

Example (run from repo root):
  bash analyses/RBD_analysis/scripts/run_hmmscan.sh

  python analyses/RBD_analysis/scripts/parse_rbd_domtbl.py \\
      --domtbl data/rbd/predicted_interacting_proteins.domtbl \\
      --fasta data/sequences/predicted_interacting_proteins.fa \\
      --rbd-json data/rbd/semiconservative_rbd2pfam.json \\
      --out analyses/RBD_analysis/results/rbd_annotations
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------- #
# I/O helpers
# ---------------------------------------------------------------------------- #

def parse_fasta_lengths(fasta_path):
    """Return {protein_id: length} keyed by the first whitespace token after '>'."""
    lengths = {}
    cur_id, cur_len = None, 0
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    lengths[cur_id] = cur_len
                cur_id = line[1:].split()[0]
                cur_len = 0
            else:
                cur_len += len(line)
        if cur_id is not None:
            lengths[cur_id] = cur_len
    return lengths


def strip_pfam_version(acc):
    """'PF00076.24' -> 'PF00076'.  Robust to already-stripped input."""
    return acc.split(".", 1)[0]


def parse_domtbl(domtbl_path, allowed_accs):
    """
    Yield filtered hit dicts from an hmmscan --domtblout file.

    domtblout has 22 fixed whitespace-delimited fields followed by a free-form
    description that may contain spaces; we split with maxsplit=22.

    Field index (1-based, per HMMER docs):
       1 target_name      Pfam family name (e.g. RRM_1)
       2 target_accession Pfam accession with version (e.g. PF00076.24)
       3 tlen             HMM length
       4 query_name       protein ID (matches FASTA header token)
       5 query_accession  usually '-'
       6 qlen             protein length (aa)
       7 full_evalue      whole-sequence E-value
       8 full_score       whole-sequence bit score
       9 full_bias        whole-sequence bias
      10 dom_idx          this domain's index for this (target, query)
      11 dom_total        total # domains for this (target, query)
      12 c_evalue         conditional E-value (this domain)
      13 i_evalue         independent E-value (this domain)
      14 dom_score        bit score (this domain)
      15 dom_bias         bias (this domain)
      16 hmm_from / 17 hmm_to     HMM coordinates
      18 ali_from / 19 ali_to     alignment coordinates on the protein
      20 env_from / 21 env_to     envelope coordinates on the protein
      22 acc              mean posterior probability of aligned residues
      23+ description     (rest of line)
    """
    with open(domtbl_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(None, 22)
            if len(parts) < 22:
                continue

            tname, tacc, _tlen, qname, _qacc, qlen, \
                _full_e, _full_s, _full_b, \
                _dom_i, _dom_n, _c_e, i_e, dom_s, _dom_b, \
                _hmm_f, _hmm_t, ali_f, ali_t, env_f, env_t, post = parts[:22]

            acc_short = strip_pfam_version(tacc)
            if acc_short not in allowed_accs:
                continue

            yield {
                "protein_id": qname,
                "qlen": int(qlen),
                "pfam_name": tname,
                "pfam_acc": acc_short,
                "i_evalue": float(i_e),
                "dom_score": float(dom_s),
                "ali_from": int(ali_f),
                "ali_to": int(ali_t),
                "env_from": int(env_f),
                "env_to": int(env_t),
                "acc_post": float(post),
            }


# ---------------------------------------------------------------------------- #
# Interval merging (1-based inclusive)
# ---------------------------------------------------------------------------- #

def merge_intervals(intervals):
    """
    intervals: list of (start, end, label) with 1-based inclusive coordinates.

    Merge intervals where next.start <= current.end + 1, i.e. overlapping
    *or touching* spans collapse — this is the right convention for closed
    integer intervals when the goal is a residue set: residue 20 in family A
    plus residue 21 in family B together cover residues 20..21 contiguously.

    Returns: list of (merged_start, merged_end, [contributing_labels...]).
    """
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = []
    cur_s, cur_e, cur_labels = sorted_iv[0][0], sorted_iv[0][1], [sorted_iv[0][2]]
    for s, e, lab in sorted_iv[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
            cur_labels.append(lab)
        else:
            merged.append((cur_s, cur_e, cur_labels))
            cur_s, cur_e, cur_labels = s, e, [lab]
    merged.append((cur_s, cur_e, cur_labels))
    return merged


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--domtbl", required=True,
                    help="hmmscan --domtblout output")
    ap.add_argument("--fasta", required=True,
                    help="FASTA used as hmmscan input (defines protein universe + lengths)")
    ap.add_argument("--rbd-json", required=True,
                    help="JSON mapping family_name -> Pfam accession (e.g. semiconservative_rbd2pfam.json)")
    ap.add_argument("--out", required=True,
                    help="Output prefix")
    ap.add_argument("--coord", choices=["env", "ali"], default="env",
                    help="Coordinate system used for intervals + coverage (default: env)")
    ap.add_argument("--min-len", type=int, default=100,
                    help="Min protein length for passes_filters flag (Step 2; default 100)")
    ap.add_argument("--max-rbd-coverage", type=float, default=0.70,
                    help="Max RBD coverage for passes_filters flag (Step 2; default 0.70)")
    args = ap.parse_args()

    # -- Load whitelist ------------------------------------------------------ #
    rbd_map = json.loads(Path(args.rbd_json).read_text())
    allowed_accs = set(rbd_map.values())
    sys.stderr.write(f"RBD whitelist: {len(allowed_accs)} Pfam accessions "
                     f"({len(rbd_map)} family names)\n")

    # -- Load FASTA universe ------------------------------------------------- #
    fasta_lengths = parse_fasta_lengths(args.fasta)
    sys.stderr.write(f"FASTA: {len(fasta_lengths)} proteins, "
                     f"total {sum(fasta_lengths.values()):,} aa\n")

    # -- Parse + filter domtbl ---------------------------------------------- #
    hits_by_protein = defaultdict(list)
    n_hits = 0
    families_seen = set()
    for hit in parse_domtbl(args.domtbl, allowed_accs):
        hits_by_protein[hit["protein_id"]].append(hit)
        families_seen.add(hit["pfam_acc"])
        n_hits += 1
    sys.stderr.write(
        f"RBD hits after whitelist filter: {n_hits} hits across "
        f"{len(hits_by_protein)} proteins / {len(families_seen)} distinct RBD families\n"
    )

    # -- Sanity check 1: any protein in domtbl missing from FASTA? ---------- #
    missing = set(hits_by_protein) - set(fasta_lengths)
    if missing:
        ex = sorted(missing)[:3]
        sys.stderr.write(
            f"WARNING: {len(missing)} proteins have hits but are missing from the FASTA "
            f"(e.g. {ex}); these will be DROPPED from the output.\n"
        )

    # -- Sanity check 2: which whitelist families produced zero hits? ------- #
    unseen = allowed_accs - families_seen
    if unseen:
        sys.stderr.write(
            f"NOTE: {len(unseen)}/{len(allowed_accs)} whitelist families had zero hits "
            f"in this dataset (e.g. {sorted(unseen)[:5]}). Verify accessions are still "
            f"current in your Pfam release if any are core families you expected to see.\n"
        )

    # -- Output paths -------------------------------------------------------- #
    out = args.out
    hits_path = f"{out}.rbd_hits.tsv"
    intervals_path = f"{out}.rbd_intervals.tsv"
    summary_path = f"{out}.protein_summary.tsv"

    coord_from = f"{args.coord}_from"
    coord_to = f"{args.coord}_to"

    # ---------------------------------------------------------------------- #
    # 1. Long-format hits table
    # ---------------------------------------------------------------------- #
    with open(hits_path, "w") as fh:
        fh.write("protein_id\tpfam_acc\tpfam_name\tenv_from\tenv_to\t"
                 "ali_from\tali_to\ti_evalue\tdom_score\tacc_post\n")
        for pid in sorted(hits_by_protein):
            if pid not in fasta_lengths:
                continue
            for h in sorted(hits_by_protein[pid], key=lambda x: x[coord_from]):
                fh.write(
                    f"{pid}\t{h['pfam_acc']}\t{h['pfam_name']}\t"
                    f"{h['env_from']}\t{h['env_to']}\t"
                    f"{h['ali_from']}\t{h['ali_to']}\t"
                    f"{h['i_evalue']:g}\t{h['dom_score']:g}\t{h['acc_post']:g}\n"
                )
    sys.stderr.write(f"Wrote {hits_path}\n")

    # ---------------------------------------------------------------------- #
    # 2. Merged intervals + 3. Per-protein summary
    # ---------------------------------------------------------------------- #
    n_pass_len = n_pass_cov = n_pass_all = n_with_rbd = 0

    with open(intervals_path, "w") as iv_fh, open(summary_path, "w") as sm_fh:
        iv_fh.write("protein_id\tmerged_start\tmerged_end\tlength\tsource_rbds\n")
        sm_fh.write("protein_id\tprotein_length\tn_rbd_hits\tn_distinct_rbds\t"
                    "rbd_residues\trbd_coverage\tdominant_rbd\t"
                    "passes_len\tpasses_coverage\tpasses_filters\n")

        # Iterate over the FASTA universe so zero-RBD proteins are kept (with 0/0).
        for pid in sorted(fasta_lengths):
            plen = fasta_lengths[pid]
            hits = hits_by_protein.get(pid, [])

            ivs = [(h[coord_from], h[coord_to], h["pfam_name"]) for h in hits]
            merged = merge_intervals(ivs)

            rbd_residues = sum((e - s + 1) for s, e, _ in merged)
            coverage = rbd_residues / plen if plen else 0.0

            # "Dominant" = family contributing the most residues (pre-merge).
            # Tie-break by name for reproducibility.
            family_residues = defaultdict(int)
            for h in hits:
                family_residues[h["pfam_name"]] += h[coord_to] - h[coord_from] + 1
            if family_residues:
                dominant = sorted(family_residues.items(),
                                  key=lambda kv: (-kv[1], kv[0]))[0][0]
            else:
                dominant = ""

            distinct_rbds = len({h["pfam_name"] for h in hits})

            passes_len = plen >= args.min_len
            # Step-2 spec: coverage <= max AND protein has at least one RBD
            # (a protein with zero RBD residues has nothing to test).
            passes_cov = (coverage <= args.max_rbd_coverage) and (rbd_residues > 0)
            passes = passes_len and passes_cov

            n_pass_len += int(passes_len)
            n_pass_cov += int(passes_cov)
            n_pass_all += int(passes)
            n_with_rbd += int(rbd_residues > 0)

            sm_fh.write(
                f"{pid}\t{plen}\t{len(hits)}\t{distinct_rbds}\t"
                f"{rbd_residues}\t{coverage:.4f}\t{dominant}\t"
                f"{int(passes_len)}\t{int(passes_cov)}\t{int(passes)}\n"
            )
            for s, e, labels in merged:
                iv_fh.write(
                    f"{pid}\t{s}\t{e}\t{e - s + 1}\t"
                    f"{','.join(sorted(set(labels)))}\n"
                )

    sys.stderr.write(f"Wrote {intervals_path}\n")
    sys.stderr.write(f"Wrote {summary_path}\n")

    # -- Final counts -------------------------------------------------------- #
    n_total = len(fasta_lengths)
    sys.stderr.write(
        "\n"
        f"Summary:\n"
        f"  proteins in FASTA              : {n_total}\n"
        f"  proteins with >=1 RBD hit      : {n_with_rbd} ({n_with_rbd/n_total:.1%})\n"
        f"  pass length >= {args.min_len:<5}            : {n_pass_len} ({n_pass_len/n_total:.1%})\n"
        f"  pass coverage in (0, {args.max_rbd_coverage}]   : {n_pass_cov} ({n_pass_cov/n_total:.1%})\n"
        f"  pass both Step-2 filters       : {n_pass_all} ({n_pass_all/n_total:.1%})\n"
        f"\nNext: join protein_summary.tsv to your CORAL prediction table"
    )


if __name__ == "__main__":
    main()
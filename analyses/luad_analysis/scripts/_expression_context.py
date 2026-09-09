"""Pure transformations for LUAD expression-contextualized networks."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


COHORTS = ("primary_tumor", "solid_tissue_normal")


def logged_to_tpm(values: pd.DataFrame | pd.Series | np.ndarray) -> Any:
    """Convert Xena log2(TPM + 0.001) values to nonnegative TPM."""
    return np.maximum(np.exp2(values.astype(np.float64)) - 0.001, 0.0)


def build_rbp_expression(
    logged_expression: pd.DataFrame,
    transcripts: pd.DataFrame,
    rbp_config: Mapping[str, Mapping[str, Any]],
    cutoff: float,
) -> dict[str, pd.DataFrame]:
    """Sum raw TPM from protein-coding isoforms into the canonical RBP proxy."""
    result: dict[str, pd.DataFrame] = {}
    for protein_id, spec in rbp_config.items():
        rows = transcripts.loc[
            (transcripts["gene_id"] == spec["gene_id"])
            & (transcripts["gene_name"] == spec["gene_symbol"])
        ]
        if len(rows) != int(spec["transcript_count"]):
            raise ValueError(f"{protein_id}: unexpected RBP transcript count")
        transcript_ids = rows.loc[rows["transcript_biotype"] == "protein_coding", "transcript_id"].tolist()
        if not transcript_ids:
            raise ValueError(f"{protein_id}: no protein-coding isoforms")
        missing = set(transcript_ids) - set(logged_expression.index)
        if missing:
            raise ValueError(f"{protein_id}: expression missing RBP transcripts")
        gene_tpm = logged_to_tpm(logged_expression.loc[transcript_ids]).sum(axis=0)
        result[protein_id] = pd.DataFrame(
            {
                "rbp_gene_tpm": gene_tpm,
                "rbp_available": gene_tpm >= cutoff,
            },
            index=logged_expression.columns,
        )
    return result


def build_sample_edge_states(
    top_edges: pd.DataFrame,
    logged_expression: pd.DataFrame,
    samples: pd.DataFrame,
    rbp_expression: Mapping[str, pd.DataFrame],
    cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create normalized sample-edge states and per-sample RBP summaries."""
    threshold = np.log2(cutoff + 0.001)
    sample_ids = samples["sample_id"].tolist()
    state_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for protein_id, edges in top_edges.groupby("Protein_id", sort=False):
        transcript_ids = edges["transcript_id"].tolist()
        logged = logged_expression.loc[transcript_ids, sample_ids]
        target_tpm = logged_to_tpm(logged).T
        target_available = logged.ge(threshold).T
        rbp = rbp_expression[protein_id].loc[sample_ids]
        sample_count, edge_count = target_tpm.shape
        active = target_available.to_numpy() & rbp["rbp_available"].to_numpy()[:, None]

        state_parts.append(
            pd.DataFrame(
                {
                    "sample_id": np.repeat(sample_ids, edge_count),
                    "sample_type": np.repeat(samples["sample_type"].to_numpy(), edge_count),
                    "patient_id": np.repeat(samples["patient_id"].to_numpy(), edge_count),
                    "pair_status": np.repeat(samples["pair_status"].to_numpy(), edge_count),
                    "Protein_id": protein_id,
                    "transcript_id": np.tile(transcript_ids, sample_count),
                    "target_tpm": target_tpm.to_numpy().ravel(),
                    "target_available": target_available.to_numpy().ravel(),
                    "rbp_gene_tpm": np.repeat(rbp["rbp_gene_tpm"].to_numpy(), edge_count),
                    "rbp_available": np.repeat(rbp["rbp_available"].to_numpy(), edge_count),
                    "active": active.ravel(),
                }
            )
        )
        summary = samples.copy()
        summary["Protein_id"] = protein_id
        summary["rbp_gene_tpm"] = rbp["rbp_gene_tpm"].to_numpy()
        summary["rbp_available"] = rbp["rbp_available"].to_numpy()
        summary["target_available_edge_count"] = target_available.sum(axis=1).to_numpy()
        summary["active_edge_count"] = active.sum(axis=1)
        summary["candidate_edge_count"] = edge_count
        summary["active_edge_fraction"] = summary["active_edge_count"] / edge_count
        summary_parts.append(summary)

    states = pd.concat(state_parts, ignore_index=True)
    summaries = pd.concat(summary_parts, ignore_index=True)
    if states[["sample_id", "Protein_id", "transcript_id"]].duplicated().any():
        raise ValueError("sample-edge states contain duplicate composite keys")
    return states, summaries


def bh_adjust(frame: pd.DataFrame, family: str, alpha: float) -> pd.DataFrame:
    """Attach Benjamini-Hochberg adjusted p-values to one declared family."""
    output = frame.copy()
    if output.empty:
        return output.assign(padj=pd.Series(dtype=float), reject_bh=pd.Series(dtype=bool), bh_family=family)
    rejected, adjusted, _, _ = multipletests(output["pvalue"], alpha=alpha, method="fdr_bh")
    output["padj"] = adjusted
    output["reject_bh"] = rejected
    output["bh_family"] = family
    return output


def conditional_prevalence(
    active_count: pd.Series | np.ndarray, rbp_available_count: pd.Series | np.ndarray
) -> np.ndarray:
    """Return active prevalence among RBP-available samples, or NaN with no denominator."""
    numerator = np.asarray(active_count, dtype=np.float64)
    denominator = np.asarray(rbp_available_count, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def wilcoxon_test(
    differences: np.ndarray,
    *,
    zero_method: str,
    alternative: str,
    resamples: int,
    seed: int,
) -> tuple[float, float, str]:
    """Run a deterministic permutation Wilcoxon test with an all-zero guard."""
    values = np.asarray(differences, dtype=np.float64)
    if np.all(values == 0):
        return 0.0, 1.0, "all_zero_differences"
    method = stats.PermutationMethod(
        n_resamples=resamples,
        batch=min(10_000, resamples),
        rng=np.random.default_rng(seed),
    )
    result = stats.wilcoxon(
        values,
        zero_method=zero_method,
        alternative=alternative,
        method=method,
    )
    return float(result.statistic), float(result.pvalue), "ok"


def exact_mcnemar(normal: np.ndarray, tumor: np.ndarray) -> dict[str, Any]:
    """Return paired active-state counts and an exact two-sided McNemar p-value."""
    normal_state = np.asarray(normal, dtype=bool)
    tumor_state = np.asarray(tumor, dtype=bool)
    both_inactive = int(np.count_nonzero(~normal_state & ~tumor_state))
    tumor_only = int(np.count_nonzero(~normal_state & tumor_state))
    normal_only = int(np.count_nonzero(normal_state & ~tumor_state))
    both_active = int(np.count_nonzero(normal_state & tumor_state))
    if tumor_only + normal_only == 0:
        statistic, pvalue = 0.0, 1.0
    else:
        result = mcnemar(
            [[both_inactive, tumor_only], [normal_only, both_active]], exact=True
        )
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {
        "matched_both_inactive": both_inactive,
        "matched_tumor_only": tumor_only,
        "matched_normal_only": normal_only,
        "matched_both_active": both_active,
        "mcnemar_statistic": statistic,
        "pvalue": pvalue,
    }



# --- tables ---

def matched_sample_maps(samples: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return stable patient, normal-sample, and tumor-sample lists."""
    matched = samples.loc[samples["pair_status"] == "matched"]
    normal = matched.loc[matched["sample_type"] == COHORTS[1]].set_index("patient_id")
    tumor = matched.loc[matched["sample_type"] == COHORTS[0]].set_index("patient_id")
    if normal.index.duplicated().any() or tumor.index.duplicated().any() or set(normal.index) != set(tumor.index):
        raise ValueError("matched patients require exactly one tumor and one normal")
    patients = sorted(normal.index)
    return patients, normal.loc[patients, "sample_id"].tolist(), tumor.loc[patients, "sample_id"].tolist()


def build_edge_table(
    top: pd.DataFrame,
    states: pd.DataFrame,
    proteins: pd.DataFrame,
    samples: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Attach cohort prevalence and matched exact McNemar results to candidate edges."""
    keys = ["Protein_id", "transcript_id"]
    rows = top.merge(
        proteins[["Protein_id", "gene_symbol"]], on="Protein_id", validate="many_to_one"
    ).rename(columns={"gene_symbol": "rbp_gene"})
    aggregate = (
        states.groupby([*keys, "sample_type"], sort=False)
        .agg(
            cohort_samples=("active", "size"),
            target_available_count=("target_available", "sum"),
            rbp_available_count=("rbp_available", "sum"),
            active_count=("active", "sum"),
        )
        .reset_index()
    )
    for cohort, prefix in ((COHORTS[0], "tumor"), (COHORTS[1], "normal")):
        scoped = aggregate.loc[aggregate["sample_type"] == cohort].drop(columns="sample_type")
        scoped = scoped.rename(
            columns={name: f"{prefix}_{name}" for name in scoped.columns if name not in keys}
        )
        rows = rows.merge(scoped, on=keys, validate="one_to_one")
        rows[f"{prefix}_target_available_fraction"] = (
            rows[f"{prefix}_target_available_count"] / rows[f"{prefix}_cohort_samples"]
        )
        rows[f"{prefix}_active_fraction"] = (
            rows[f"{prefix}_active_count"] / rows[f"{prefix}_cohort_samples"]
        )
        rows[f"{prefix}_active_given_rbp_available"] = conditional_prevalence(
            rows[f"{prefix}_active_count"], rows[f"{prefix}_rbp_available_count"]
        )

    patients, normal_ids, tumor_ids = matched_sample_maps(samples)
    test_rows: list[dict[str, Any]] = []
    for protein_id, group in rows.groupby("Protein_id", sort=False):
        matrix = states.loc[states["Protein_id"] == protein_id].pivot(
            index="transcript_id", columns="sample_id", values="active"
        )
        for transcript_id in group["transcript_id"]:
            result = exact_mcnemar(
                matrix.loc[transcript_id, normal_ids].to_numpy(),
                matrix.loc[transcript_id, tumor_ids].to_numpy(),
            )
            test_rows.append({"Protein_id": protein_id, "transcript_id": transcript_id, **result})
    tests = bh_adjust(pd.DataFrame(test_rows), "all_top1pct_transcript_edges", alpha)
    tests = tests.rename(
        columns={
            "pvalue": "mcnemar_pvalue",
            "padj": "mcnemar_padj",
            "reject_bh": "mcnemar_reject_bh",
        }
    ).drop(columns="bh_family")
    output = rows.merge(tests, on=keys, validate="one_to_one")
    if len(output) != len(top):
        raise ValueError("edge prevalence output has unexpected dimensions")
    return output.sort_values(["Protein_id", "primary_transcript_rank"], kind="stable")


def build_cohort_summary(sample_context: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    """Summarize RBP availability and candidate-edge burden by cohort."""
    symbols = dict(zip(proteins["Protein_id"], proteins["gene_symbol"], strict=True))
    rows: list[dict[str, Any]] = []
    for (protein_id, cohort), group in sample_context.groupby(["Protein_id", "sample_type"]):
        row: dict[str, Any] = {
            "Protein_id": protein_id,
            "rbp_gene": symbols[protein_id],
            "sample_type": cohort,
            "sample_count": len(group),
            "rbp_available_count": int(group["rbp_available"].sum()),
            "rbp_available_fraction": float(group["rbp_available"].mean()),
        }
        for column in ("rbp_gene_tpm", "target_available_edge_count", "active_edge_count"):
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_q1"] = float(group[column].quantile(0.25))
            row[f"{column}_q3"] = float(group[column].quantile(0.75))
        available = group.loc[group["rbp_available"], "target_available_edge_count"]
        for quantile, suffix in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
            row[f"target_available_edge_count_given_rbp_available_{suffix}"] = (
                float(available.quantile(quantile)) if len(available) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Protein_id", "sample_type"], kind="stable")


def build_paired_outputs(
    sample_context: pd.DataFrame,
    proteins: pd.DataFrame,
    samples: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build auditable paired changes and six oncofetal-expression QC comparisons."""
    patients, normal_ids, tumor_ids = matched_sample_maps(samples)
    expression_pc = float(config["expression_pseudocount"])
    count_pc = float(config["active_count_pseudocount"])
    method = config["wilcoxon"]
    symbols = dict(zip(proteins["Protein_id"], proteins["gene_symbol"], strict=True))
    changes: list[pd.DataFrame] = []
    tests: list[dict[str, Any]] = []
    for protein_id in proteins["Protein_id"]:
        group = sample_context.loc[
            sample_context["Protein_id"] == protein_id
        ].set_index("sample_id")
        normal_tpm = group.loc[normal_ids, "rbp_gene_tpm"].to_numpy(float)
        tumor_tpm = group.loc[tumor_ids, "rbp_gene_tpm"].to_numpy(float)
        normal_count = group.loc[normal_ids, "active_edge_count"].to_numpy(int)
        tumor_count = group.loc[tumor_ids, "active_edge_count"].to_numpy(int)
        expression_log2fc = np.round(
            np.log2(tumor_tpm + expression_pc) - np.log2(normal_tpm + expression_pc),
            int(method["difference_round_decimals"]),
        )
        count_difference = tumor_count - normal_count
        count_log2fc = np.log2((tumor_count + count_pc) / (normal_count + count_pc))
        changes.append(
            pd.DataFrame(
                {
                    "patient_id": patients,
                    "candidate_edge_count": int(group["candidate_edge_count"].iloc[0]),
                    "Protein_id": protein_id,
                    "rbp_gene": symbols[protein_id],
                    "normal_sample_id": normal_ids,
                    "tumor_sample_id": tumor_ids,
                    "normal_rbp_gene_tpm": normal_tpm,
                    "tumor_rbp_gene_tpm": tumor_tpm,
                    "normal_rbp_log2_tpm": np.log2(normal_tpm + expression_pc),
                    "tumor_rbp_log2_tpm": np.log2(tumor_tpm + expression_pc),
                    "normal_rbp_available": group.loc[normal_ids, "rbp_available"].to_numpy(bool),
                    "tumor_rbp_available": group.loc[tumor_ids, "rbp_available"].to_numpy(bool),
                    "rbp_tpm_difference": tumor_tpm - normal_tpm,
                    "rbp_log2_fold_change": expression_log2fc,
                    "normal_active_edge_count": normal_count,
                    "tumor_active_edge_count": tumor_count,
                    "normal_active_edge_fraction": normal_count / int(group["candidate_edge_count"].iloc[0]),
                    "tumor_active_edge_fraction": tumor_count / int(group["candidate_edge_count"].iloc[0]),
                    "active_edge_count_difference": count_difference,
                    "active_edge_count_log2_fold_change": count_log2fc,
                }
            )
        )
        endpoint_specs = (
            (
                "rbp_expression",
                expression_log2fc,
                normal_tpm,
                tumor_tpm,
                tumor_tpm - normal_tpm,
                expression_log2fc,
                "log2(TPM+0.001)",
            ),
            (
                "active_edge_count",
                count_difference,
                normal_count,
                tumor_count,
                count_difference,
                count_log2fc,
                "raw count difference",
            ),
        )
        for endpoint, test_values, normal, tumor, difference, log2fc, scale in endpoint_specs:
            statistic, pvalue, status = wilcoxon_test(
                test_values,
                zero_method=method["zero_method"],
                alternative=method["alternative"],
                resamples=int(method["permutation_resamples"]),
                seed=int(method["seed"]),
            )
            tests.append(
                {
                    "endpoint_family": endpoint,
                    "analysis_role": "oncofetal_expression_qc",
                    "Protein_id": protein_id,
                    "rbp_gene": symbols[protein_id],
                    "matched_pairs": len(patients),
                    "normal_median": float(np.median(normal)),
                    "tumor_median": float(np.median(tumor)),
                    "median_paired_difference": float(np.median(difference)),
                    "median_paired_log2_fold_change": float(np.median(log2fc)),
                    "zero_differences": int(np.count_nonzero(test_values == 0)),
                    "test_scale": scale,
                    "wilcoxon_statistic": statistic,
                    "pvalue": pvalue,
                    "test_status": status,
                }
            )
    test_frame = pd.DataFrame(tests)
    corrected = [
        bh_adjust(group, family, float(config["bh_alpha"]))
        for family, group in test_frame.groupby("endpoint_family", sort=False)
    ]
    return pd.concat(changes, ignore_index=True), pd.concat(corrected, ignore_index=True)




def build_anchor_context(
    anchors: pd.DataFrame,
    logged_expression: pd.DataFrame,
    rbp_expression: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    cutoff: float,
) -> pd.DataFrame:
    """Report expression gates for every literature anchor without changing its rank."""
    threshold = np.log2(cutoff + 0.001)
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        target_logged = logged_expression.loc[anchor["representative_transcript_id"]]
        target_tpm = logged_to_tpm(target_logged)
        target_available = target_logged >= threshold
        rbp = rbp_expression[anchor["Protein_id"]]
        joint = target_available & rbp["rbp_available"]
        top_gate = bool(anchor["top1pct_candidate"])
        active = joint & top_gate
        row = {**anchor, "top1pct_candidate": top_gate}
        for cohort, prefix in ((COHORTS[0], "tumor"), (COHORTS[1], "normal")):
            sample_ids = samples.loc[samples["sample_type"] == cohort, "sample_id"]
            size = len(sample_ids)
            values = target_tpm.loc[sample_ids]
            row[f"{prefix}_target_tpm_median"] = float(values.median())
            for name, state in (
                ("target_available", target_available),
                ("rbp_available", rbp["rbp_available"]),
                ("joint_available", joint),
                ("active", active),
            ):
                count = int(state.loc[sample_ids].sum())
                row[f"{prefix}_{name}_count"] = count
                row[f"{prefix}_{name}_fraction"] = count / size
            denominator = int(rbp.loc[sample_ids, "rbp_available"].sum())
            for name, state in (("joint_available", joint), ("active", active)):
                row[f"{prefix}_{name}_given_rbp_available"] = (
                    int(state.loc[sample_ids].sum()) / denominator if denominator else np.nan
                )
        rows.append(row)
    output = pd.DataFrame(rows)
    return output

# --- recurrence ---

def availability_matrix(sample_context: pd.DataFrame) -> pd.DataFrame:
    """Align the recorded RBP gates by sample, rejecting incomplete subsets."""
    gates = sample_context.pivot(index="sample_id", columns="Protein_id", values="rbp_available")
    if gates.isna().any().any() or not gates.isin([True, False]).all().all():
        raise ValueError("RBP availability matrix is incomplete or non-boolean")
    return gates.astype(bool)


def build_rbp_subsets(sample_context: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    """Record every individual, pairwise, and triple RBP-positive cohort denominator."""
    gates = availability_matrix(sample_context)
    symbols = proteins.set_index("Protein_id")["gene_symbol"].to_dict()
    ordered = sorted(symbols, key=symbols.get)
    metadata = sample_context[["sample_id", "sample_type"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for degree in range(1, len(ordered) + 1):
        for linked in combinations(ordered, degree):
            eligible = gates[list(linked)].all(axis=1)
            for cohort in COHORTS:
                ids = sorted(metadata.loc[metadata["sample_type"] == cohort, "sample_id"])
                positive_ids = eligible.loc[ids].loc[lambda values: values].index.tolist()
                rows.append({
                    "rbp_genes": ";".join(symbols[protein] for protein in linked),
                    "Protein_ids": ";".join(linked),
                    "rbp_degree": degree,
                    "sample_type": cohort,
                    "cohort_samples": len(ids),
                    "rbp_available_count": len(positive_ids),
                    "rbp_available_fraction": len(positive_ids) / len(ids) if ids else np.nan,
                    "rbp_available_sample_ids": ";".join(positive_ids),
                })
    return pd.DataFrame(rows)


def build_recurrent_targets(
    top: pd.DataFrame,
    states: pd.DataFrame,
    proteins: pd.DataFrame,
    samples: pd.DataFrame,
    sample_context: pd.DataFrame,
) -> pd.DataFrame:
    """Keep all recurrent genes and report both cohort and RBP-conditioned availability.

    Each RBP's own positive samples define its conditional fraction. Cross-RBP
    co-availability uses the intersection of the linked RBP-positive samples.
    These are descriptive any-isoform gene summaries, not new target units.
    """
    symbols = proteins.set_index("Protein_id")["gene_symbol"].to_dict()
    gates = availability_matrix(sample_context).loc[samples["sample_id"]]
    annotated = states.merge(
        top[["Protein_id", "transcript_id", "gene_id", "gene_name"]],
        on=["Protein_id", "transcript_id"], validate="many_to_one",
    )
    gene_states = (
        annotated.groupby(["Protein_id", "gene_id", "sample_id"], sort=False)["active"]
        .any().reset_index()
    )
    degree = top.groupby("gene_id")["Protein_id"].nunique().rename("rbp_degree")
    relationships = top.merge(degree.loc[degree >= 2], on="gene_id", validate="many_to_one")
    cohorts = {
        prefix: samples.loc[samples["sample_type"] == cohort, "sample_id"]
        for cohort, prefix in zip(COHORTS, ("tumor", "normal"), strict=True)
    }
    rows: list[dict[str, Any]] = []
    for (gene_id, gene_name), gene_edges in relationships.groupby(["gene_id", "gene_name"]):
        linked = sorted(gene_edges["Protein_id"].unique(), key=symbols.get)
        per_sample = gene_states.loc[gene_states["gene_id"] == gene_id].pivot(
            index="sample_id", columns="Protein_id", values="active"
        ).loc[samples["sample_id"], linked]
        if per_sample.isna().any().any():
            raise ValueError(f"incomplete gene states: {gene_id}")
        all_linked = per_sample.all(axis=1)
        eligible = gates[linked].all(axis=1)
        common: dict[str, Any] = {
            "gene_id": gene_id, "gene_name": gene_name, "rbp_degree": len(linked),
            "linked_rbp_genes": ";".join(symbols[protein] for protein in linked),
            "worst_best_transcript_rank": int(
                gene_edges.groupby("Protein_id")["primary_transcript_rank"].min().max()
            ),
        }
        for prefix, ids in cohorts.items():
            count = int(all_linked.loc[ids].sum())
            denominator = int(eligible.loc[ids].sum())
            if count > denominator:
                raise ValueError(f"gene activity exceeds common RBP denominator: {gene_id}")
            common[f"{prefix}_cohort_samples"] = len(ids)
            common[f"all_linked_active_{prefix}_count"] = count
            common[f"all_linked_active_{prefix}_fraction"] = count / len(ids) if len(ids) else np.nan
            common[f"all_linked_rbp_available_{prefix}_count"] = denominator
            common[f"all_linked_active_{prefix}_given_all_rbps_available"] = (
                count / denominator if denominator else np.nan
            )
        for protein_id, rbp_edges in gene_edges.groupby("Protein_id", sort=False):
            ordered = rbp_edges.sort_values("primary_transcript_rank", kind="stable")
            row = {
                **common, "Protein_id": protein_id, "rbp_gene": symbols[protein_id],
                "supporting_transcript_count": len(ordered),
                "supporting_transcript_ids": ";".join(ordered["transcript_id"]),
                "supporting_transcript_biotypes": ";".join(ordered["transcript_biotype"]),
                "supporting_transcript_ranks": ";".join(ordered["primary_transcript_rank"].astype(str)),
                "best_transcript_id": ordered.iloc[0]["transcript_id"],
                "best_transcript_rank": int(ordered.iloc[0]["primary_transcript_rank"]),
                "best_logit_margin": float(ordered.iloc[0]["logit_margin"]),
            }
            for prefix, ids in cohorts.items():
                count = int(per_sample.loc[ids, protein_id].sum())
                denominator = int(gates.loc[ids, protein_id].sum())
                shared_count = int((per_sample[protein_id] & eligible).loc[ids].sum())
                shared_denominator = common[f"all_linked_rbp_available_{prefix}_count"]
                if count > denominator:
                    raise ValueError(f"gene activity exceeds RBP denominator: {gene_id}, {protein_id}")
                row[f"{prefix}_gene_active_count"] = count
                row[f"{prefix}_gene_active_fraction"] = count / len(ids) if len(ids) else np.nan
                row[f"{prefix}_rbp_available_count"] = denominator
                row[f"{prefix}_gene_active_given_rbp_available"] = (
                    count / denominator if denominator else np.nan
                )
                row[f"{prefix}_gene_active_in_all_rbps_available_count"] = shared_count
                row[f"{prefix}_gene_active_given_all_rbps_available"] = (
                    shared_count / shared_denominator if shared_denominator else np.nan
                )
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["gene_id", "gene_name", "Protein_id", "rbp_degree", "rbp_gene"])
    return pd.DataFrame(rows).sort_values(["gene_id", "Protein_id"], kind="stable")

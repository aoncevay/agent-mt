#!/usr/bin/env python3
"""
Plot agent contribution summaries produced by analyze_agent_contributions.py.

Usage:
  python report/plot_agent_contributions.py \
    --analysis_dir report/contribution_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing plotting dependencies. Install project requirements first:\n"
        "  pip install -r requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot agent contribution summaries")
    parser.add_argument(
        "--analysis_dir",
        default="report/contribution_analysis",
        help="Directory containing summary CSV files from analyze_agent_contributions.py",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where plots are saved (default: same as analysis_dir)",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=None,
        help="Optional workflow filter (exact names from transitions_raw.csv)",
    )
    return parser.parse_args()


def plot_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["curr_agent_type"], df[value_col], color="#3A7CA5")
    ax.set_title(title)
    ax.set_xlabel("Agent Type")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_workflow_step_summary(transitions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-workflow, per-step summary from raw transitions.
    """
    if transitions_df.empty:
        return pd.DataFrame()

    group_cols = ["workflow", "curr_step_name", "curr_agent_type"]

    observed = (
        transitions_df.groupby(group_cols, dropna=False)
        .agg(
            observed_transitions=("curr_step_name", "size"),
            median_transition_order=("transition_order", "median"),
        )
        .reset_index()
    )

    comparable = transitions_df[transitions_df["is_comparable"] == True].copy()
    if comparable.empty:
        summary = observed.copy()
        summary["analyzable_transitions"] = 0
        summary["changed_steps"] = 0
        summary["pct_changed"] = np.nan
        summary["avg_edit_distance_when_changed"] = np.nan
        return summary

    comparable["changed_int"] = comparable["changed"].astype(int)
    comparable["edit_distance_when_changed"] = np.where(
        comparable["changed"] == True, comparable["edit_distance"], np.nan
    )

    agg = (
        comparable.groupby(group_cols, dropna=False)
        .agg(
            analyzable_transitions=("is_comparable", "size"),
            changed_steps=("changed_int", "sum"),
            avg_edit_distance_when_changed=("edit_distance_when_changed", "mean"),
        )
        .reset_index()
    )

    summary = observed.merge(agg, on=group_cols, how="left")
    summary["analyzable_transitions"] = summary["analyzable_transitions"].fillna(0).astype(int)
    summary["changed_steps"] = summary["changed_steps"].fillna(0).astype(int)
    summary["pct_changed"] = np.where(
        summary["analyzable_transitions"] > 0,
        100.0 * summary["changed_steps"] / summary["analyzable_transitions"],
        np.nan,
    )
    return summary


def _sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def plot_per_workflow_step_contributions(summary_df: pd.DataFrame, output_dir: Path) -> int:
    """
    Create one figure per workflow with bars per step.
    Top panel: % changed
    Bottom panel: avg char edit distance when changed
    """
    if summary_df.empty:
        return 0

    num_written = 0
    workflows = sorted(summary_df["workflow"].dropna().unique())
    for workflow in workflows:
        wf_df = summary_df[summary_df["workflow"] == workflow].copy()
        if wf_df.empty:
            continue

        # Keep workflow step order (from transition_order), then step name.
        wf_df = wf_df.sort_values(["median_transition_order", "curr_step_name"]).reset_index(drop=True)
        x = np.arange(len(wf_df))
        labels = [f"{s}\n({a})" for s, a in zip(wf_df["curr_step_name"], wf_df["curr_agent_type"])]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(10, 1.6 * len(wf_df)), 8), sharex=True
        )

        # Panel 1: pct changed
        pct_vals = wf_df["pct_changed"].fillna(0.0).to_numpy()
        analyzable = wf_df["analyzable_transitions"].to_numpy()
        bars1 = ax1.bar(x, pct_vals, color="#2A9D8F")
        for i, (bar, n) in enumerate(zip(bars1, analyzable)):
            if n == 0:
                bar.set_color("#BDBDBD")
                bar.set_hatch("//")
            changed = int(wf_df.iloc[i]["changed_steps"])
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.0,
                f"{changed}/{int(n)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax1.set_ylabel("% changed")
        ax1.set_ylim(0, max(100.0, float(np.nanmax(pct_vals) + 12.0)))
        ax1.set_title(f"{workflow}: Step-Level Change Rate")
        ax1.grid(axis="y", linestyle="--", alpha=0.35)

        # Panel 2: avg edit distance when changed
        edit_vals = wf_df["avg_edit_distance_when_changed"].fillna(0.0).to_numpy()
        has_edit = wf_df["avg_edit_distance_when_changed"].notna().to_numpy()
        bars2 = ax2.bar(x, edit_vals, color="#3A7CA5")
        for bar, present in zip(bars2, has_edit):
            if not present:
                bar.set_color("#BDBDBD")
                bar.set_hatch("//")
        ax2.set_ylabel("Avg char edit dist\n(when changed)")
        ax2.set_title(f"{workflow}: Step-Level Edit Magnitude")
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, ha="right")
        ax2.set_xlabel("Step (agent type)")

        fig.tight_layout()
        out_path = output_dir / f"workflow_{_sanitize_filename(workflow)}_step_contributions.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        num_written += 1

    return num_written


def main() -> int:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_file = analysis_dir / "summary_global_by_agent_type.csv"
    transitions_file = analysis_dir / "transitions_raw.csv"
    if not source_file.exists():
        raise SystemExit(
            f"Missing input file: {source_file}\n"
            "Run report/analyze_agent_contributions.py first."
        )
    if not transitions_file.exists():
        raise SystemExit(
            f"Missing input file: {transitions_file}\n"
            "Run report/analyze_agent_contributions.py first."
        )

    # Existing global plots by agent type.
    df = pd.read_csv(source_file)
    if df.empty:
        raise SystemExit(f"No rows in {source_file}.")

    df_unchanged = df.dropna(subset=["pct_unchanged"]).sort_values("pct_unchanged", ascending=False)
    plot_bar(
        df=df_unchanged,
        value_col="pct_unchanged",
        title="Percentage of Unchanged Transitions by Agent Type",
        ylabel="% unchanged (comparable transitions)",
        out_path=output_dir / "agent_type_pct_unchanged.png",
    )

    df_changed = df.dropna(subset=["avg_edit_distance_when_changed"]).sort_values(
        "avg_edit_distance_when_changed", ascending=False
    )
    plot_bar(
        df=df_changed,
        value_col="avg_edit_distance_when_changed",
        title="Average Character Edit Distance (When Changed) by Agent Type",
        ylabel="Average char-level edit distance",
        out_path=output_dir / "agent_type_avg_edit_distance_when_changed.png",
    )

    # New: one figure per workflow with bars per step.
    transitions_df = pd.read_csv(transitions_file)
    if args.workflows:
        workflows_set = set(args.workflows)
        transitions_df = transitions_df[transitions_df["workflow"].isin(workflows_set)].copy()

    summary_by_workflow_step = compute_workflow_step_summary(transitions_df)
    summary_csv = output_dir / "summary_by_workflow_step_for_plots.csv"
    summary_by_workflow_step.to_csv(summary_csv, index=False)
    num_workflow_figs = plot_per_workflow_step_contributions(summary_by_workflow_step, output_dir)

    print(f"Saved plots to: {output_dir}")
    print(f"Saved per-workflow step summary: {summary_csv}")
    print(f"Workflow-specific figures written: {num_workflow_figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

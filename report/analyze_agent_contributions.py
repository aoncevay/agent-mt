#!/usr/bin/env python3
"""
Analyze per-step agent contribution from saved workflow outputs.

The script reads report files and per-agent output files:
  outputs/{dataset}/{lang_pair}/{workflow_dir}/{model}/sample_*_agent_*.txt

For each sample, it reconstructs workflow step states and computes transition-level
change metrics:
1) Percentage of steps that leave translation unchanged
2) Character-level edit distance when changes occur (grouped by agent type)

Usage:
  python report/analyze_agent_contributions.py
  python report/analyze_agent_contributions.py --outputs_dirs outputs outputs_qwen3
  python report/analyze_agent_contributions.py --models gpt-4-1 qwen3-32b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies for analysis. Install project requirements first:\n"
        "  pip install -r requirements.txt"
    ) from exc

try:
    from rapidfuzz.distance import Levenshtein as RapidFuzzLevenshtein

    def char_edit_distance(a: str, b: str) -> int:
        return int(RapidFuzzLevenshtein.distance(a, b))

    EDIT_DISTANCE_IMPL = "rapidfuzz"
except Exception:
    # Fallback to a pure Python implementation if rapidfuzz is unavailable.
    def char_edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        # Standard DP with O(min(m, n)) memory
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (ca != cb)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    EDIT_DISTANCE_IMPL = "python_fallback"

try:
    from .agent_step_specs import resolve_workflow_step_specs
except ImportError:
    from agent_step_specs import resolve_workflow_step_specs


REPORT_SAMPLES_RE = re.compile(r"^report_(\d+)_samples\.json$")
AGENT_FILE_RE = re.compile(r"^(sample_.+)_agent_(\d+)\.txt$")
WMT25_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-step agent contributions")
    parser.add_argument(
        "--outputs_dirs",
        nargs="+",
        default=["outputs", "outputs_qwen3"],
        help="Output roots to scan (default: outputs outputs_qwen3)",
    )
    parser.add_argument(
        "--output_dir",
        default="report/contribution_analysis",
        help="Directory where analysis artifacts will be written",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset filter (e.g., dolfin wmt25)",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=None,
        help="Optional workflow filter (matches workflow name or workflow dir/acronym)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model filter",
    )
    parser.add_argument(
        "--require_complete_reports",
        action="store_true",
        help="Only include reports where total_samples == successful_samples and > 0",
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw",
        help="Base data directory used to resolve source sample character length",
    )
    return parser.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _sample_file_prefix(sample_id: Any, sample_idx: Any) -> str:
    sample_idx_int = _safe_int(sample_idx, 0)
    sample_id_str = str(sample_id) if sample_id is not None else str(sample_idx_int)
    if sample_id_str and sample_id_str != str(sample_idx_int):
        safe_id = sample_id_str.replace("/", "_").replace("\\", "_")[:50]
        return f"sample_{safe_id}"
    return f"sample_{sample_idx_int:05d}"


def _extract_report_priority(report_path: Path) -> Tuple[int, int]:
    # Higher tuple wins: (is_report_json, sample_limit)
    if report_path.name == "report.json":
        return (1, 10**9)
    match = REPORT_SAMPLES_RE.match(report_path.name)
    if match:
        return (0, int(match.group(1)))
    return (0, -1)


def discover_report_files(outputs_dir: Path) -> List[Path]:
    """
    Discover one preferred report file per experiment directory.
    """
    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for report_path in outputs_dir.rglob("report*.json"):
        try:
            rel = report_path.relative_to(outputs_dir)
        except ValueError:
            continue
        # Expected: {dataset}/{lang_pair}/{workflow_dir}/{model}/{report*.json}
        if len(rel.parts) != 5:
            continue
        grouped[report_path.parent].append(report_path)

    chosen: List[Path] = []
    for _exp_dir, candidates in grouped.items():
        best = max(candidates, key=_extract_report_priority)
        chosen.append(best)
    return sorted(chosen)


def parse_experiment_metadata(outputs_dir: Path, report_path: Path) -> Optional[Dict[str, str]]:
    try:
        rel = report_path.relative_to(outputs_dir)
    except ValueError:
        return None
    if len(rel.parts) != 5:
        return None
    dataset, lang_pair, workflow_dir, model, _report_name = rel.parts
    workflow_acronym = workflow_dir.replace(".term", "")
    return {
        "dataset": dataset,
        "lang_pair": lang_pair,
        "workflow_dir": workflow_dir,
        "workflow_acronym": workflow_acronym,
        "model": model,
    }


def _matches_filters(
    metadata: Dict[str, str],
    workflow_name: str,
    datasets: Optional[set],
    workflows: Optional[set],
    models: Optional[set],
) -> bool:
    if datasets and metadata["dataset"] not in datasets:
        return False
    if models and metadata["model"] not in models:
        return False
    if workflows:
        workflow_candidates = {
            workflow_name,
            metadata["workflow_dir"],
            metadata["workflow_acronym"],
        }
        if workflow_candidates.isdisjoint(workflows):
            return False
    return True


def load_agent_outputs_for_sample(model_dir: Path, sample_prefix: str) -> List[str]:
    """
    Load all agent outputs for one sample, sorted by agent index.
    """
    files = list(model_dir.glob(f"{sample_prefix}_agent_*.txt"))
    indexed: List[Tuple[int, Path]] = []
    for file_path in files:
        match = AGENT_FILE_RE.match(file_path.name)
        if not match:
            continue
        try:
            agent_idx = int(match.group(2))
        except ValueError:
            continue
        indexed.append((agent_idx, file_path))
    indexed.sort(key=lambda x: x[0])

    outputs: List[str] = []
    for _idx, file_path in indexed:
        try:
            outputs.append(file_path.read_text(encoding="utf-8").strip())
        except Exception:
            # If one output file cannot be read, skip it.
            continue
    return outputs


def derive_translation_state(
    mode: str,
    output_index: int,
    outputs: List[str],
    previous_translation_state: Optional[str],
) -> Optional[str]:
    if mode == "none":
        return None
    if mode == "direct":
        if 0 <= output_index < len(outputs):
            return outputs[output_index]
        return None
    if mode == "carry":
        return previous_translation_state
    if mode == "append":
        if output_index < 0:
            return None
        if output_index >= len(outputs):
            return None
        return "\n\n".join(outputs[: output_index + 1]).strip()
    raise ValueError(f"Unknown translation mode: {mode}")


def summarize_transitions(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Summarize transition metrics for any grouping level.
    """
    if df.empty:
        return pd.DataFrame()

    if not group_cols:
        observed = len(df)
        analyzable_df = df[df["is_comparable"] == True].copy()
        analyzable = len(analyzable_df)
        if analyzable == 0:
            return pd.DataFrame(
                [
                    {
                        "observed_transitions": observed,
                        "analyzable_transitions": 0,
                        "non_comparable_transitions": observed,
                        "changed_steps": 0,
                        "unchanged_steps": 0,
                        "pct_unchanged": np.nan,
                        "avg_edit_distance_when_changed": np.nan,
                        "median_edit_distance_when_changed": np.nan,
                        "avg_normalized_edit_when_changed": np.nan,
                    }
                ]
            )

        changed_steps = int(analyzable_df["changed"].sum())
        unchanged_steps = int((~analyzable_df["changed"]).sum())
        changed_only = analyzable_df[analyzable_df["changed"] == True]

        return pd.DataFrame(
            [
                {
                    "observed_transitions": observed,
                    "analyzable_transitions": analyzable,
                    "non_comparable_transitions": observed - analyzable,
                    "changed_steps": changed_steps,
                    "unchanged_steps": unchanged_steps,
                    "pct_unchanged": (100.0 * unchanged_steps / analyzable) if analyzable else np.nan,
                    "avg_edit_distance_when_changed": changed_only["edit_distance"].mean() if not changed_only.empty else np.nan,
                    "median_edit_distance_when_changed": changed_only["edit_distance"].median() if not changed_only.empty else np.nan,
                    "avg_normalized_edit_when_changed": changed_only["normalized_edit_distance"].mean() if not changed_only.empty else np.nan,
                }
            ]
        )

    observed = (
        df.groupby(group_cols, dropna=False)
        .size()
        .rename("observed_transitions")
        .reset_index()
    )

    analyzable_df = df[df["is_comparable"] == True].copy()
    if analyzable_df.empty:
        summary = observed.copy()
        summary["analyzable_transitions"] = 0
        summary["non_comparable_transitions"] = summary["observed_transitions"]
        summary["changed_steps"] = 0
        summary["unchanged_steps"] = 0
        summary["pct_unchanged"] = np.nan
        summary["avg_edit_distance_when_changed"] = np.nan
        summary["median_edit_distance_when_changed"] = np.nan
        summary["avg_normalized_edit_when_changed"] = np.nan
        return summary

    analyzable_df["changed_int"] = analyzable_df["changed"].astype(int)
    analyzable_df["unchanged_int"] = 1 - analyzable_df["changed_int"]
    analyzable_df["edit_distance_when_changed"] = np.where(
        analyzable_df["changed"], analyzable_df["edit_distance"], np.nan
    )
    analyzable_df["normalized_edit_when_changed"] = np.where(
        analyzable_df["changed"], analyzable_df["normalized_edit_distance"], np.nan
    )

    aggregated = (
        analyzable_df.groupby(group_cols, dropna=False)
        .agg(
            analyzable_transitions=("is_comparable", "size"),
            changed_steps=("changed_int", "sum"),
            unchanged_steps=("unchanged_int", "sum"),
            avg_edit_distance_when_changed=("edit_distance_when_changed", "mean"),
            median_edit_distance_when_changed=("edit_distance_when_changed", "median"),
            avg_normalized_edit_when_changed=("normalized_edit_when_changed", "mean"),
        )
        .reset_index()
    )

    summary = observed.merge(aggregated, on=group_cols, how="left")
    for col in ["analyzable_transitions", "changed_steps", "unchanged_steps"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["non_comparable_transitions"] = (
        summary["observed_transitions"] - summary["analyzable_transitions"]
    )
    summary["pct_unchanged"] = np.where(
        summary["analyzable_transitions"] > 0,
        100.0 * summary["unchanged_steps"] / summary["analyzable_transitions"],
        np.nan,
    )
    summary = summary.sort_values(group_cols).reset_index(drop=True)
    return summary


class SourceLengthResolver:
    """
    Resolve source (untranslated) sample character lengths from dataset files.
    """

    def __init__(self, base_data_dir: Path):
        self.base_data_dir = base_data_dir.resolve()
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @staticmethod
    def _norm_lang_pair(lang_pair: str) -> str:
        return lang_pair.replace("-", "_")

    @staticmethod
    def _source_lang_from_pair(lang_pair: str) -> Optional[str]:
        if "_" in lang_pair:
            parts = lang_pair.split("_", 1)
        elif "-" in lang_pair:
            parts = lang_pair.split("-", 1)
        else:
            return None
        if len(parts) != 2:
            return None
        return parts[0]

    @staticmethod
    def _lang_key_for_sample(source_lang: str) -> str:
        # WMT25 files use "zh" key for traditional Chinese.
        if source_lang == "zht":
            return "zh"
        return source_lang

    def _build_cache(self, dataset: str, lang_pair: str) -> Dict[str, Any]:
        key = (dataset, lang_pair)
        if key in self._cache:
            return self._cache[key]

        by_sample_id: Dict[str, int] = {}
        by_pair_index: Dict[int, int] = {}
        source_lang = self._source_lang_from_pair(lang_pair)
        if source_lang is None:
            cache_obj = {"by_sample_id": by_sample_id, "by_pair_index": by_pair_index}
            self._cache[key] = cache_obj
            return cache_obj

        source_key = self._lang_key_for_sample(source_lang)

        if dataset == "dolfin":
            norm_pair = self._norm_lang_pair(lang_pair)
            file_path = self.base_data_dir / "dolfin" / f"dolfin_test_{norm_pair}.jsonl"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    pair_idx = 0
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        source_text = sample.get(source_key, "")
                        char_len = len(source_text) if isinstance(source_text, str) else 0
                        sample_id = sample.get("id") or sample.get("_id")
                        if sample_id is not None:
                            by_sample_id[str(sample_id)] = char_len
                        by_pair_index[pair_idx] = char_len
                        pair_idx += 1

        elif dataset == "wmt25":
            data_dir = self.base_data_dir / "wmt25-terminology-track2"
            # WMT25 direction by year:
            # odd years: en->zht ; even years: zht->en
            src = source_lang
            years: List[int]
            if src == "en":
                years = [y for y in WMT25_YEARS if y % 2 == 1]
            elif src in {"zh", "zht"}:
                years = [y for y in WMT25_YEARS if y % 2 == 0]
            else:
                years = WMT25_YEARS

            pair_idx = 0
            for year in years:
                file_path = data_dir / f"full_data_{year}.jsonl"
                if not file_path.exists():
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        source_text = sample.get(source_key, "")
                        char_len = len(source_text) if isinstance(source_text, str) else 0
                        sample_id = sample.get("id") or sample.get("_id")
                        if sample_id is not None:
                            by_sample_id[str(sample_id)] = char_len
                        by_pair_index[pair_idx] = char_len
                        pair_idx += 1

        cache_obj = {"by_sample_id": by_sample_id, "by_pair_index": by_pair_index}
        self._cache[key] = cache_obj
        return cache_obj

    def get_source_char_len(
        self,
        dataset: str,
        lang_pair: str,
        sample_id: Any,
        sample_idx: Any,
    ) -> Optional[int]:
        cache = self._build_cache(dataset, lang_pair)
        by_sample_id = cache["by_sample_id"]
        by_pair_index = cache["by_pair_index"]

        sample_id_key = str(sample_id) if sample_id is not None else None
        if sample_id_key is not None and sample_id_key in by_sample_id:
            return by_sample_id[sample_id_key]

        idx = _safe_int(sample_idx, -1)
        if idx >= 0 and idx in by_pair_index:
            return by_pair_index[idx]

        return None


def build_workflow_agent_langpair_json(
    steps_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    source_len_resolver: SourceLengthResolver,
) -> Dict[str, Any]:
    """
    Build nested JSON structure:
      workflow -> Agent N -> language_pair -> metrics

    For each workflow/agent/language_pair, include:
      - step_change_rate (0..1, based on comparable transitions)
      - avg_char_change (mean edit distance across comparable transitions, includes zeros)
      - char_changes (list of per-sample edit distances)
      - char_changes_by_sample (sample_id/sample_idx/edit_distance triples)
    """
    result: Dict[str, Any] = {}
    if steps_df.empty:
        return result

    # Use only unique workflow-step metadata (step order is workflow-relative).
    step_meta = (
        steps_df.groupby(
            ["workflow", "step_order", "step_name", "agent_type", "translation_mode", "output_index"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_rows")
        .sort_values(["workflow", "step_order"])
    )

    for workflow in sorted(step_meta["workflow"].dropna().unique()):
        wf_steps = step_meta[step_meta["workflow"] == workflow].copy()
        wf_lang_pairs = sorted(
            steps_df.loc[steps_df["workflow"] == workflow, "lang_pair"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        workflow_dict: Dict[str, Any] = {}
        for _, step in wf_steps.iterrows():
            step_order = int(step["step_order"])
            agent_label = f"Agent {step_order + 1}"
            step_name = str(step["step_name"])
            agent_type = str(step["agent_type"])
            translation_mode = str(step["translation_mode"])
            output_index = int(step["output_index"])

            agent_entry: Dict[str, Any] = {
                "step_name": step_name,
                "agent_type": agent_type,
                "step_order": step_order,
                "output_index": output_index,
                "translation_mode": translation_mode,
                "language_pairs": {},
            }

            for lang_pair in wf_lang_pairs:
                subset = transitions_df[
                    (transitions_df["workflow"] == workflow)
                    & (transitions_df["lang_pair"] == lang_pair)
                    & (transitions_df["transition_order"] == step_order)
                    & (transitions_df["is_comparable"] == True)
                ].copy()

                if not subset.empty:
                    subset = subset.sort_values(["sample_idx", "sample_id"])
                    edit_values = [int(v) for v in subset["edit_distance"].dropna().tolist()]
                    changed_values = [v for v in edit_values if v > 0]
                    comparable_samples = len(edit_values)
                    changed_samples = len(changed_values)
                    step_change_rate = (
                        float(changed_samples) / float(comparable_samples)
                        if comparable_samples > 0
                        else None
                    )
                    avg_char_change = (
                        float(np.mean(edit_values)) if comparable_samples > 0 else None
                    )
                    avg_char_change_when_changed = (
                        float(np.mean(changed_values)) if changed_values else None
                    )

                    char_changes_by_sample = [
                        {
                            "dataset": str(row["dataset"]),
                            "lang_pair": str(row["lang_pair"]),
                            "sample_idx": int(row["sample_idx"]),
                            "sample_id": str(row["sample_id"]),
                            "char_change": int(row["edit_distance"]),
                            "prev_char_len": (
                                int(row["prev_translation_len_chars"])
                                if pd.notna(row["prev_translation_len_chars"])
                                else None
                            ),
                            "curr_char_len": (
                                int(row["curr_translation_len_chars"])
                                if pd.notna(row["curr_translation_len_chars"])
                                else None
                            ),
                            "sample_char_len": (
                                source_len_resolver.get_source_char_len(
                                    dataset=str(row["dataset"]),
                                    lang_pair=str(row["lang_pair"]),
                                    sample_id=row["sample_id"],
                                    sample_idx=row["sample_idx"],
                                )
                            ),
                            "char_len_delta": (
                                int(row["curr_translation_len_chars"]) - int(row["prev_translation_len_chars"])
                                if pd.notna(row["curr_translation_len_chars"]) and pd.notna(row["prev_translation_len_chars"])
                                else None
                            ),
                        }
                        for _, row in subset.iterrows()
                        if pd.notna(row["edit_distance"])
                    ]
                else:
                    edit_values = []
                    comparable_samples = 0
                    changed_samples = 0
                    step_change_rate = None
                    avg_char_change = None
                    avg_char_change_when_changed = None
                    char_changes_by_sample = []

                agent_entry["language_pairs"][lang_pair] = {
                    "comparable_samples": comparable_samples,
                    "changed_samples": changed_samples,
                    "step_change_rate": step_change_rate,
                    "avg_char_change": avg_char_change,
                    "avg_char_change_when_changed": avg_char_change_when_changed,
                    "char_changes": edit_values,
                    "char_changes_by_sample": char_changes_by_sample,
                }

            workflow_dict[agent_label] = agent_entry

        result[str(workflow)] = workflow_dict

    return result


def analyze(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_filter = set(args.datasets) if args.datasets else None
    workflows_filter = set(args.workflows) if args.workflows else None
    models_filter = set(args.models) if args.models else None

    stats: Dict[str, Any] = {
        "edit_distance_impl": EDIT_DISTANCE_IMPL,
        "reports_discovered": 0,
        "reports_processed": 0,
        "reports_skipped_incomplete": 0,
        "samples_seen": 0,
        "samples_processed": 0,
        "samples_missing_outputs": 0,
    }

    step_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []

    outputs_dirs = [Path(p).resolve() for p in args.outputs_dirs]
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            continue

        report_files = discover_report_files(outputs_dir)
        stats["reports_discovered"] += len(report_files)

        for report_path in report_files:
            metadata = parse_experiment_metadata(outputs_dir, report_path)
            if metadata is None:
                continue

            try:
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            workflow_name = str(report_data.get("workflow", "")).strip()
            if not workflow_name:
                workflow_name = metadata["workflow_acronym"]

            if not _matches_filters(
                metadata,
                workflow_name,
                datasets_filter,
                workflows_filter,
                models_filter,
            ):
                continue

            total_samples = _safe_int(report_data.get("total_samples"), 0)
            successful_samples = _safe_int(report_data.get("successful_samples"), 0)
            if args.require_complete_reports and (
                total_samples <= 0 or successful_samples != total_samples
            ):
                stats["reports_skipped_incomplete"] += 1
                continue

            samples = report_data.get("samples", [])
            if not isinstance(samples, list):
                continue

            stats["reports_processed"] += 1
            model_dir = report_path.parent

            for sample in samples:
                stats["samples_seen"] += 1
                if not isinstance(sample, dict):
                    continue
                if sample.get("error"):
                    continue

                sample_idx = _safe_int(sample.get("sample_idx"), 0)
                sample_id = sample.get("sample_id", str(sample_idx))
                sample_prefix = _sample_file_prefix(sample_id, sample_idx)
                outputs = load_agent_outputs_for_sample(model_dir, sample_prefix)
                if not outputs:
                    stats["samples_missing_outputs"] += 1
                    continue

                specs = resolve_workflow_step_specs(workflow_name, len(outputs))
                if not specs:
                    continue

                stats["samples_processed"] += 1
                sample_step_rows: List[Dict[str, Any]] = []
                previous_translation_state: Optional[str] = None

                for step_order, spec in enumerate(specs):
                    state = derive_translation_state(
                        mode=spec.translation_mode,
                        output_index=spec.output_index,
                        outputs=outputs,
                        previous_translation_state=previous_translation_state,
                    )
                    if state is not None:
                        previous_translation_state = state

                    row = {
                        "dataset": metadata["dataset"],
                        "lang_pair": metadata["lang_pair"],
                        "workflow": workflow_name,
                        "workflow_dir": metadata["workflow_dir"],
                        "workflow_acronym": metadata["workflow_acronym"],
                        "model": metadata["model"],
                        "report_file": str(report_path),
                        "sample_idx": sample_idx,
                        "sample_id": str(sample_id),
                        "step_order": step_order,
                        "step_name": spec.step_name,
                        "agent_type": spec.agent_type,
                        "translation_mode": spec.translation_mode,
                        "output_index": spec.output_index,
                        "translation_state": state,
                        "translation_available": state is not None,
                        "translation_state_hash": (
                            hashlib.sha1(state.encode("utf-8")).hexdigest() if state is not None else None
                        ),
                        "translation_len_chars": len(state) if state is not None else np.nan,
                    }
                    sample_step_rows.append(row)
                    step_rows.append(
                        {
                            k: v
                            for k, v in row.items()
                            if k != "translation_state"
                        }
                    )

                for i in range(1, len(sample_step_rows)):
                    prev_row = sample_step_rows[i - 1]
                    curr_row = sample_step_rows[i]
                    prev_state = prev_row["translation_state"]
                    curr_state = curr_row["translation_state"]

                    is_comparable = prev_state is not None and curr_state is not None
                    changed: Optional[bool] = None
                    edit_distance: Optional[int] = None
                    normalized_edit_distance: Optional[float] = None

                    if is_comparable:
                        edit_distance = char_edit_distance(prev_state, curr_state)
                        changed = edit_distance != 0
                        max_len = max(len(prev_state), len(curr_state), 1)
                        normalized_edit_distance = float(edit_distance) / float(max_len)

                    transition_rows.append(
                        {
                            "dataset": metadata["dataset"],
                            "lang_pair": metadata["lang_pair"],
                            "workflow": workflow_name,
                            "workflow_dir": metadata["workflow_dir"],
                            "workflow_acronym": metadata["workflow_acronym"],
                            "model": metadata["model"],
                            "report_file": str(report_path),
                            "sample_idx": sample_idx,
                            "sample_id": str(sample_id),
                            "transition_order": i,
                            "prev_step_name": prev_row["step_name"],
                            "curr_step_name": curr_row["step_name"],
                            "prev_agent_type": prev_row["agent_type"],
                            "curr_agent_type": curr_row["agent_type"],
                            "prev_translation_mode": prev_row["translation_mode"],
                            "curr_translation_mode": curr_row["translation_mode"],
                            "is_comparable": is_comparable,
                            "changed": changed,
                            "unchanged": (not changed) if changed is not None else None,
                            "edit_distance": edit_distance,
                            "normalized_edit_distance": normalized_edit_distance,
                            "prev_translation_len_chars": (
                                len(prev_state) if prev_state is not None else np.nan
                            ),
                            "curr_translation_len_chars": (
                                len(curr_state) if curr_state is not None else np.nan
                            ),
                        }
                    )

    steps_df = pd.DataFrame(step_rows)
    transitions_df = pd.DataFrame(transition_rows)

    step_file = output_dir / "step_states_raw.csv"
    transition_file = output_dir / "transitions_raw.csv"
    steps_df.to_csv(step_file, index=False)
    transitions_df.to_csv(transition_file, index=False)

    by_setting_cols = ["dataset", "lang_pair", "workflow", "model"]
    by_setting_agent_cols = by_setting_cols + ["curr_agent_type"]
    by_setting_step_cols = by_setting_cols + ["curr_step_name"]

    summary_by_setting = summarize_transitions(transitions_df, by_setting_cols)
    summary_by_agent = summarize_transitions(transitions_df, by_setting_agent_cols)
    summary_by_step = summarize_transitions(transitions_df, by_setting_step_cols)
    summary_global_by_agent = summarize_transitions(transitions_df, ["curr_agent_type"])
    summary_global = summarize_transitions(transitions_df, [])

    summary_by_setting.to_csv(output_dir / "summary_by_setting.csv", index=False)
    summary_by_agent.to_csv(output_dir / "summary_by_agent_type.csv", index=False)
    summary_by_step.to_csv(output_dir / "summary_by_step.csv", index=False)
    summary_global_by_agent.to_csv(output_dir / "summary_global_by_agent_type.csv", index=False)
    summary_global.to_csv(output_dir / "summary_global_overall.csv", index=False)

    # Requested nested JSON artifact: workflow -> Agent N -> language pair.
    source_len_resolver = SourceLengthResolver(base_data_dir=Path(args.data_dir))
    nested_metrics = build_workflow_agent_langpair_json(
        steps_df=steps_df,
        transitions_df=transitions_df,
        source_len_resolver=source_len_resolver,
    )
    nested_json_file = output_dir / "workflow_agent_langpair_metrics.json"
    nested_json_file.write_text(
        json.dumps(nested_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats["step_rows"] = int(len(steps_df))
    stats["transition_rows"] = int(len(transitions_df))
    stats["comparable_transition_rows"] = int(
        transitions_df["is_comparable"].sum() if not transitions_df.empty else 0
    )
    stats["changed_transition_rows"] = int(
        transitions_df["changed"].fillna(False).sum() if not transitions_df.empty else 0
    )
    stats["unchanged_transition_rows"] = int(
        transitions_df["unchanged"].fillna(False).sum() if not transitions_df.empty else 0
    )

    metadata_file = output_dir / "analysis_metadata.json"
    metadata_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Agent Contribution Analysis")
    print("=" * 80)
    print(f"Edit distance implementation: {stats['edit_distance_impl']}")
    print(f"Reports discovered: {stats['reports_discovered']}")
    print(f"Reports processed: {stats['reports_processed']}")
    if args.require_complete_reports:
        print(f"Reports skipped (incomplete): {stats['reports_skipped_incomplete']}")
    print(f"Samples seen: {stats['samples_seen']}")
    print(f"Samples processed: {stats['samples_processed']}")
    print(f"Samples missing outputs: {stats['samples_missing_outputs']}")
    print(f"Step rows: {stats['step_rows']}")
    print(f"Transition rows: {stats['transition_rows']}")
    print(f"Comparable transitions: {stats['comparable_transition_rows']}")
    print(f"Changed transitions: {stats['changed_transition_rows']}")
    print(f"Unchanged transitions: {stats['unchanged_transition_rows']}")
    print(f"\nWrote raw step states to: {step_file}")
    print(f"Wrote raw transitions to: {transition_file}")
    print(f"Wrote nested workflow-step-language metrics JSON to: {nested_json_file}")
    print(f"Wrote summaries to: {output_dir}")

    return 0


def main() -> int:
    args = parse_args()
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())

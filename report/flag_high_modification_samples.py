#!/usr/bin/env python3
"""
Flag samples with unusually high per-step character changes.

Reads workflow_agent_langpair_metrics.json and marks samples where:
    char_change > multiplier * mean_char_change

Outputs:
  1) JSON summary grouped by workflow/model/agent/lang_pair
  2) Flat CSV of flagged samples

Usage:
  python report/flag_high_modification_samples.py
  python report/flag_high_modification_samples.py --multiplier 2.0
  python report/flag_high_modification_samples.py \
      --workflow MaMT_translate_postedit_proofread \
      --model gpt-4-1 \
      --lang_pair en_it
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag high-modification samples")
    parser.add_argument(
        "--input_json",
        default="report/contribution_analysis/workflow_agent_langpair_metrics.json",
        help="Input metrics JSON from analyze_agent_contributions.py",
    )
    parser.add_argument(
        "--output_json",
        default="report/contribution_analysis/high_modification_flags.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output_csv",
        default="report/contribution_analysis/high_modification_flags.csv",
        help="Output CSV path (flat list of flagged samples)",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=2.0,
        help="Flag threshold multiplier over mean char_change (default: 2.0)",
    )
    parser.add_argument(
        "--min_char_change",
        type=int,
        default=0,
        help="Optional minimum absolute char_change required for flagging",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Optional workflow filter",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model filter",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Optional agent filter (e.g., 'Agent 2')",
    )
    parser.add_argument(
        "--lang_pair",
        default=None,
        help="Optional language pair filter (e.g., en_it)",
    )
    return parser.parse_args()


def _matches_filters(
    workflow: str,
    model: str,
    agent: str,
    lang_pair: str,
    args: argparse.Namespace,
) -> bool:
    if args.workflow and workflow != args.workflow:
        return False
    if args.model and model != args.model:
        return False
    if args.agent and agent != args.agent:
        return False
    if args.lang_pair and lang_pair != args.lang_pair:
        return False
    return True


def process(
    input_data: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "criteria": {
            "rule": "char_change > multiplier * mean_char_change AND char_change >= min_char_change",
            "multiplier": args.multiplier,
            "min_char_change": args.min_char_change,
        },
        "summary": {
            "groups_processed": 0,
            "groups_with_flags": 0,
            "total_samples_seen": 0,
            "total_flagged_samples": 0,
        },
        "workflows": {},
        "flagged_samples_flat": [],
    }

    for workflow, workflow_data in input_data.items():
        if not isinstance(workflow_data, dict):
            continue

        workflow_out: Dict[str, Any] = {}
        for model, model_data in workflow_data.items():
            if not isinstance(model_data, dict):
                continue

            model_out: Dict[str, Any] = {}
            for agent, agent_data in model_data.items():
                if not isinstance(agent_data, dict):
                    continue

                language_pairs = agent_data.get("language_pairs", {})
                if not isinstance(language_pairs, dict):
                    continue

                agent_out: Dict[str, Any] = {
                    "step_name": agent_data.get("step_name"),
                    "agent_type": agent_data.get("agent_type"),
                    "language_pairs": {},
                }

                for lang_pair, lp_data in language_pairs.items():
                    if not _matches_filters(workflow, model, agent, lang_pair, args):
                        continue

                    char_changes_by_sample = lp_data.get("char_changes_by_sample", [])
                    if not isinstance(char_changes_by_sample, list):
                        continue

                    values = [
                        int(s.get("char_change", 0))
                        for s in char_changes_by_sample
                        if isinstance(s, dict) and s.get("char_change") is not None
                    ]

                    if not values:
                        continue

                    mean_change = float(mean(values))
                    threshold = float(args.multiplier * mean_change)

                    flagged_samples: List[Dict[str, Any]] = []
                    for sample in char_changes_by_sample:
                        if not isinstance(sample, dict):
                            continue
                        char_change = sample.get("char_change")
                        if char_change is None:
                            continue
                        char_change = int(char_change)
                        if char_change > threshold and char_change >= args.min_char_change:
                            ratio = (float(char_change) / mean_change) if mean_change > 0 else None
                            item = {
                                "sample_idx": sample.get("sample_idx"),
                                "sample_id": sample.get("sample_id"),
                                "char_change": char_change,
                                "sample_char_len": sample.get("sample_char_len"),
                                "prev_char_len": sample.get("prev_char_len"),
                                "curr_char_len": sample.get("curr_char_len"),
                                "char_len_delta": sample.get("char_len_delta"),
                                "change_to_mean_ratio": ratio,
                            }
                            flagged_samples.append(item)
                            result["flagged_samples_flat"].append(
                                {
                                    "workflow": workflow,
                                    "model": model,
                                    "agent": agent,
                                    "step_name": agent_data.get("step_name"),
                                    "agent_type": agent_data.get("agent_type"),
                                    "lang_pair": lang_pair,
                                    **item,
                                }
                            )

                    result["summary"]["groups_processed"] += 1
                    result["summary"]["total_samples_seen"] += len(values)
                    result["summary"]["total_flagged_samples"] += len(flagged_samples)
                    if flagged_samples:
                        result["summary"]["groups_with_flags"] += 1

                    agent_out["language_pairs"][lang_pair] = {
                        "num_samples": len(values),
                        "mean_char_change": mean_change,
                        "threshold_char_change": threshold,
                        "num_flagged_samples": len(flagged_samples),
                        "flagged_samples": sorted(
                            flagged_samples,
                            key=lambda x: x.get("change_to_mean_ratio") or 0.0,
                            reverse=True,
                        ),
                    }

                if agent_out["language_pairs"]:
                    model_out[agent] = agent_out

            if model_out:
                workflow_out[model] = model_out

        if workflow_out:
            result["workflows"][workflow] = workflow_out

    # Sort flat list by severity ratio.
    result["flagged_samples_flat"] = sorted(
        result["flagged_samples_flat"],
        key=lambda x: x.get("change_to_mean_ratio") or 0.0,
        reverse=True,
    )
    return result


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workflow",
        "model",
        "agent",
        "step_name",
        "agent_type",
        "lang_pair",
        "sample_idx",
        "sample_id",
        "char_change",
        "sample_char_len",
        "prev_char_len",
        "curr_char_len",
        "char_len_delta",
        "change_to_mean_ratio",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()

    if not input_json.exists():
        raise SystemExit(f"Input file not found: {input_json}")

    with input_json.open("r", encoding="utf-8") as f:
        input_data = json.load(f)

    result = process(input_data, args)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    write_csv(result["flagged_samples_flat"], output_csv)

    summary = result["summary"]
    print("=" * 80)
    print("High Modification Flagging")
    print("=" * 80)
    print(f"Input: {input_json}")
    print(f"Output JSON: {output_json}")
    print(f"Output CSV: {output_csv}")
    print(f"Rule: char_change > {args.multiplier} * mean_char_change and >= {args.min_char_change}")
    print(f"Groups processed: {summary['groups_processed']}")
    print(f"Groups with flags: {summary['groups_with_flags']}")
    print(f"Total samples seen: {summary['total_samples_seen']}")
    print(f"Total flagged samples: {summary['total_flagged_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

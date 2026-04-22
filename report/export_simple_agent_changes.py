#!/usr/bin/env python3
"""
Export a compact view of per-agent character changes from
workflow_agent_langpair_metrics.json.

Output shape (one block per workflow+model+language_pair):
{
  "workflow": "...",
  "model": "...",
  "language_pair": "...",
  "agents_in_workflow": ["Agent 1", "Agent 2", ...],
  "num_samples_in_language_pair": 13,
  "agent_char_changes": {
    "Agent 1": [...],
    "Agent 2": [...],
    ...
  }
}

Usage:
  python report/export_simple_agent_changes.py
  python report/export_simple_agent_changes.py --workflow MaMT_translate_postedit_proofread --model gpt-4-1 --lang_pair en_it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export simple per-agent char-change summaries")
    parser.add_argument(
        "--input_json",
        default="report/contribution_analysis/workflow_agent_langpair_metrics.json",
        help="Input metrics JSON from analyze_agent_contributions.py",
    )
    parser.add_argument(
        "--output_json",
        default="report/contribution_analysis/simple_agent_changes.json",
        help="Output JSON path",
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
        "--lang_pair",
        default=None,
        help="Optional language pair filter",
    )
    parser.add_argument(
        "--missing_value",
        default="null",
        choices=["null", "zero"],
        help="How to fill missing sample values for agents with no comparable changes",
    )
    return parser.parse_args()


def _matches_filters(
    workflow: str, model: str, lang_pair: str, args: argparse.Namespace
) -> bool:
    if args.workflow and workflow != args.workflow:
        return False
    if args.model and model != args.model:
        return False
    if args.lang_pair and lang_pair != args.lang_pair:
        return False
    return True


def _sample_key(sample: Dict[str, Any]) -> Tuple[int, str]:
    sample_idx = sample.get("sample_idx")
    if sample_idx is None:
        sample_idx = 10**12
    try:
        sample_idx = int(sample_idx)
    except Exception:
        sample_idx = 10**12
    sample_id = str(sample.get("sample_id", ""))
    return (sample_idx, sample_id)


def _fill_value(missing_mode: str) -> Optional[int]:
    if missing_mode == "zero":
        return 0
    return None


def build_simple_blocks(
    metrics: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    settings_count = 0

    for workflow, workflow_data in metrics.items():
        if not isinstance(workflow_data, dict):
            continue

        for model, model_data in workflow_data.items():
            if not isinstance(model_data, dict):
                continue

            # Collect all language pairs available under this workflow/model.
            lang_pairs = set()
            for _agent, agent_data in model_data.items():
                lp_map = agent_data.get("language_pairs", {})
                if isinstance(lp_map, dict):
                    lang_pairs.update(lp_map.keys())

            for lang_pair in sorted(lang_pairs):
                if not _matches_filters(workflow, model, lang_pair, args):
                    continue

                # Collect the union of sample keys across all agents for alignment.
                sample_keys = set()
                for _agent, agent_data in model_data.items():
                    lp_data = agent_data.get("language_pairs", {}).get(lang_pair, {})
                    for sample in lp_data.get("char_changes_by_sample", []):
                        if isinstance(sample, dict):
                            sample_keys.add(_sample_key(sample))

                ordered_sample_keys = sorted(sample_keys)
                num_samples = len(ordered_sample_keys)

                agents_in_workflow = sorted(
                    [a for a in model_data.keys()],
                    key=lambda x: int(x.split()[-1]) if x.startswith("Agent ") and x.split()[-1].isdigit() else 10**9,
                )

                fill = _fill_value(args.missing_value)
                agent_char_changes: Dict[str, List[Optional[int]]] = {}

                for agent in agents_in_workflow:
                    agent_data = model_data.get(agent, {})
                    lp_data = agent_data.get("language_pairs", {}).get(lang_pair, {})
                    sample_rows = lp_data.get("char_changes_by_sample", [])
                    sample_to_change: Dict[Tuple[int, str], Optional[int]] = {}
                    if isinstance(sample_rows, list):
                        for row in sample_rows:
                            if not isinstance(row, dict):
                                continue
                            key = _sample_key(row)
                            char_change = row.get("char_change")
                            if char_change is None:
                                sample_to_change[key] = fill
                            else:
                                try:
                                    sample_to_change[key] = int(char_change)
                                except Exception:
                                    sample_to_change[key] = fill

                    values = [sample_to_change.get(k, fill) for k in ordered_sample_keys]
                    agent_char_changes[agent] = values

                blocks.append(
                    {
                        "workflow": workflow,
                        "model": model,
                        "language_pair": lang_pair,
                        "agents_in_workflow": agents_in_workflow,
                        "num_samples_in_language_pair": num_samples,
                        "sample_order": [
                            {"sample_idx": idx, "sample_id": sid}
                            for idx, sid in ordered_sample_keys
                        ],
                        "agent_char_changes": agent_char_changes,
                    }
                )
                settings_count += 1

    return {
        "settings_count": settings_count,
        "missing_value_mode": args.missing_value,
        "blocks": blocks,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    result = build_simple_blocks(metrics, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Simple Agent Change Export")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Blocks written: {result['settings_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

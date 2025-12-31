#!/usr/bin/env python3
"""
Sanity check script for report.json files.

This script verifies that:
1. Token totals in summary match the sum of tokens from all successful samples
2. Score averages in summary match the average of scores from all successful samples
3. No duplicate samples (same sample_id or sample_idx+lang_pair)
4. No samples with errors that have token counts (should be 0 or None)
5. Sample counts match (total_samples, successful_samples, failed_samples)

Usage:
    python run/sanity_check_reports.py --outputs_dirs zhijin/agent-mt-main/outputs outputs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


def parse_report(report_path: Path) -> Optional[Dict]:
    """Parse a report.json file."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error reading {report_path}: {e}")
        return None


def check_report(report_path: Path, report: Dict) -> Tuple[bool, List[str]]:
    """
    Check a single report for consistency issues.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    dataset = report.get("dataset", "unknown")
    workflow = report.get("workflow", "unknown")
    model = report.get("model", "unknown")
    
    samples = report.get("samples", [])
    summary = report.get("summary", {})
    
    # Check sample counts
    total_samples = report.get("total_samples", 0)
    successful_samples = report.get("successful_samples", 0)
    failed_samples = report.get("failed_samples", 0)
    
    actual_total = len(samples)
    actual_successful = sum(1 for s in samples if not s.get("error"))
    actual_failed = sum(1 for s in samples if s.get("error"))
    
    if actual_total != total_samples:
        issues.append(f"  Sample count mismatch: total_samples={total_samples} but len(samples)={actual_total}")
    if actual_successful != successful_samples:
        issues.append(f"  Successful count mismatch: successful_samples={successful_samples} but actual={actual_successful}")
    if actual_failed != failed_samples:
        issues.append(f"  Failed count mismatch: failed_samples={failed_samples} but actual={actual_failed}")
    
    # Check for duplicate samples
    seen_sample_ids: Set[str] = set()
    seen_sample_keys: Set[str] = set()
    duplicates = []
    
    for i, sample in enumerate(samples):
        sample_id = sample.get("sample_id")
        sample_idx = sample.get("sample_idx")
        lang_pair = sample.get("lang_pair", "")
        
        # Check by sample_id
        if sample_id:
            sample_key = str(sample_id)
            if sample_key in seen_sample_ids:
                duplicates.append(f"    Sample index {i}: duplicate sample_id='{sample_id}'")
            seen_sample_ids.add(sample_key)
        
        # Check by sample_idx + lang_pair
        if sample_idx is not None:
            sample_key = f"{lang_pair}_{sample_idx}"
            if sample_key in seen_sample_keys:
                duplicates.append(f"    Sample index {i}: duplicate (lang_pair='{lang_pair}', sample_idx={sample_idx})")
            seen_sample_keys.add(sample_key)
    
    if duplicates:
        issues.append(f"  Found {len(duplicates)} duplicate sample(s):")
        issues.extend(duplicates)
    
    # Recalculate totals from successful samples only
    recalc_tokens_input = 0
    recalc_tokens_output = 0
    recalc_base_tokens_input = 0
    recalc_base_tokens_output = 0
    recalc_latency = 0.0
    recalc_chrf_scores = []
    recalc_bleu_scores = []
    recalc_term_success_rates = []
    
    samples_with_errors_but_tokens = []
    
    for i, sample in enumerate(samples):
        error = sample.get("error")
        has_error = error is not None and error != ""
        
        tokens_input = sample.get("tokens_input", 0) or 0
        tokens_output = sample.get("tokens_output", 0) or 0
        
        # Check if sample with error has tokens (should be 0 or None)
        if has_error and (tokens_input > 0 or tokens_output > 0):
            samples_with_errors_but_tokens.append(
                f"    Sample index {i} (sample_id={sample.get('sample_id')}, sample_idx={sample.get('sample_idx')}): "
                f"has error but tokens_input={tokens_input}, tokens_output={tokens_output}"
            )
        
        # Only count successful samples
        if not has_error:
            recalc_tokens_input += tokens_input
            recalc_tokens_output += tokens_output
            recalc_base_tokens_input += sample.get("base_model_tokens_input", 0) or 0
            recalc_base_tokens_output += sample.get("base_model_tokens_output", 0) or 0
            
            if sample.get("latency"):
                recalc_latency += sample.get("latency", 0.0)
            
            # Collect scores
            if sample.get("chrf_scores"):
                chrf_list = sample["chrf_scores"]
                if isinstance(chrf_list, list) and len(chrf_list) > 0:
                    recalc_chrf_scores.append(chrf_list[-1])  # Use last agent's score
            
            if sample.get("bleu_scores"):
                bleu_list = sample["bleu_scores"]
                if isinstance(bleu_list, list) and len(bleu_list) > 0:
                    last_bleu = bleu_list[-1]
                    if last_bleu is not None:
                        recalc_bleu_scores.append(last_bleu)
            
            if sample.get("term_success_rates"):
                term_list = sample["term_success_rates"]
                if isinstance(term_list, list) and len(term_list) > 0:
                    last_term = term_list[-1]
                    if last_term is not None and last_term >= 0:
                        recalc_term_success_rates.append(last_term)
    
    if samples_with_errors_but_tokens:
        issues.append(f"  Found {len(samples_with_errors_but_tokens)} sample(s) with errors but non-zero tokens:")
        issues.extend(samples_with_errors_but_tokens)
    
    # Compare recalculated totals with summary
    summary_tokens_input = summary.get("total_tokens_input", 0) or 0
    summary_tokens_output = summary.get("total_tokens_output", 0) or 0
    summary_base_tokens_input = summary.get("total_base_model_tokens_input", 0) or 0
    summary_base_tokens_output = summary.get("total_base_model_tokens_output", 0) or 0
    summary_latency = summary.get("total_latency_seconds", 0.0) or 0.0
    
    if abs(recalc_tokens_input - summary_tokens_input) > 1:  # Allow 1 token difference for rounding
        issues.append(
            f"  Token input mismatch: summary={summary_tokens_input}, recalculated={recalc_tokens_input} "
            f"(diff={recalc_tokens_input - summary_tokens_input})"
        )
    
    if abs(recalc_tokens_output - summary_tokens_output) > 1:
        issues.append(
            f"  Token output mismatch: summary={summary_tokens_output}, recalculated={recalc_tokens_output} "
            f"(diff={recalc_tokens_output - summary_tokens_output})"
        )
    
    if abs(recalc_base_tokens_input - summary_base_tokens_input) > 1:
        issues.append(
            f"  Base model token input mismatch: summary={summary_base_tokens_input}, "
            f"recalculated={recalc_base_tokens_input} (diff={recalc_base_tokens_input - summary_base_tokens_input})"
        )
    
    if abs(recalc_base_tokens_output - summary_base_tokens_output) > 1:
        issues.append(
            f"  Base model token output mismatch: summary={summary_base_tokens_output}, "
            f"recalculated={recalc_base_tokens_output} (diff={recalc_base_tokens_output - summary_base_tokens_output})"
        )
    
    if abs(recalc_latency - summary_latency) > 0.01:  # Allow 0.01s difference
        issues.append(
            f"  Latency mismatch: summary={summary_latency:.2f}s, recalculated={recalc_latency:.2f}s "
            f"(diff={recalc_latency - summary_latency:.2f}s)"
        )
    
    # Check score averages
    if recalc_chrf_scores:
        recalc_avg_chrf = sum(recalc_chrf_scores) / len(recalc_chrf_scores)
        summary_avg_chrf = summary.get("avg_chrf_score")
        if summary_avg_chrf is not None:
            if abs(recalc_avg_chrf - summary_avg_chrf) > 0.001:  # Allow 0.001 difference
                issues.append(
                    f"  chrF++ average mismatch: summary={summary_avg_chrf:.4f}, "
                    f"recalculated={recalc_avg_chrf:.4f} (diff={recalc_avg_chrf - summary_avg_chrf:.4f})"
                )
        elif len(recalc_chrf_scores) > 0:
            issues.append(f"  chrF++ average missing in summary but {len(recalc_chrf_scores)} scores found")
    
    if recalc_bleu_scores:
        recalc_avg_bleu = sum(recalc_bleu_scores) / len(recalc_bleu_scores)
        summary_avg_bleu = summary.get("avg_bleu_score")
        if summary_avg_bleu is not None:
            if abs(recalc_avg_bleu - summary_avg_bleu) > 0.001:
                issues.append(
                    f"  BLEU average mismatch: summary={summary_avg_bleu:.4f}, "
                    f"recalculated={recalc_avg_bleu:.4f} (diff={recalc_avg_bleu - summary_avg_bleu:.4f})"
                )
    
    if recalc_term_success_rates:
        recalc_avg_term = sum(recalc_term_success_rates) / len(recalc_term_success_rates)
        summary_avg_term = summary.get("avg_term_success_rate")
        if summary_avg_term is not None:
            if abs(recalc_avg_term - summary_avg_term) > 0.001:
                issues.append(
                    f"  TermAcc average mismatch: summary={summary_avg_term:.4f}, "
                    f"recalculated={recalc_avg_term:.4f} (diff={recalc_avg_term - summary_avg_term:.4f})"
                )
    
    return len(issues) == 0, issues


def collect_reports(outputs_dirs: List[Path]) -> Dict[Tuple[str, str, str, str], Path]:
    """
    Collect all report.json files from output directories.
    
    Returns:
        Dictionary mapping (dataset, lang_pair, workflow, model) -> report_path
    """
    reports = {}
    
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            continue
        
        # Iterate through dataset/lang_pair/workflow/model/report.json
        for dataset_dir in outputs_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name
            
            for lang_pair_dir in dataset_dir.iterdir():
                if not lang_pair_dir.is_dir():
                    continue
                
                lang_pair = lang_pair_dir.name
                
                for workflow_dir in lang_pair_dir.iterdir():
                    if not workflow_dir.is_dir():
                        continue
                    
                    workflow = workflow_dir.name
                    
                    for model_dir in workflow_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        
                        model = model_dir.name
                        
                        report_file = model_dir / "report.json"
                        if report_file.exists():
                            key = (dataset, lang_pair, workflow, model)
                            # Use first found report if there's overlap
                            if key not in reports:
                                reports[key] = report_file
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check report.json files for consistency"
    )
    parser.add_argument(
        "--outputs_dirs",
        type=str,
        nargs='+',
        default=["zhijin/agent-mt-main/outputs", "outputs"],
        help="Paths to outputs directories containing report.json files (can specify multiple)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show details for all reports, even valid ones"
    )
    
    args = parser.parse_args()
    
    outputs_dirs = [Path(d) for d in args.outputs_dirs]
    
    # Check if at least one outputs directory exists
    existing_dirs = [d for d in outputs_dirs if d.exists()]
    if not existing_dirs:
        print("Error: None of the specified outputs directories exist:")
        for d in outputs_dirs:
            print(f"  - {d}")
        return 1
    
    print("Collecting reports...")
    print(f"Searching in {len(existing_dirs)} directory/directories:")
    for d in existing_dirs:
        print(f"  - {d}")
    
    reports = collect_reports(existing_dirs)
    print(f"\nFound {len(reports)} report.json files\n")
    
    # Check each report
    valid_count = 0
    invalid_count = 0
    issues_by_type = defaultdict(int)
    
    for (dataset, lang_pair, workflow, model), report_path in sorted(reports.items()):
        report = parse_report(report_path)
        if report is None:
            continue
        
        is_valid, issues = check_report(report_path, report)
        
        if is_valid:
            valid_count += 1
            if args.verbose:
                print(f"✓ {dataset}/{lang_pair}/{workflow}/{model}")
        else:
            invalid_count += 1
            print(f"✗ {dataset}/{lang_pair}/{workflow}/{model}")
            print(f"  Path: {report_path}")
            for issue in issues:
                print(issue)
                # Count issue types
                if "mismatch" in issue.lower():
                    issues_by_type["mismatch"] += 1
                elif "duplicate" in issue.lower():
                    issues_by_type["duplicate"] += 1
                elif "error" in issue.lower() and "token" in issue.lower():
                    issues_by_type["error_with_tokens"] += 1
            print()
    
    # Summary
    print("=" * 80)
    print("Summary:")
    print(f"  Total reports checked: {len(reports)}")
    print(f"  Valid reports: {valid_count}")
    print(f"  Invalid reports: {invalid_count}")
    
    if issues_by_type:
        print("\n  Issue types found:")
        for issue_type, count in sorted(issues_by_type.items()):
            print(f"    - {issue_type}: {count}")
    
    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    exit(main())


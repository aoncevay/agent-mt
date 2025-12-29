#!/usr/bin/env python3
"""
Check which experiments have completed successfully.

Scans the outputs/ directory and checks if report.json files indicate
that all samples were processed successfully (total_samples == successful_samples).

Usage:
    python run/check_completed_experiments.py
    python run/check_completed_experiments.py --dataset wmt25
    python run/check_completed_experiments.py --dataset wmt25 --workflow IRB_refine
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Get project root (parent of run/)
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def find_report_files(outputs_dir: Path, dataset: Optional[str] = None) -> List[Tuple[Path, Dict]]:
    """
    Find all report.json files in the outputs directory.
    
    Returns:
        List of tuples: (report_path, report_data)
    """
    reports = []
    
    if not outputs_dir.exists():
        return reports
    
    # Look for report.json files
    for report_file in outputs_dir.rglob("report.json"):
        # Parse path: outputs/{dataset}/{lang_pair}/{workflow_dir}/{model}/report.json
        parts = report_file.parts
        try:
            # Find where 'outputs' is in the path
            outputs_idx = parts.index('outputs')
            if len(parts) < outputs_idx + 5:
                continue
            
            dataset_name = parts[outputs_idx + 1]
            lang_pair = parts[outputs_idx + 2]
            workflow_dir = parts[outputs_idx + 3]
            model_name = parts[outputs_idx + 4]
            
            # Filter by dataset if specified
            if dataset and dataset_name != dataset:
                continue
            
            # Load report
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                reports.append((report_file, report_data))
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠ Warning: Could not read {report_file}: {e}")
                continue
                
        except (ValueError, IndexError):
            # Skip if path doesn't match expected structure
            continue
    
    return reports


def check_completion_status(report_data: Dict) -> Tuple[bool, int, int]:
    """
    Check if an experiment is completed.
    
    Returns:
        (is_completed, total_samples, successful_samples)
    """
    total = report_data.get('total_samples', 0)
    successful = report_data.get('successful_samples', 0)
    is_completed = (total > 0) and (total == successful)
    return is_completed, total, successful


def parse_workflow_from_dir(workflow_dir: str) -> Tuple[str, bool]:
    """
    Parse workflow name and terminology flag from directory name.
    
    Examples:
        "MaMT" -> ("MaMT_translate_postedit", False) or ("MaMT_translate_postedit_proofread", False)
        "MaMT.term" -> ("MaMT_translate_postedit", True) or ("MaMT_translate_postedit_proofread", True)
        "ZS" -> ("zero_shot", False)
        "IRB" -> ("IRB_refine", False)
    """
    use_terminology = False
    if workflow_dir.endswith('.term'):
        workflow_acronym = workflow_dir[:-5]  # Remove .term
        use_terminology = True
    else:
        workflow_acronym = workflow_dir
    
    # Reverse lookup workflow name from acronym
    # This is approximate - we'll use the workflow name from report.json if available
    return workflow_acronym, use_terminology


def generate_report(
    reports: List[Tuple[Path, Dict]],
    dataset_filter: Optional[str] = None,
    workflow_filter: Optional[str] = None,
    with_terminology: bool = False
) -> None:
    """
    Generate and print completion report.
    """
    # Organize by dataset -> workflow -> model -> lang_pair
    by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for report_path, report_data in reports:
        dataset = report_data.get('dataset', 'unknown')
        workflow = report_data.get('workflow', 'unknown')
        model = report_data.get('model', 'unknown')
        
        # Parse lang_pair and terminology status from path
        parts = report_path.parts
        use_terminology = False
        try:
            outputs_idx = parts.index('outputs')
            lang_pair = parts[outputs_idx + 2]
            workflow_dir = parts[outputs_idx + 3]
            # Check if this is a terminology experiment
            if workflow_dir.endswith('.term'):
                use_terminology = True
        except (ValueError, IndexError):
            lang_pair = report_data.get('lang_pair', 'unknown')
        
        # Apply filters
        if dataset_filter and dataset != dataset_filter:
            continue
        if workflow_filter and workflow != workflow_filter:
            continue
        # Filter by terminology: if with_terminology is False, skip .term experiments
        # if with_terminology is True, skip non-.term experiments
        if not with_terminology and use_terminology:
            continue
        if with_terminology and not use_terminology:
            continue
        
        is_completed, total, successful = check_completion_status(report_data)
        
        # Get avg_chrf_score from summary if available
        summary = report_data.get('summary', {})
        avg_chrf = summary.get('avg_chrf_score')
        
        by_dataset[dataset][workflow][model][lang_pair].append({
            'path': report_path,
            'completed': is_completed,
            'total': total,
            'successful': successful,
            'failed': report_data.get('failed_samples', 0),
            'use_terminology': use_terminology,
            'avg_chrf_score': avg_chrf
        })
    
    # Print report
    print("=" * 80)
    if with_terminology:
        print("Experiment Completion Status (Terminology Experiments Only)")
    else:
        print("Experiment Completion Status (Normal Experiments Only)")
    print("=" * 80)
    print()
    
    if not by_dataset:
        print("No experiments found in outputs/ directory.")
        return
    
    for dataset in sorted(by_dataset.keys()):
        print(f"📊 Dataset: {dataset}")
        print("-" * 80)
        
        dataset_completed = 0
        dataset_total = 0
        
        for workflow in sorted(by_dataset[dataset].keys()):
            workflow_completed = 0
            workflow_total = 0
            
            for model in sorted(by_dataset[dataset][workflow].keys()):
                model_completed = 0
                model_total = 0
                lang_pairs_status = []
                
                for lang_pair in sorted(by_dataset[dataset][workflow][model].keys()):
                    for status in by_dataset[dataset][workflow][model][lang_pair]:
                        model_total += 1
                        term_suffix = " (term)" if status['use_terminology'] else ""
                        if status['completed']:
                            model_completed += 1
                            # Add chrF++ score if available
                            chrf_info = ""
                            if status.get('avg_chrf_score') is not None:
                                chrf_info = f" [chrF++: {status['avg_chrf_score']:.2f}]"
                            lang_pairs_status.append(
                                f"  ✓ {lang_pair}{term_suffix}: {status['successful']}/{status['total']} samples{chrf_info}"
                            )
                        else:
                            lang_pairs_status.append(
                                f"  ✗ {lang_pair}{term_suffix}: {status['successful']}/{status['total']} samples "
                                f"({status['failed']} failed)"
                            )
                
                workflow_completed += model_completed
                workflow_total += model_total
                
                # Print model status with avg chrF++ if all completed
                status_icon = "✓" if model_completed == model_total else "✗"
                chrf_info = ""
                if model_completed == model_total:
                    # Get average chrF++ across all lang_pairs for this model
                    chrf_scores = []
                    for lang_pair in sorted(by_dataset[dataset][workflow][model].keys()):
                        for status in by_dataset[dataset][workflow][model][lang_pair]:
                            if status['completed'] and status.get('avg_chrf_score') is not None:
                                chrf_scores.append(status['avg_chrf_score'])
                    if chrf_scores:
                        avg_chrf = sum(chrf_scores) / len(chrf_scores)
                        chrf_info = f" [avg chrF++: {avg_chrf:.2f}]"
                print(f"  {status_icon} {workflow} | {model}: {model_completed}/{model_total} completed{chrf_info}")
                
                # Show lang_pair details if not all completed
                if model_completed < model_total:
                    for status_line in lang_pairs_status:
                        print(status_line)
            
            dataset_completed += workflow_completed
            dataset_total += workflow_total
        
        print(f"\n  Summary: {dataset_completed}/{dataset_total} experiments completed")
        print()
    
    # Print overall summary
    total_completed = sum(
        sum(
            sum(
                sum(
                    sum(1 for status in by_dataset[d][w][m][lp] if status['completed'])
                    for lp in by_dataset[d][w][m].keys()
                )
                for m in by_dataset[d][w].keys()
            )
            for w in by_dataset[d].keys()
        )
        for d in by_dataset.keys()
    )
    total_experiments = sum(
        sum(
            sum(
                sum(
                    len(by_dataset[d][w][m][lp])
                    for lp in by_dataset[d][w][m].keys()
                )
                for m in by_dataset[d][w].keys()
            )
            for w in by_dataset[d].keys()
        )
        for d in by_dataset.keys()
    )
    
    print("=" * 80)
    print(f"Overall: {total_completed}/{total_experiments} experiments completed")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check which experiments have completed successfully"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset (e.g., 'wmt25', 'dolfin')"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Filter by workflow (e.g., 'IRB_refine', 'MaMT_translate_postedit_proofread')"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Custom outputs directory path (default: outputs/ relative to project root). "
             "Example: --outputs-dir zhijin/agent-mt-main/outputs"
    )
    parser.add_argument(
        "--with-terminology",
        action="store_true",
        help="Show only terminology experiments (*.term). By default, shows only normal experiments."
    )
    
    args = parser.parse_args()
    
    # Determine outputs directory
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir).resolve()
    else:
        outputs_dir = OUTPUTS_DIR
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory does not exist: {outputs_dir}")
        return 1
    
    # Find all report files
    print(f"Scanning {outputs_dir}...")
    reports = find_report_files(outputs_dir, dataset=args.dataset)
    print(f"Found {len(reports)} report file(s)\n")
    
    if not reports:
        print("No report files found.")
        return 0
    
    # Generate and print report
    generate_report(
        reports, 
        dataset_filter=args.dataset, 
        workflow_filter=args.workflow,
        with_terminology=args.with_terminology
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


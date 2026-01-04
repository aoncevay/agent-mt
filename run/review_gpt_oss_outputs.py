#!/usr/bin/env python3
"""
Review and recompute metrics for gpt-oss-* model outputs by removing reasoning blocks.

This script:
1. Finds all report.json files for gpt-oss-* models
2. Loads the last agent output file for each sample
3. Removes <reasoning>...</reasoning> blocks from the output
4. Recomputes chrF++ and term accuracy scores
5. Updates report.json with reviewed_* scores in the summary

Usage:
    python run/review_gpt_oss_outputs.py --outputs_dir outputs zhijin/agent-mt-main/outputs
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from evaluation import compute_chrf, compute_term_success_rate


REASONING_PATTERN = re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL)


def remove_reasoning_blocks(text: str) -> str:
    """
    Remove all <reasoning>...</reasoning> blocks from text.
    
    Args:
        text: Input text that may contain reasoning blocks
    
    Returns:
        Text with reasoning blocks removed
    """
    # Remove all reasoning blocks (non-greedy, dot matches newline)
    cleaned = REASONING_PATTERN.sub('', text)
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned


def get_last_agent_output_file(output_dir: Path, sample_id: str, sample_idx: int) -> Optional[Path]:
    """
    Get the last agent output file for a sample.
    
    Args:
        output_dir: Directory containing sample files
        sample_id: Sample ID from report
        sample_idx: Sample index from report
    
    Returns:
        Path to the last agent output file, or None if not found
    """
    # Determine filename prefix (same logic as in run.py)
    if sample_id and sample_id != str(sample_idx):
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")[:50]
        file_prefix = f"sample_{safe_id}"
    else:
        file_prefix = f"sample_{sample_idx:05d}"
    
    # Find all agent files for this sample
    pattern = f"{file_prefix}_agent_*.txt"
    agent_files = list(output_dir.glob(pattern))
    
    if not agent_files:
        return None
    
    # Extract agent IDs and find the highest one
    agent_ids = []
    for file_path in agent_files:
        match = re.match(rf'{re.escape(file_prefix)}_agent_(\d+)\.txt', file_path.name)
        if match:
            agent_ids.append(int(match.group(1)))
    
    if not agent_ids:
        return None
    
    max_agent_id = max(agent_ids)
    last_file = output_dir / f"{file_prefix}_agent_{max_agent_id}.txt"
    
    if last_file.exists():
        return last_file
    
    return None


def recompute_metrics_for_sample(
    output_file: Path,
    source_text: str,
    reference_text: str,
    terminology: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Recompute chrF++ and term accuracy for a sample after removing reasoning blocks.
    
    Args:
        output_file: Path to the output file
        source_text: Source text
        reference_text: Reference text
        target_lang: Target language code
        terminology: Optional terminology dictionary (for WMT25)
    
    Returns:
        Tuple of (chrf_score, term_success_rate)
    """
    # Load output text
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_text = f.read().strip()
    except Exception as e:
        print(f"  Warning: Could not read {output_file}: {e}")
        return None, None
    
    # Remove reasoning blocks
    cleaned_output = remove_reasoning_blocks(output_text)
    
    if not cleaned_output:
        print(f"  Warning: Output is empty after removing reasoning blocks for {output_file.name}")
        return None, None
    
    # Compute chrF++
    chrf_result = compute_chrf(cleaned_output, reference_text)
    chrf_score = chrf_result.get("score") if chrf_result else None
    
    # Compute term accuracy if terminology is available
    term_success_rate = None
    if terminology:
        term_success_rate = compute_term_success_rate(
            source_text, cleaned_output, reference_text, terminology, lowercase=True
        )
        if term_success_rate < 0:
            term_success_rate = None
    
    return chrf_score, term_success_rate


def process_report(report_path: Path) -> bool:
    """
    Process a single report.json file and update it with reviewed scores.
    
    Args:
        report_path: Path to report.json file
    
    Returns:
        True if successfully processed, False otherwise
    """
    output_dir = report_path.parent
    
    # Load report
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"Error loading {report_path}: {e}")
        return False
    
    model = report.get("model", "")
    dataset = report.get("dataset", "")
    workflow = report.get("workflow", "")
    
    # Process models where the final/postedit agent is a gpt-oss-* model
    # This includes:
    # - Single gpt-oss-* models (e.g., "gpt-oss-120b")
    # - Combinations where postedit model is gpt-oss-* (e.g., "gpt-4-1+gpt-oss-120b")
    # We don't need to process combinations where base is gpt-oss-* but postedit is not
    # (e.g., "gpt-oss-120b+gpt-4-1") because the final agent won't have reasoning blocks
    should_process = False
    if model.startswith("gpt-oss-"):
        # Single gpt-oss-* model
        should_process = True
    elif "+" in model:
        # Combination: check if postedit model (part after "+") is gpt-oss-*
        parts = model.split("+", 1)
        if len(parts) == 2:
            postedit_model = parts[1]
            if postedit_model.startswith("gpt-oss-"):
                should_process = True
    
    if not should_process:
        return False
    
    print(f"Processing: {dataset} / {workflow} / {model}")
    
    samples = report.get("samples", [])
    if not samples:
        print("  No samples in report")
        return False
    
    # Collect reviewed scores
    reviewed_chrf_scores = []
    reviewed_term_success_rates = []
    processed_count = 0
    skipped_count = 0
    
    for sample in samples:
        if sample.get("error"):
            # Skip samples with errors
            continue
        
        sample_idx = sample.get("sample_idx")
        sample_id = sample.get("sample_id", str(sample_idx))
        source_text = sample.get("source_text")
        reference_text = sample.get("reference_text")
        target_lang = sample.get("target_lang")
        
        if not source_text or not reference_text or target_lang is None:
            skipped_count += 1
            continue
        
        # Get last agent output file
        output_file = get_last_agent_output_file(output_dir, sample_id, sample_idx)
        if not output_file:
            skipped_count += 1
            continue
        
        # Terminology is not stored in report.json, so we skip term accuracy computation
        # (To compute term accuracy, we would need to load terminology from original data files)
        terminology = None
        
        # Recompute metrics
        chrf_score, term_success_rate = recompute_metrics_for_sample(
            output_file, source_text, reference_text, terminology
        )
        
        if chrf_score is not None:
            reviewed_chrf_scores.append(chrf_score)
            processed_count += 1
        
        if term_success_rate is not None:
            reviewed_term_success_rates.append(term_success_rate)
    
    # Update report summary with reviewed scores
    if "summary" not in report:
        report["summary"] = {}
    
    summary = report["summary"]
    
    if reviewed_chrf_scores:
        summary["reviewed_avg_chrf_score"] = sum(reviewed_chrf_scores) / len(reviewed_chrf_scores)
        summary["reviewed_min_chrf_score"] = min(reviewed_chrf_scores)
        summary["reviewed_max_chrf_score"] = max(reviewed_chrf_scores)
    
    if reviewed_term_success_rates:
        summary["reviewed_avg_term_success_rate"] = sum(reviewed_term_success_rates) / len(reviewed_term_success_rates)
        summary["reviewed_min_term_success_rate"] = min(reviewed_term_success_rates)
        summary["reviewed_max_term_success_rate"] = max(reviewed_term_success_rates)
    
    # Save updated report
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Processed {processed_count} samples, skipped {skipped_count}")
        if reviewed_chrf_scores:
            print(f"    Reviewed avg chrF++: {summary['reviewed_avg_chrf_score']:.2f}")
        if reviewed_term_success_rates:
            print(f"    Reviewed avg term acc: {summary['reviewed_avg_term_success_rate']:.4f}")
        return True
    except Exception as e:
        print(f"  Error saving report: {e}")
        return False


def find_reports(outputs_dirs: List[Path]) -> List[Path]:
    """Find all report.json files for gpt-oss-* models."""
    reports = []
    
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            continue
        
        # Find all report.json files
        for report_path in outputs_dir.rglob("report.json"):
            # Quick check: read model name without loading full JSON
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    # Read first few lines to get model name
                    for line in f:
                        if '"model"' in line:
                            # Extract model name
                            match = re.search(r'"model"\s*:\s*"([^"]+)"', line)
                            if match:
                                model = match.group(1)
                                if model.startswith("gpt-oss-"):
                                    reports.append(report_path)
                                    break
                            break
            except Exception:
                continue
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Review and recompute metrics for gpt-oss-* model outputs"
    )
    parser.add_argument(
        "--outputs_dirs",
        type=str,
        nargs='+',
        default=["outputs", "zhijin/agent-mt-main/outputs"],
        help="Output directories to search for reports"
    )
    
    args = parser.parse_args()
    
    outputs_dirs = [Path(d) for d in args.outputs_dirs]
    
    print("Finding report.json files for gpt-oss-* models...")
    reports = find_reports(outputs_dirs)
    
    print(f"Found {len(reports)} reports for gpt-oss-* models")
    print()
    
    processed = 0
    failed = 0
    
    for report_path in reports:
        if process_report(report_path):
            processed += 1
        else:
            failed += 1
        print()
    
    print(f"Processed: {processed}")
    if failed > 0:
        print(f"Failed: {failed}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


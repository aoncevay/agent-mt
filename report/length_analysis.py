#!/usr/bin/env python3
"""
Plot length-based analysis comparing chrF++ scores for shorter vs longer texts.

Splits datasets into two halves based on English source text token length,
then compares chrF++ performance across all workflows for each half.

Usage:
    python report/length_analysis.py --outputs_dirs outputs_qwen3 outputs --models gpt-4-1 gpt-4-1-nano
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import from plot_main_results
import importlib.util
plot_script_path = Path(__file__).parent / "plot_main_results.py"
spec = importlib.util.spec_from_file_location("plot_main_results", plot_script_path)
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)

WORKFLOW_COLORS = plot_module.WORKFLOW_COLORS
WORKFLOW_ACRONYMS = plot_module.WORKFLOW_ACRONYMS
get_workflow_acronym = plot_module.get_workflow_acronym
MODEL_DISPLAY_NAMES = plot_module.MODEL_DISPLAY_NAMES

# Workflow order for consistent plotting
WORKFLOW_ORDER = ["ZS", "IRB", "MaMT", "SbS_chat", "MAATS_multi", "DeLTA"]

# Workflow display names
WORKFLOW_DISPLAY_NAMES = {
    "ZS": "Zero-shot",
    "IRB": "IRB",
    "MaMT": "MaMT",
    "SbS_chat": "Step-by-step",
    "MAATS_multi": "MAATS",
    "DeLTA": "DelTA",
}

# Marker styles for shorter vs longer halves
SHORTER_MARKER = "x"  # Cross
LONGER_MARKER = "*"   # Star


def count_tokens(text: str) -> int:
    """
    Simple token counter (approximate).
    For more accurate counting, we could use tiktoken or similar,
    but for splitting purposes, word count should be sufficient.
    """
    # Split by whitespace and count
    return len(text.split())


def load_source_text_from_data(
    dataset: str,
    lang_pair: str,
    sample_id: str,
    data_dir: Path
) -> Optional[str]:
    """Load source text from original data files using sample_id."""
    try:
        if dataset == "dolfin":
            source_lang, target_lang = lang_pair.split("_")
            file_path = data_dir / "raw" / "dolfin" / f"dolfin_test_{lang_pair}.jsonl"
        elif dataset == "wmt25":
            # For WMT25, determine direction from lang_pair
            if lang_pair.startswith("en-"):
                source_lang = "en"
                target_lang = lang_pair.split("-")[1]
            else:
                source_lang = lang_pair.split("-")[0]
                target_lang = "en"
            # WMT25 files are organized differently - need to find the right file
            # For now, try common patterns
            file_path = data_dir / "raw" / "wmt25-terminology-track2" / f"full_data_2024.jsonl"
        else:
            return None
        
        if not file_path.exists():
            return None
        
        # Search for sample in data file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                sample = json.loads(line)
                # Match by id or _id
                if sample.get("id") == sample_id or sample.get("_id") == sample_id:
                    return sample.get(source_lang, "")
        
        return None
    except Exception as e:
        return None


def parse_report_with_samples(report_path: Path, data_dir: Optional[Path] = None) -> Optional[Dict]:
    """Parse a report.json file and return data with individual samples."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if experiment is complete
        total_samples = data.get("total_samples", 0)
        successful_samples = data.get("successful_samples", 0)
        
        if total_samples == 0 or total_samples != successful_samples:
            return None  # Incomplete experiment
        
        # Extract workflow, model, dataset, lang_pair
        workflow = data.get("workflow", "")
        model = data.get("model", "")
        dataset = data.get("dataset", "")
        lang_pair = data.get("lang_pair", "")
        
        # Get samples
        samples = data.get("samples", [])
        if not samples:
            return None
        
        # Extract sample-level data
        sample_data = []
        for sample in samples:
            if sample.get("error"):
                continue  # Skip failed samples
            
            # Try to get source text from sample, or load from data files
            source_text = sample.get("source_text", "")
            if not source_text and data_dir:
                sample_id = sample.get("sample_id", sample.get("sample_idx", ""))
                source_text = load_source_text_from_data(dataset, lang_pair, str(sample_id), data_dir)
            
            if not source_text:
                # Fall back to using tokens_input as proxy (not ideal but better than nothing)
                # tokens_input correlates with source text length
                tokens_input = sample.get("tokens_input", 0)
                if tokens_input == 0:
                    continue
                # Use tokens_input as proxy for length (will be approximate)
                token_count = tokens_input
            else:
                # Count tokens in source text (English side)
                token_count = count_tokens(source_text)
            
            # Get chrF score (use last evaluation if multiple)
            chrf_scores = sample.get("chrf_scores", [])
            if not chrf_scores:
                continue
            
            chrf_score = chrf_scores[-1] if isinstance(chrf_scores, list) else chrf_scores
            
            # Get TermAcc score (use last evaluation if multiple, only for WMT25)
            term_success_rates = sample.get("term_success_rates", [])
            term_acc = None
            if term_success_rates:
                term_acc = term_success_rates[-1] if isinstance(term_success_rates, list) else term_success_rates
                # Filter out invalid values (-1.0 indicates no terminology data)
                if term_acc is not None and term_acc < 0:
                    term_acc = None
            
            sample_data.append({
                "chrf_score": chrf_score,
                "term_acc": term_acc,
                "token_count": token_count,
                "sample_id": sample.get("sample_id", sample.get("sample_idx", ""))
            })
        
        if not sample_data:
            return None
        
        return {
            "workflow": workflow,
            "model": model,
            "dataset": dataset,
            "lang_pair": lang_pair,
            "samples": sample_data
        }
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not parse {report_path}: {e}")
        return None


def collect_reports_by_model(
    outputs_dirs: List[Path],
    target_models: List[str],
    datasets: List[str] = ["dolfin", "wmt25"],
    data_dir: Optional[Path] = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Collect reports grouped by model and dataset.
    
    Returns:
        Dictionary: model -> dataset -> list of report data
    """
    reports_by_model = defaultdict(lambda: defaultdict(list))
    
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            continue
        
        # Iterate through dataset directories
        for dataset_dir in outputs_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name
            if dataset not in datasets:
                continue
            
            # Iterate through language pair directories
            for lang_pair_dir in dataset_dir.iterdir():
                if not lang_pair_dir.is_dir():
                    continue
                
                lang_pair = lang_pair_dir.name
                
                # Iterate through workflow directories
                for workflow_dir in lang_pair_dir.iterdir():
                    if not workflow_dir.is_dir():
                        continue
                    
                    workflow_dir_name = workflow_dir.name
                    # For WMT25, use .term workflows; for DOLFIN, use regular workflows
                    if workflow_dir_name.endswith(".term"):
                        if dataset == "wmt25":
                            # For WMT25, we want .term workflows
                            workflow_base = workflow_dir_name.replace(".term", "")
                        else:
                            continue  # Skip .term for non-WMT25
                    else:
                        # Regular workflow (no .term suffix)
                        if dataset == "wmt25":
                            continue  # For WMT25, only use .term workflows
                        workflow_base = workflow_dir_name
                    
                    # Get workflow acronym
                    workflow_acronym = get_workflow_acronym(workflow_base)
                    
                    # Iterate through model directories
                    for model_dir in workflow_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        
                        model = model_dir.name
                        
                        # Only include target models
                        if model not in target_models:
                            continue
                        
                        # Skip combinations (only single models for this analysis)
                        if "+" in model:
                            continue
                        
                        report_path = model_dir / "report.json"
                        if not report_path.exists():
                            continue
                        
                        report_data = parse_report_with_samples(report_path, data_dir)
                        if report_data is None:
                            continue
                        
                        # Add workflow acronym to report data
                        report_data["workflow_acronym"] = workflow_acronym
                        
                        reports_by_model[model][dataset].append(report_data)
    
    return reports_by_model


def split_samples_by_length(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into two halves based on token length.
    
    Returns:
        (shorter_half, longer_half)
    """
    if not samples:
        return [], []
    
    # Sort by token count
    sorted_samples = sorted(samples, key=lambda x: x["token_count"])
    
    # Split in half
    mid = len(sorted_samples) // 2
    shorter_half = sorted_samples[:mid]
    longer_half = sorted_samples[mid:]
    
    return shorter_half, longer_half


def calculate_avg_chrf(samples: List[Dict]) -> Optional[float]:
    """Calculate average chrF++ score from samples."""
    if not samples:
        return None
    
    scores = [s["chrf_score"] for s in samples if s.get("chrf_score") is not None]
    if not scores:
        return None
    
    return sum(scores) / len(scores)


def calculate_avg_termacc(samples: List[Dict]) -> Optional[float]:
    """Calculate average TermAcc score from samples."""
    if not samples:
        return None
    
    scores = [s["term_acc"] for s in samples if s.get("term_acc") is not None and s.get("term_acc") >= 0]
    if not scores:
        return None
    
    return sum(scores) / len(scores)


def aggregate_by_workflow(
    reports: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate data by workflow, calculating chrF++ and TermAcc for shorter and longer halves.
    
    Returns:
        Dictionary: workflow_acronym -> {
            "shorter_chrf": avg_chrf, "longer_chrf": avg_chrf,
            "shorter_termacc": avg_termacc, "longer_termacc": avg_termacc
        }
    """
    workflow_data = defaultdict(lambda: {"samples": []})
    
    # Collect all samples for each workflow (across language pairs)
    for report in reports:
        workflow = report.get("workflow_acronym", "")
        if not workflow:
            continue
        
        samples = report.get("samples", [])
        workflow_data[workflow]["samples"].extend(samples)
    
    # Calculate averages for each workflow
    result = {}
    for workflow, data in workflow_data.items():
        all_samples = data["samples"]
        if not all_samples:
            continue
        
        shorter_half, longer_half = split_samples_by_length(all_samples)
        
        shorter_chrf = calculate_avg_chrf(shorter_half)
        longer_chrf = calculate_avg_chrf(longer_half)
        shorter_termacc = calculate_avg_termacc(shorter_half)
        longer_termacc = calculate_avg_termacc(longer_half)
        
        if shorter_chrf is not None and longer_chrf is not None:
            result[workflow] = {
                "shorter_chrf": shorter_chrf,
                "longer_chrf": longer_chrf,
                "shorter_termacc": shorter_termacc,
                "longer_termacc": longer_termacc,
                "shorter_count": len(shorter_half),
                "longer_count": len(longer_half)
            }
    
    return result


def plot_length_analysis(
    workflow_data: Dict[str, Dict[str, float]],
    dataset_name: str,
    model_name: str,
    output_path: Path
):
    """Create plot comparing shorter vs longer text performance with two subplots (chrF++ and TermAcc)."""
    
    # Filter workflows to only those in WORKFLOW_ORDER
    filtered_workflows = []
    for workflow in WORKFLOW_ORDER:
        if workflow in workflow_data:
            filtered_workflows.append(workflow)
    
    if not filtered_workflows:
        print(f"Warning: No workflow data for {model_name} on {dataset_name}")
        return
    
    # Create figure with two subplots (square shape, narrow width for two per column)
    # Width: 2.5 per subplot = 5 total, Height: 3 (reduced by 1 from 4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
    
    # Prepare data
    x_positions = range(len(filtered_workflows))
    workflow_labels = [WORKFLOW_DISPLAY_NAMES.get(w, w) for w in filtered_workflows]
    
    # Plot chrF++ subplot (left)
    shorter_chrf_scores = [workflow_data[w].get("shorter_chrf") for w in filtered_workflows]
    longer_chrf_scores = [workflow_data[w].get("longer_chrf") for w in filtered_workflows]
    
    plot_subplot(ax1, filtered_workflows, workflow_data, x_positions, workflow_labels,
                 "shorter_chrf", "longer_chrf", "chrF++", "chrF++")
    
    # Plot TermAcc subplot (right) - only if we have TermAcc data
    has_termacc = any(workflow_data[w].get("shorter_termacc") is not None or 
                     workflow_data[w].get("longer_termacc") is not None 
                     for w in filtered_workflows)
    
    if has_termacc:
        plot_subplot(ax2, filtered_workflows, workflow_data, x_positions, workflow_labels,
                     "shorter_termacc", "longer_termacc", "TermAcc", "Terminology Accuracy")
    else:
        # Hide second subplot if no TermAcc data
        ax2.axis('off')
    
    # Legend (shared between subplots, close to plots)
    shorter_handle = plt.Line2D([0], [0], marker=SHORTER_MARKER, color='black', 
                                linestyle='None', markersize=8, label='Shorter half')
    longer_handle = plt.Line2D([0], [0], marker=LONGER_MARKER, color='black',
                               linestyle='None', markersize=10, label='Longer half')
    fig.legend(handles=[shorter_handle, longer_handle], loc='upper center', 
              ncol=2, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave minimal space for legend
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created length analysis plot: {output_path}")


def plot_subplot(ax, workflows: List[str], workflow_data: Dict, 
                 x_positions: List[int], workflow_labels: List[str],
                 shorter_key: str, longer_key: str, ylabel: str, title: str):
    """Plot a single subplot for a metric (chrF++ or TermAcc)."""
    
    # Plot lines connecting shorter to longer for each workflow
    all_scores = []
    for i, workflow in enumerate(workflows):
        shorter = workflow_data[workflow].get(shorter_key)
        longer = workflow_data[workflow].get(longer_key)
        
        if shorter is None or longer is None:
            continue
        
        all_scores.extend([shorter, longer])
        
        # Draw vertical line (all black, no colors)
        ax.plot([i, i], [shorter, longer], color='black', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Plot markers (all black)
        # For 'x' marker, don't use edgecolors (it's unfilled)
        if SHORTER_MARKER == 'x':
            ax.scatter(i, shorter, marker=SHORTER_MARKER, s=120, c='black',
                      linewidths=1.5, alpha=0.8, zorder=3)
        else:
            ax.scatter(i, shorter, marker=SHORTER_MARKER, s=120, c='black',
                      edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)
        
        ax.scatter(i, longer, marker=LONGER_MARKER, s=150, c='black',
                  edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(workflow_labels, rotation=45, ha='right', fontsize=9)
    
    # Set y-axis
    ax.set_ylabel(ylabel, fontsize=10)
    # Title removed (redundant with y-axis label)
    
    # Auto-scale y-axis with appropriate steps
    if all_scores:
        y_min = min(all_scores)
        y_max = max(all_scores)
        
        # For TermAcc, use 0.05 steps; for chrF++, use 5-point steps
        if "termacc" in shorter_key.lower() or "term" in ylabel.lower():
            y_min_rounded = 0.05 * (int(y_min / 0.05))
            y_max_rounded = 0.05 * ((int(y_max / 0.05) + 1))
            step = max(0.05, (y_max_rounded - y_min_rounded) / 10)
            step = 0.05 * ((int(step / 0.05) + 1))
            y_ticks = np.arange(y_min_rounded, y_max_rounded + step, step)
            ax.set_yticks(y_ticks)
            ax.set_ylim(y_min_rounded, y_max_rounded)
        else:
            # chrF++: 5-point steps
            y_min_rounded = 5 * (int(y_min) // 5)
            y_max_rounded = 5 * ((int(y_max) + 4) // 5)
            step = max(5, (y_max_rounded - y_min_rounded) // 10)
            step = 5 * ((step + 4) // 5)
            y_ticks = list(range(y_min_rounded, y_max_rounded + step, step))
            ax.set_yticks(y_ticks)
            ax.set_ylim(y_min_rounded, y_max_rounded)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)


def main():
    parser = argparse.ArgumentParser(
        description="Plot length-based analysis comparing shorter vs longer text performance"
    )
    parser.add_argument(
        "--outputs_dirs",
        type=str,
        nargs='+',
        default=["outputs_qwen3", "outputs"],
        help="Paths to outputs directories containing report.json files"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=["gpt-4-1", "gpt-4-1-nano"],
        help="Model names to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="report/figs",
        help="Directory to save output PDF plots"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (for loading source text if not in reports)"
    )
    
    args = parser.parse_args()
    
    outputs_dirs = [Path(d) for d in args.outputs_dirs]
    output_dir = Path(args.output_dir)
    target_models = args.models
    data_dir = Path(args.data_dir) if args.data_dir else None
    
    # If data_dir not provided, try to find it relative to project root
    if data_dir is None:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        if not data_dir.exists():
            data_dir = None
            print("Warning: Data directory not found. Will use tokens_input as proxy for length.")
    
    # Check if at least one outputs directory exists
    existing_dirs = [d for d in outputs_dirs if d.exists()]
    if not existing_dirs:
        print("Error: None of the specified outputs directories exist:")
        for d in outputs_dirs:
            print(f"  - {d}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect reports
    print("Collecting reports...")
    reports_by_model = collect_reports_by_model(existing_dirs, target_models, data_dir=data_dir)
    
    # Generate plots for each model and dataset
    for model in target_models:
        if model not in reports_by_model:
            print(f"Warning: No reports found for model {model}")
            continue
        
        for dataset, reports in reports_by_model[model].items():
            if not reports:
                continue
            
            print(f"\nProcessing {model} on {dataset}...")
            
            # Aggregate by workflow
            workflow_data = aggregate_by_workflow(reports)
            
            if not workflow_data:
                print(f"  No workflow data available")
                continue
            
            # Create plot
            dataset_display = dataset.upper().replace("WMT25", "WMT25+T")
            output_filename = f"length_{dataset_display}_{model}.pdf"
            output_path = output_dir / output_filename
            
            plot_length_analysis(workflow_data, dataset, model, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())


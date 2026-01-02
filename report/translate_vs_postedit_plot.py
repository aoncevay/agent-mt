#!/usr/bin/env python3
"""
Plot translation vs post-editing model comparison.

Creates a figure with two subplots (DOLFIN and WMT25-Term) showing chrF++ vs Cost
for experiments using different base models for translation and post-editing.

Focuses on 3 workflows: IRB_refine, MaMT_translate_postedit_proofread, MAATS_multi_agents
with model combinations: gpt-4-1, gpt-oss-120b, and their combinations.

Usage:
    python report/translate_vs_postedit_plot.py --outputs_dirs zhijin/agent-mt-main/outputs outputs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
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
parse_report = plot_module.parse_report
calculate_cost = plot_module.calculate_cost
MODEL_DISPLAY_NAMES = plot_module.MODEL_DISPLAY_NAMES

# Workflows to analyze
TARGET_WORKFLOWS = {
    "IRB_refine": "IRB",
    "MaMT_translate_postedit_proofread": "MaMT",
    "MAATS_multi_agents": "MAATS_multi"
}

# Base models
BASE_MODELS = ["gpt-4-1", "gpt-oss-120b"]

# Model combinations mapping
# Format: model_name -> (base_model, postedit_model)
def parse_model_combination(model_name: str) -> Optional[Tuple[str, str]]:
    """Parse model name in format {base-model}+{model} into (base_model, postedit_model)."""
    if "+" not in model_name:
        # Single model (base model only, no post-editing)
        return (model_name, None)
    
    parts = model_name.split("+")
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None

# Marker styles for different model configurations
# Base models use existing markers, combinations use new ones
MODEL_MARKERS = {
    "gpt-4-1": "P",              # Plus (filled) - existing
    "gpt-oss-120b": "v",         # Triangle down - existing
    "gpt-oss-120b+gpt-4-1": "p", # Pentagon - new
    "gpt-4-1+gpt-oss-120b": "X", # Filled cross X - new
}

# Model display names for combinations
def get_model_display_name(model_name: str) -> str:
    """Get display name for model or combination."""
    parsed = parse_model_combination(model_name)
    if parsed is None:
        return model_name
    
    base_model, postedit_model = parsed
    
    if postedit_model is None:
        # Single model
        return MODEL_DISPLAY_NAMES.get(base_model, base_model)
    else:
        # Combination: {base_model} -> {postedit_model}
        base_display = MODEL_DISPLAY_NAMES.get(base_model, base_model)
        postedit_display = MODEL_DISPLAY_NAMES.get(postedit_model, postedit_model)
        return f"{base_display} → {postedit_display}"


def collect_translate_postedit_reports(outputs_dirs: List[Path]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Collect reports for translate vs postedit analysis.
    
    Returns:
        Dictionary mapping dataset -> lang_pair -> list of reports
    """
    reports_by_dataset = defaultdict(lambda: defaultdict(list))
    
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            continue
        
        # Iterate through dataset directories
        for dataset_dir in outputs_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name
            if dataset not in ["dolfin", "wmt25"]:
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
                    
                    workflow_name = workflow_dir.name
                    
                    # Check if this is one of our target workflows
                    if workflow_name not in TARGET_WORKFLOWS:
                        continue
                    
                    # Iterate through model directories
                    for model_dir in workflow_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        
                        model = model_dir.name
                        
                        # Parse model combination
                        parsed = parse_model_combination(model)
                        if parsed is None:
                            continue
                        
                        base_model, postedit_model = parsed
                        
                        # Only include if base model is one of our target base models
                        if base_model not in BASE_MODELS:
                            continue
                        
                        # Check if it's a combination or single model
                        # We want: gpt-4-1, gpt-oss-120b, gpt-oss-120b+gpt-4-1, gpt-4-1+gpt-oss-120b
                        if postedit_model is not None and postedit_model not in BASE_MODELS:
                            continue
                        
                        report_path = model_dir / "report.json"
                        if not report_path.exists():
                            continue
                        
                        report_data = parse_report(report_path)
                        if report_data is None:
                            continue
                        
                        # Add model name to report data
                        report_data["model_name"] = model
                        report_data["base_model"] = base_model
                        report_data["postedit_model"] = postedit_model
                        
                        # Extract base_model_tokens from summary if available
                        # (parse_report already extracts tokens_input/output, but we need base_model_tokens)
                        try:
                            with open(report_path, 'r', encoding='utf-8') as f:
                                full_data = json.load(f)
                            summary = full_data.get("summary", {})
                            report_data["base_model_tokens_input"] = summary.get("total_base_model_tokens_input", 0)
                            report_data["base_model_tokens_output"] = summary.get("total_base_model_tokens_output", 0)
                        except (json.JSONDecodeError, KeyError, FileNotFoundError):
                            report_data["base_model_tokens_input"] = 0
                            report_data["base_model_tokens_output"] = 0
                        
                        reports_by_dataset[dataset][lang_pair].append(report_data)
    
    return reports_by_dataset


def calculate_combination_cost(
    tokens_input: int,
    tokens_output: int,
    base_model_tokens_input: int,
    base_model_tokens_output: int,
    base_model: str,
    postedit_model: Optional[str],
    use_batch: bool = False
) -> Optional[float]:
    """
    Calculate total cost for a model combination.
    
    For combinations (base_model+postedit_model):
    - base_model_tokens_* are used for base model cost
    - tokens_* are used for post-edit model cost
    
    For single models:
    - tokens_* are used for the model cost
    - base_model_tokens_* should be 0
    """
    total_cost = 0.0
    
    # Calculate base model cost (if base_model_tokens exist)
    if base_model_tokens_input > 0 or base_model_tokens_output > 0:
        base_cost = calculate_cost(
            base_model_tokens_input,
            base_model_tokens_output,
            base_model,
            use_batch=use_batch
        )
        if base_cost is None:
            return None
        total_cost += base_cost
    
    # Calculate post-edit model cost (if postedit_model exists)
    if postedit_model:
        # Combination case: tokens_* are for the post-edit model
        postedit_cost = calculate_cost(
            tokens_input,
            tokens_output,
            postedit_model,
            use_batch=use_batch
        )
        if postedit_cost is None:
            return None
        total_cost += postedit_cost
    elif base_model_tokens_input == 0 and base_model_tokens_output == 0:
        # Single model case: use tokens_* for the base model
        single_cost = calculate_cost(
            tokens_input,
            tokens_output,
            base_model,
            use_batch=use_batch
        )
        if single_cost is None:
            return None
        total_cost += single_cost
    else:
        # This shouldn't happen: we have base_model_tokens but no postedit_model
        # This means it's a single model but with base_model_tokens (shouldn't occur)
        # Fall back to using tokens_* for base_model
        single_cost = calculate_cost(
            tokens_input,
            tokens_output,
            base_model,
            use_batch=use_batch
        )
        if single_cost is None:
            return None
        total_cost += single_cost
    
    return total_cost


def aggregate_data_by_workflow_model(
    reports: List[Dict]
) -> Dict[Tuple[str, str], Dict]:
    """
    Aggregate data across language pairs for each workflow+model combination.
    
    Returns:
        Dictionary mapping (workflow, model_name) -> {
            "chrf": average chrF++,
            "cost": total cost,
            "base_model": base model name,
            "postedit_model": postedit model name (or None)
        }
    """
    aggregated = defaultdict(lambda: {"chrf_values": [], "costs": [], "base_model": None, "postedit_model": None})
    
    for report in reports:
        workflow_name = report.get("workflow", "")
        workflow = get_workflow_acronym(workflow_name)
        
        # Only include target workflows
        if workflow not in TARGET_WORKFLOWS.values():
            continue
        
        model_name = report.get("model_name", "")
        base_model = report.get("base_model")
        postedit_model = report.get("postedit_model")
        
        chrf = report.get("chrf")
        if chrf is None:
            continue
        
        # Get token counts
        tokens_input = report.get("tokens_input", 0)
        tokens_output = report.get("tokens_output", 0)
        base_model_tokens_input = report.get("base_model_tokens_input", 0)
        base_model_tokens_output = report.get("base_model_tokens_output", 0)
        
        # Calculate cost for combination
        cost = calculate_combination_cost(
            tokens_input,
            tokens_output,
            base_model_tokens_input,
            base_model_tokens_output,
            base_model,
            postedit_model,
            use_batch=False
        )
        
        if cost is None:
            continue
        
        key = (workflow, model_name)
        aggregated[key]["chrf_values"].append(chrf)
        aggregated[key]["costs"].append(cost)
        if aggregated[key]["base_model"] is None:
            aggregated[key]["base_model"] = base_model
        if aggregated[key]["postedit_model"] is None:
            aggregated[key]["postedit_model"] = postedit_model
    
    # Compute averages and totals
    result = {}
    for (workflow, model_name), data in aggregated.items():
        if not data["chrf_values"] or not data["costs"]:
            continue
        
        result[(workflow, model_name)] = {
            "chrf": sum(data["chrf_values"]) / len(data["chrf_values"]),
            "cost": sum(data["costs"]),
            "base_model": data["base_model"],
            "postedit_model": data["postedit_model"]
        }
    
    return result


def plot_translate_vs_postedit(
    dolfin_data: Dict[Tuple[str, str], Dict],
    wmt25_data: Dict[Tuple[str, str], Dict],
    output_path: Path
):
    """Create figure with two subplots comparing translation vs post-editing models."""
    
    # Create figure with two subplots side by side (square shape)
    # Each subplot should be square, so if width is 3.5, height should also be 3.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))  # Two square subplots side by side
    
    # Plot DOLFIN
    plot_dataset_subplot(ax1, dolfin_data, "DOLFIN")
    
    # Plot WMT25-Term
    plot_dataset_subplot(ax2, wmt25_data, "WMT25-Term")
    
    # Create legend above subplots
    create_legend_above(fig, dolfin_data, wmt25_data)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at top for legend
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created translate vs postedit plot: {output_path}")


def plot_dataset_subplot(ax, data: Dict[Tuple[str, str], Dict], dataset_name: str):
    """Plot data for a single dataset subplot."""
    
    # Organize data by workflow
    workflow_data = defaultdict(lambda: {
        "gpt-4-1": None,  # Single model
        "gpt-oss-120b": None,  # Single model
        "gpt-oss-120b+gpt-4-1": None,  # Combination
        "gpt-4-1+gpt-oss-120b": None,  # Combination
    })
    
    for (workflow, model_name), point_data in data.items():
        model_key = model_name
        if model_key not in workflow_data[workflow]:
            continue
        
        workflow_data[workflow][model_key] = {
            "chrf": point_data["chrf"],
            "cost": point_data["cost"],
            "base_model": point_data["base_model"],
            "postedit_model": point_data["postedit_model"]
        }
    
    # Plot each workflow
    for workflow in sorted(workflow_data.keys()):
        color = WORKFLOW_COLORS.get(workflow, "#000000")
        wf_data = workflow_data[workflow]
        
        # Collect points for this workflow in order: gpt-oss-120b -> combinations -> gpt-4-1
        points_order = [
            ("gpt-oss-120b", wf_data["gpt-oss-120b"]),
            ("gpt-oss-120b+gpt-4-1", wf_data["gpt-oss-120b+gpt-4-1"]),
            ("gpt-4-1+gpt-oss-120b", wf_data["gpt-4-1+gpt-oss-120b"]),
            ("gpt-4-1", wf_data["gpt-4-1"]),
        ]
        
        # Filter out None points
        valid_points = [(name, data) for name, data in points_order if data is not None]
        
        if len(valid_points) < 2:
            continue
        
        # Extract costs and chrfs for line
        costs_line = [p[1]["cost"] for p in valid_points]
        chrfs_line = [p[1]["chrf"] for p in valid_points]
        
        # Draw dotted line connecting points
        ax.plot(costs_line, chrfs_line, color=color, linestyle=':', linewidth=1.5, 
                alpha=0.6, zorder=1)
        
        # Plot markers
        for model_name, point_data in valid_points:
            marker = MODEL_MARKERS.get(model_name, "o")
            ax.scatter(point_data["cost"], point_data["chrf"], 
                      c=color, marker=marker, s=63,
                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel('Cost ($, log scale)', fontsize=10)
    ax.set_ylabel('chrF++', fontsize=10)
    ax.set_title(dataset_name, fontsize=11, fontweight='bold')
    
    # Auto-scale y-axis
    ax.set_ylim(auto=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)


def create_legend_above(fig, dolfin_data: Dict, wmt25_data: Dict):
    """Create legend above the subplots."""
    
    # Collect unique workflows and models from both datasets
    all_workflows = set()
    all_models = set()
    
    for data in [dolfin_data, wmt25_data]:
        for (workflow, model_name) in data.keys():
            all_workflows.add(workflow)
            all_models.add(model_name)
    
    # Create legend elements
    workflow_elements = []
    for workflow in sorted(all_workflows):
        if workflow in WORKFLOW_COLORS:
            color = WORKFLOW_COLORS[workflow]
            workflow_display = {
                "IRB": "IRB",
                "MaMT": "MaMT",
                "MAATS_multi": "MAATS"
            }.get(workflow, workflow)
            workflow_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black', label=workflow_display)
            )
    
    # Model elements - order: base models first, then combinations
    model_elements = []
    model_order = [
        "gpt-4-1",
        "gpt-oss-120b",
        "gpt-oss-120b+gpt-4-1",
        "gpt-4-1+gpt-oss-120b"
    ]
    
    for model_name in model_order:
        if model_name not in all_models:
            continue
        
        marker = MODEL_MARKERS.get(model_name, "o")
        display_name = get_model_display_name(model_name)
        
        model_elements.append(
            plt.Line2D([0], [0], marker=marker, color='black', linestyle='None',
                      markersize=10, label=display_name, markerfacecolor='black',
                      markeredgecolor='black')
        )
    
    # Combine elements
    all_elements = workflow_elements + model_elements
    
    # Create legend at top
    if all_elements:
        fig.legend(handles=all_elements, loc='upper center', ncol=min(len(all_elements), 7),
                  frameon=True, fontsize=9, borderpad=0.2, columnspacing=0.8,
                  handletextpad=0.3, handlelength=1.0, bbox_to_anchor=(0.5, 0.98))


def main():
    parser = argparse.ArgumentParser(
        description="Plot translation vs post-editing model comparison"
    )
    parser.add_argument(
        "--outputs_dirs",
        type=str,
        nargs='+',
        default=["zhijin/agent-mt-main/outputs", "outputs"],
        help="Paths to outputs directories containing report.json files (can specify multiple)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="report/figs",
        help="Directory to save output PDF plot"
    )
    
    args = parser.parse_args()
    
    outputs_dirs = [Path(d) for d in args.outputs_dirs]
    output_dir = Path(args.output_dir)
    
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
    reports_by_dataset = collect_translate_postedit_reports(existing_dirs)
    
    # Aggregate data for each dataset
    print("Aggregating data...")
    
    # DOLFIN: aggregate across all language pairs
    dolfin_reports = []
    for reports in reports_by_dataset.get("dolfin", {}).values():
        dolfin_reports.extend(reports)
    
    dolfin_data = aggregate_data_by_workflow_model(dolfin_reports)
    
    # WMT25-Term: aggregate across all language pairs
    wmt25_reports = []
    for reports in reports_by_dataset.get("wmt25", {}).values():
        # Only include .term workflows
        wmt25_reports.extend(reports)
    
    wmt25_data = aggregate_data_by_workflow_model(wmt25_reports)
    
    # Create plot
    print("Creating plot...")
    output_path = output_dir / "translate_vs_postedit.pdf"
    plot_translate_vs_postedit(dolfin_data, wmt25_data, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())


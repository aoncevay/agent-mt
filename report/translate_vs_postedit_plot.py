#!/usr/bin/env python3
"""
Plot translation vs post-editing model comparison.

Creates a figure with two subplots (DOLFIN and WMT25-Term) showing chrF++ vs Cost
for experiments using different base models for translation and post-editing.

Focuses on 3 workflows: IRB_refine, MaMT_translate_postedit_proofread, MAATS_multi_agents
with model combinations: gpt-4-1, gpt-4-1-nano, and their combinations.

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

# Workflows to analyze (directory names as they appear in outputs/)
# Note: Directory names are the acronyms (IRB, MaMT, MAATS_multi), not the full workflow names
TARGET_WORKFLOWS = {
    "IRB": "IRB",
    "IRB.term": "IRB",
    "MaMT": "MaMT",
    "MaMT.term": "MaMT",
    "MAATS_multi": "MAATS_multi",
    "MAATS_multi.term": "MAATS_multi"
}

# Base models
BASE_MODELS = ["gpt-4-1", "gpt-4-1-nano"]

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
    "gpt-4-1": "P",                      # Plus (filled) - existing
    "gpt-4-1-nano": "D",                 # Diamond - existing
    "gpt-4-1-nano+gpt-4-1": "h",         # Hexagon - new
    "gpt-4-1+gpt-4-1-nano": "X",         # Filled cross X - new
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
    
    print(f"DEBUG: Looking for reports in {len(outputs_dirs)} output directories")
    print(f"DEBUG: Target workflows: {list(TARGET_WORKFLOWS.keys())}")
    print(f"DEBUG: Target base models: {BASE_MODELS}")
    
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            print(f"DEBUG: Outputs directory does not exist: {outputs_dir}")
            continue
        
        print(f"DEBUG: Scanning {outputs_dir}")
        
        # Check if outputs_dir has any subdirectories
        try:
            subdirs = [d.name for d in outputs_dir.iterdir() if d.is_dir()]
            print(f"DEBUG: Found subdirectories in {outputs_dir}: {subdirs}")
        except Exception as e:
            print(f"DEBUG: Error listing {outputs_dir}: {e}")
            continue
        
        # Iterate through dataset directories
        try:
            dataset_dirs = list(outputs_dir.iterdir())
        except (PermissionError, OSError) as e:
            print(f"DEBUG: Error accessing {outputs_dir}: {e}")
            continue
        
        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name
            print(f"DEBUG: Checking dataset: {dataset}")
            if dataset not in ["dolfin", "wmt25"]:
                print(f"DEBUG: Skipping dataset {dataset} (not dolfin or wmt25)")
                continue
            
            print(f"DEBUG: Processing dataset: {dataset}")
            
            # Iterate through language pair directories
            try:
                lang_pair_dirs = list(dataset_dir.iterdir())
            except (PermissionError, OSError) as e:
                print(f"DEBUG: Error accessing {dataset_dir}: {e}")
                continue
            
            lang_pairs_found = []
            for lang_pair_dir in lang_pair_dirs:
                if lang_pair_dir.is_dir():
                    lang_pairs_found.append(lang_pair_dir.name)
            
            print(f"DEBUG: Found {len(lang_pairs_found)} language pairs in {dataset}: {lang_pairs_found}")
            
            for lang_pair_dir in lang_pair_dirs:
                if not lang_pair_dir.is_dir():
                    continue
                
                lang_pair = lang_pair_dir.name
                
                # List all workflow directories found
                try:
                    workflow_dirs = list(lang_pair_dir.iterdir())
                except (PermissionError, OSError) as e:
                    print(f"DEBUG: Error accessing {lang_pair_dir}: {e}")
                    continue
                
                workflows_found = []
                for wf_dir in workflow_dirs:
                    if wf_dir.is_dir():
                        workflows_found.append(wf_dir.name)
                print(f"DEBUG: Found {len(workflows_found)} workflow directories in {dataset}/{lang_pair}: {workflows_found}")
                
                # Iterate through workflow directories
                for workflow_dir in workflow_dirs:
                    if not workflow_dir.is_dir():
                        continue
                    
                    workflow_dir_name = workflow_dir.name
                    print(f"DEBUG: Checking workflow directory: {workflow_dir_name} in {dataset}/{lang_pair}")
                    
                    # Check if this is one of our target workflows (by directory name)
                    if workflow_dir_name not in TARGET_WORKFLOWS:
                        print(f"DEBUG: Skipping workflow {workflow_dir_name} (not in TARGET_WORKFLOWS)")
                        continue
                    
                    print(f"DEBUG: Found target workflow directory: {workflow_dir_name} in {dataset}/{lang_pair}")
                    
                    # Iterate through model directories
                    try:
                        model_dirs = list(workflow_dir.iterdir())
                    except (PermissionError, OSError) as e:
                        print(f"DEBUG: Error accessing {workflow_dir}: {e}")
                        continue
                    
                    for model_dir in model_dirs:
                        if not model_dir.is_dir():
                            continue
                        
                        try:
                            model = model_dir.name
                        except (OSError, AttributeError) as e:
                            print(f"DEBUG: Error reading model directory name: {e}")
                            continue
                        
                        # Parse model combination
                        parsed = parse_model_combination(model)
                        if parsed is None:
                            continue
                        
                        base_model, postedit_model = parsed
                        
                        # Only include if base model is one of our target base models
                        if base_model not in BASE_MODELS:
                            print(f"DEBUG: Skipping {model} - base_model {base_model} not in BASE_MODELS")
                            continue
                        
                        # Check if it's a combination or single model
                        # We want: gpt-4-1, gpt-4-1-nano, gpt-4-1-nano+gpt-4-1, gpt-4-1+gpt-4-1-nano
                        if postedit_model is not None and postedit_model not in BASE_MODELS:
                            print(f"DEBUG: Skipping {model} - postedit_model {postedit_model} not in BASE_MODELS")
                            continue
                        
                        report_path = model_dir / "report.json"
                        if not report_path.exists():
                            print(f"DEBUG: Report.json not found (experiment may still be running): {report_path}")
                            continue
                        
                        print(f"DEBUG: Parsing report: {report_path}")
                        try:
                            report_data = parse_report(report_path)
                        except Exception as e:
                            print(f"DEBUG: Error parsing report {report_path}: {e}")
                            continue
                        
                        if report_data is None:
                            print(f"DEBUG: Failed to parse report (incomplete or invalid, experiment may still be running): {report_path}")
                            continue
                        
                        print(f"DEBUG: Successfully parsed report for {workflow_dir_name}/{model}: chrF={report_data.get('chrf')}, tokens={report_data.get('tokens_input')}+{report_data.get('tokens_output')}")
                        print(f"DEBUG: Workflow name stored in report.json: '{report_data.get('workflow')}'")
                        
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
                            print(f"DEBUG: Base model tokens: {report_data['base_model_tokens_input']}+{report_data['base_model_tokens_output']}")
                        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                            report_data["base_model_tokens_input"] = 0
                            report_data["base_model_tokens_output"] = 0
                            print(f"DEBUG: Could not extract base_model_tokens: {e}")
                        
                        reports_by_dataset[dataset][lang_pair].append(report_data)
                        print(f"DEBUG: Added report to collection (total for {dataset}/{lang_pair}: {len(reports_by_dataset[dataset][lang_pair])})")
    
    # Print summary
    print(f"\nDEBUG: Collection summary:")
    for dataset, lang_pairs in reports_by_dataset.items():
        total_reports = sum(len(reports) for reports in lang_pairs.values())
        print(f"  {dataset}: {total_reports} reports across {len(lang_pairs)} language pairs")
        for lang_pair, reports in lang_pairs.items():
            if reports:
                print(f"    {lang_pair}: {len(reports)} reports")
                for r in reports:
                    print(f"      - {r.get('workflow')}/{r.get('model_name')}: chrF={r.get('chrf'):.2f}")
    
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
        
        print(f"DEBUG: Processing report - workflow_name in report.json: '{workflow_name}', acronym: '{workflow}'")
        
        # Only include target workflows
        # Note: TARGET_WORKFLOWS.values() contains the acronyms: "IRB", "MaMT", "MAATS_multi"
        target_acronyms = set(TARGET_WORKFLOWS.values())
        print(f"DEBUG: Target acronyms: {target_acronyms}")
        if workflow not in target_acronyms:
            print(f"DEBUG: Skipping workflow '{workflow}' (from '{workflow_name}') - not in target workflows {target_acronyms}")
            continue
        
        print(f"DEBUG: Workflow '{workflow}' (from '{workflow_name}') matches target workflows")
        
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
            print(f"DEBUG: Failed to calculate cost for {workflow}/{model_name}")
            continue
        
        print(f"DEBUG: Calculated cost for {workflow}/{model_name}: ${cost:.4f} (base_tokens: {base_model_tokens_input}+{base_model_tokens_output}, main_tokens: {tokens_input}+{tokens_output})")
        
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
            print(f"DEBUG: Skipping {workflow}/{model_name} - no valid data (chrf_values: {len(data['chrf_values'])}, costs: {len(data['costs'])})")
            continue
        
        result[(workflow, model_name)] = {
            "chrf": sum(data["chrf_values"]) / len(data["chrf_values"]),
            "cost": sum(data["costs"]),
            "base_model": data["base_model"],
            "postedit_model": data["postedit_model"]
        }
        print(f"DEBUG: Aggregated {workflow}/{model_name}: chrF={result[(workflow, model_name)]['chrf']:.2f}, cost=${result[(workflow, model_name)]['cost']:.4f}")
    
    print(f"\nDEBUG: Aggregation summary: {len(result)} workflow+model combinations")
    return result


def plot_translate_vs_postedit(
    dolfin_data: Dict[Tuple[str, str], Dict],
    wmt25_data: Dict[Tuple[str, str], Dict],
    output_path: Path
):
    """Create figure with two subplots comparing translation vs post-editing models."""
    
    # Create figure with two subplots side by side (square shape)
    # Each subplot should be square, so if width is 3.5, height should also be 3.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5))  # Two square subplots side by side
    
    # Plot DOLFIN
    plot_dataset_subplot(ax1, dolfin_data, "DOLFIN")
    
    # Plot WMT25-Term
    plot_dataset_subplot(ax2, wmt25_data, "WMT25-Term")
    
    # Create legend above subplots
    create_legend_above(fig, dolfin_data, wmt25_data)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space at top for two legend lines
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created translate vs postedit plot: {output_path}")


def plot_dataset_subplot(ax, data: Dict[Tuple[str, str], Dict], dataset_name: str):
    """Plot data for a single dataset subplot."""
    
    # Organize data by workflow
    workflow_data = defaultdict(lambda: {
        "gpt-4-1": None,                      # Single model
        "gpt-4-1-nano": None,                 # Single model
        "gpt-4-1-nano+gpt-4-1": None,         # Combination
        "gpt-4-1+gpt-4-1-nano": None,         # Combination
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
        
        # Define lines for different model combinations:
        # Line 1: gpt-4-1-nano -> gpt-4-1-nano+gpt-4-1 -> gpt-4-1
        # Line 2: gpt-4-1-nano -> gpt-4-1+gpt-4-1-nano -> gpt-4-1
        line1_order = [
            ("gpt-4-1-nano", wf_data["gpt-4-1-nano"]),
            ("gpt-4-1-nano+gpt-4-1", wf_data["gpt-4-1-nano+gpt-4-1"]),
            ("gpt-4-1", wf_data["gpt-4-1"]),
        ]
        line2_order = [
            ("gpt-4-1-nano", wf_data["gpt-4-1-nano"]),
            ("gpt-4-1+gpt-4-1-nano", wf_data["gpt-4-1+gpt-4-1-nano"]),
            ("gpt-4-1", wf_data["gpt-4-1"]),
        ]
        
        # Filter out None points for each line (handles missing experiments gracefully)
        valid_points_line1 = [(name, data) for name, data in line1_order if data is not None]
        valid_points_line2 = [(name, data) for name, data in line2_order if data is not None]
        
        # Draw lines if we have at least 2 points (skip if experiment is still running)
        for line_num, valid_points in enumerate([valid_points_line1, valid_points_line2], 1):
            if len(valid_points) >= 2:
                try:
                    costs = [p[1]["cost"] for p in valid_points]
                    chrfs = [p[1]["chrf"] for p in valid_points]
                    ax.plot(costs, chrfs, color=color, linestyle=':', linewidth=1.0, 
                            alpha=0.6, zorder=1)
                except (KeyError, TypeError) as e:
                    print(f"DEBUG: Error plotting line {line_num} for {workflow}: {e}")
        
        # Collect all unique points to plot markers (only plot points that exist)
        all_points = {}
        for name, data in valid_points_line1 + valid_points_line2:
            if name not in all_points and data is not None:
                all_points[name] = data
        
        # Plot markers for all points (skip missing experiments)
        for model_name, point_data in all_points.items():
            try:
                marker = MODEL_MARKERS.get(model_name, "o")
                ax.scatter(point_data["cost"], point_data["chrf"], 
                          c=color, marker=marker, s=95,  # Increased from 63 (1.5x)
                          edgecolors='black', linewidths=0.5, alpha=1.0, zorder=5)  # Non-transparent
            except (KeyError, TypeError) as e:
                print(f"DEBUG: Error plotting marker for {workflow}/{model_name}: {e}")
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel('Cost ($, log scale)', fontsize=10)
    ax.set_ylabel('chrF++', fontsize=10)
    ax.set_title(dataset_name, fontsize=11, fontweight='bold')
    
    # Auto-scale y-axis, then adjust ticks to use steps of at least 5
    ax.set_ylim(auto=True)
    y_min, y_max = ax.get_ylim()
    # Round to nearest 5 for cleaner ticks
    y_min_rounded = 5 * (int(y_min) // 5)
    y_max_rounded = 5 * ((int(y_max) + 4) // 5)  # Round up
    # Generate tick locations with step of at least 5
    step = max(5, (y_max_rounded - y_min_rounded) // 10)  # Aim for ~10 ticks, min step 5
    step = 5 * ((step + 4) // 5)  # Round step up to nearest 5
    y_ticks = list(range(y_min_rounded, y_max_rounded + step, step))
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min_rounded, y_max_rounded)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)


def create_legend_above(fig, dolfin_data: Dict, wmt25_data: Dict):
    """Create legend above the subplots with workflows on top line and models on bottom line."""
    
    # Collect unique workflows and models from both datasets
    all_workflows = set()
    all_models = set()
    
    for data in [dolfin_data, wmt25_data]:
        for (workflow, model_name) in data.keys():
            all_workflows.add(workflow)
            all_models.add(model_name)
    
    # Create legend elements for workflows
    workflow_elements = []
    workflow_order = ["IRB", "MaMT", "MAATS_multi"]  # Maintain consistent order
    for workflow in workflow_order:
        if workflow not in all_workflows:
            continue
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
        "gpt-4-1-nano",
        "gpt-4-1-nano+gpt-4-1",
        "gpt-4-1+gpt-4-1-nano"
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
    
    # Create two separate legends: workflows on top, models below
    if workflow_elements:
        fig.legend(handles=workflow_elements, loc='upper center', ncol=len(workflow_elements),
                  frameon=True, fontsize=9, borderpad=0.2, columnspacing=0.8,
                  handletextpad=0.3, handlelength=1.0, bbox_to_anchor=(0.5, 0.98))
    
    if model_elements:
        fig.legend(handles=model_elements, loc='upper center', ncol=len(model_elements),
                  frameon=True, fontsize=9, borderpad=0.2, columnspacing=0.8,
                  handletextpad=0.3, handlelength=1.0, bbox_to_anchor=(0.5, 0.94))


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


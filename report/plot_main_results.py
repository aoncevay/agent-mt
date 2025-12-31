#!/usr/bin/env python3
"""
Plot cost-performance trade-off for agentic MT workflows.

Creates PDF plots showing chrF score vs total tokens (log scale) for different
workflows and models. One plot per dataset/lang_pair combination.

Usage:
    python report/plot_main_results.py --outputs_dir zhijin/agent-mt-main/outputs --output_dir report/figs

The script will:
- Read all report.json files from the outputs directory
- Filter out incomplete experiments (total_samples != successful_samples)
- Skip *.term workflows for WMT25 dataset
- Create one PDF plot per dataset/lang_pair (e.g., "dolfin_en_de_cost_performance.pdf")
- Create a separate legend PDF
- Print a list of incomplete settings at the end
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Language ID to name mapping
LANGUAGE_ID2NAME = {
    "en": "English",
    "zht": "Traditional Chinese",
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "ko": "Korean",
    "vi": "Vietnamese"
}

# Dataset display names
DATASET_DISPLAY_NAMES = {
    "dolfin": "DOLFIN",
    "wmt25": "WMT25",
    "irs": "IRS"
}

# Workflow name to acronym mapping (from src/workflow_acronyms.py)
WORKFLOW_ACRONYMS = {
    "zero_shot": "ZS",
    "zero_shot_term": "ZS",
    "MaMT_translate_postedit": "MaMT",
    "MaMT_translate_postedit_proofread": "MaMT",
    "SbS_step_by_step": "SbS",
    "SbS_chat_step_by_step": "SbS_chat",
    "MAATS_multi_agents": "MAATS_multi",
    "MAATS_single_agent": "MAATS_single",
    "IRB_refine": "IRB",
    "DeLTA_multi_agents": "DeLTA",
    "ADT_multi_agents": "ADT"
}

# Workflow colors - using a colorblind-friendly palette
WORKFLOW_COLORS = {
    "ZS": "#1f77b4",           # blue
    "MaMT": "#ff7f0e",          # orange
    "SbS_chat": "#2ca02c",      # green
    "MAATS_multi": "#d62728",   # red
    "IRB": "#9467bd",           # purple
    "DeLTA": "#8c564b",         # brown
    "ADT": "#e377c2",           # pink
    "MAATS_single": "#7f7f7f",  # gray
    "SbS": "#bcbd22",           # olive
}

# Workflow display names for legend
WORKFLOW_DISPLAY_NAMES = {
    "ZS": "Zero-shot",
    "MaMT": "MaMT",
    "SbS_chat": "Step-by-step",
    "MAATS_multi": "MAATS",
    "IRB": "IRB",
    "DeLTA": "DeLTA",
    "ADT": "ADT",
    "MAATS_single": "MAATS (single)",
    "SbS": "SbS",
}

# Model markers - using different shapes
MODEL_MARKERS = {
    "qwen3-32b": "o",           # circle
    "qwen3-235b": "s",          # square
    "gpt-oss-20b": "^",         # triangle up
    "gpt-oss-120b": "v",        # triangle down
    "claude-sonnet-4": "P",     # plus (filled)
    "gpt-4-1": "*"             # star
}

# Default marker if model not in dict
DEFAULT_MARKER = "o"

# Pricing dictionary - prices per 1,000 tokens (standard tier)
# Note: OpenAI gpt-4-1 is per 1M tokens, converted to per 1k here
# Format: {model: {"input": price_per_1k, "output": price_per_1k}}
MODEL_PRICING_STANDARD = {
    # Qwen models (Bedrock) - per 1k tokens
    "qwen3-32b": {"input": 0.00015, "output": 0.0006},
    "qwen3-235b": {"input": 0.00022, "output": 0.00088},
    
    # OpenAI models (Bedrock) - per 1k tokens
    "gpt-oss-20b": {"input": 0.00007, "output": 0.0003},
    "gpt-oss-120b": {"input": 0.00015, "output": 0.0006},
    
    # Anthropic models (Bedrock) - per 1k tokens
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "claude-opus-4-5": {"input": 0.005, "output": 0.025},
    
    # OpenAI models (OpenAI API) - per 1M tokens, converted to per 1k
    # https://platform.openai.com/docs/pricing?latest-pricing=standard
    "gpt-4-1": {"input": 0.002, "output": 0.008},  # $2.00/1M = $0.002/1k, $8.00/1M = $0.008/1k
}

# Pricing dictionary - prices per 1,000 tokens (batch tier)
MODEL_PRICING_BATCH = {
    # Qwen models (Bedrock) - per 1k tokens
    "qwen3-32b": {"input": 0.000075, "output": 0.0003},
    "qwen3-235b": {"input": 0.00011, "output": 0.00044},
    
    # OpenAI models (Bedrock) - per 1k tokens
    "gpt-oss-20b": {"input": 0.000035, "output": 0.00015},
    "gpt-oss-120b": {"input": 0.000075, "output": 0.0003},
    
    # Anthropic models (Bedrock) - per 1k tokens
    "claude-sonnet-4": {"input": 0.0015, "output": 0.0075},
    "claude-opus-4-5": {"input": 0.0025, "output": 0.0125},
    
    # OpenAI models (OpenAI API) - per 1M tokens, converted to per 1k
    # https://platform.openai.com/docs/pricing?latest-pricing=batch
    "gpt-4-1": {"input": 0.001, "output": 0.004},  
}


def format_lang_pair(lang_pair: str) -> str:
    """Format language pair for display (e.g., 'en_de' -> 'en-de', 'en-zht' -> 'en-zht')."""
    if "_" in lang_pair:
        return lang_pair.replace("_", "-")
    return lang_pair


def get_lang_pair_display(lang_pair: str) -> str:
    """Get display name for language pair (e.g., 'en_de' -> 'en-de')."""
    return format_lang_pair(lang_pair)


def parse_report(report_path: Path) -> Optional[Dict]:
    """Parse a report.json file and return relevant data."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if experiment is complete
        total_samples = data.get("total_samples", 0)
        successful_samples = data.get("successful_samples", 0)
        
        # Experiment is incomplete if:
        # - total_samples is 0 (no samples processed)
        # - total_samples != successful_samples (some samples failed)
        if total_samples == 0 or total_samples != successful_samples:
            return None  # Incomplete experiment
        
        summary = data.get("summary", {})
        avg_chrf = summary.get("avg_chrf_score")
        avg_term_acc = summary.get("avg_term_success_rate")
        total_tokens_input = summary.get("total_tokens_input", 0)
        total_tokens_output = summary.get("total_tokens_output", 0)
        total_tokens = total_tokens_input + total_tokens_output
        
        if avg_chrf is None or total_tokens == 0:
            return None
        
        return {
            "workflow": data.get("workflow", ""),
            "model": data.get("model", ""),
            "dataset": data.get("dataset", ""),
            "lang_pair": data.get("lang_pair", ""),
            "chrf": avg_chrf,
            "term_acc": avg_term_acc,  # Can be None if not available
            "total_tokens": total_tokens,
            "tokens_input": total_tokens_input,
            "tokens_output": total_tokens_output,
        }
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not parse {report_path}: {e}")
        return None


def collect_reports_from_dir(outputs_dir: Path, seen_settings: Set[Tuple[str, str, str, str]], 
                              reports_by_dataset_lang: Dict, reports_by_dataset_lang_term: Dict,
                              incomplete_settings: Set):
    """
    Collect all valid reports from a single outputs directory.
    Updates seen_settings, reports_by_dataset_lang, reports_by_dataset_lang_term, and incomplete_settings in place.
    
    Args:
        outputs_dir: Path to outputs directory
        seen_settings: Set of (dataset, lang_pair, workflow, model) tuples already found
        reports_by_dataset_lang: Dictionary to update with reports (non-term workflows)
        reports_by_dataset_lang_term: Dictionary to update with reports (term workflows)
        incomplete_settings: Set to update with incomplete settings
    """
    # Iterate through dataset directories
    for dataset_dir in outputs_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset = dataset_dir.name
        
        # Iterate through language pair directories
        for lang_pair_dir in dataset_dir.iterdir():
            if not lang_pair_dir.is_dir():
                continue
            
            lang_pair = lang_pair_dir.name
            
            # Iterate through workflow directories
            for workflow_dir in lang_pair_dir.iterdir():
                if not workflow_dir.is_dir():
                    continue
                
                workflow_acronym = workflow_dir.name
                is_term_workflow = workflow_acronym.endswith(".term")
                
                # Determine which dictionary to use
                target_dict = reports_by_dataset_lang_term if is_term_workflow else reports_by_dataset_lang
                
                # Only process *.term workflows for WMT25
                if is_term_workflow and dataset != "wmt25":
                    continue
                
                # Iterate through model directories
                for model_dir in workflow_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    model = model_dir.name
                    
                    # Check if we've already seen this setting (from a previous directory)
                    setting_key = (dataset, lang_pair, workflow_acronym, model)
                    if setting_key in seen_settings:
                        continue  # Skip if already found in a previous directory
                    
                    report_path = model_dir / "report.json"
                    
                    if not report_path.exists():
                        incomplete_settings.add(setting_key)
                        continue
                    
                    report_data = parse_report(report_path)
                    
                    if report_data is None:
                        incomplete_settings.add(setting_key)
                        continue
                    
                    # Mark as seen and add to reports
                    seen_settings.add(setting_key)
                    target_dict[(dataset, lang_pair)].append(report_data)


def collect_reports(outputs_dirs: List[Path]) -> Tuple[Dict, Dict, Set[Tuple[str, str, str, str]]]:
    """
    Collect all valid reports from multiple outputs directories.
    If there's overlap, uses the first one found.
    
    Args:
        outputs_dirs: List of paths to outputs directories
    
    Returns:
        - Dictionary mapping (dataset, lang_pair) -> list of report data (non-term workflows)
        - Dictionary mapping (dataset, lang_pair) -> list of report data (term workflows)
        - Set of incomplete settings (dataset, lang_pair, workflow, model)
    """
    reports_by_dataset_lang = defaultdict(list)
    reports_by_dataset_lang_term = defaultdict(list)  # For *.term workflows
    incomplete_settings = set()
    seen_settings = set()  # Track which settings we've already found
    
    # Process each outputs directory in order
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            print(f"Warning: Outputs directory does not exist: {outputs_dir}")
            continue
        
        collect_reports_from_dir(outputs_dir, seen_settings, reports_by_dataset_lang, 
                                 reports_by_dataset_lang_term, incomplete_settings)
    
    return reports_by_dataset_lang, reports_by_dataset_lang_term, incomplete_settings


def get_workflow_acronym(workflow_name: str) -> str:
    """Convert workflow name to acronym."""
    return WORKFLOW_ACRONYMS.get(workflow_name, workflow_name)


def get_workflow_acronym_from_dir(workflow_dir_name: str) -> str:
    """Extract workflow acronym from directory name (removes .term suffix if present)."""
    if workflow_dir_name.endswith(".term"):
        return workflow_dir_name[:-5]  # Remove ".term"
    return workflow_dir_name


def get_model_base_cost(model: str, use_batch: bool = False) -> Optional[float]:
    """
    Get the base API cost for a model (average of input and output prices per 1k tokens).
    Used for ordering models by cost.
    
    Args:
        model: Model name
        use_batch: Whether to use batch pricing (default: False, uses standard)
    
    Returns:
        Average cost per 1k tokens, or None if model pricing not available
    """
    pricing_dict = MODEL_PRICING_BATCH if use_batch else MODEL_PRICING_STANDARD
    
    if model not in pricing_dict:
        return None
    
    prices = pricing_dict[model]
    # Use average of input and output prices as base cost
    return (prices["input"] + prices["output"]) / 2.0


def calculate_cost(tokens_input: int, tokens_output: int, model: str, 
                   use_batch: bool = False) -> Optional[float]:
    """
    Calculate cost in dollars from input and output tokens.
    
    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        model: Model name
        use_batch: Whether to use batch pricing (default: False, uses standard)
    
    Returns:
        Total cost in dollars, or None if model pricing not available
    """
    pricing_dict = MODEL_PRICING_BATCH if use_batch else MODEL_PRICING_STANDARD
    
    if model not in pricing_dict:
        return None
    
    prices = pricing_dict[model]
    cost_input = (tokens_input / 1000.0) * prices["input"]
    cost_output = (tokens_output / 1000.0) * prices["output"]
    
    return cost_input + cost_output


def create_workflow_legend(workflows: Set[str], output_path: Path):
    """Create a separate figure with the workflow legend."""
    _fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')
    
    # Create legend entries for workflows
    workflow_elements = []
    
    for workflow in sorted(workflows):
        if workflow in WORKFLOW_COLORS:
            color = WORKFLOW_COLORS[workflow]
            display_name = WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)
            workflow_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black', label=display_name)
            )
    
    if workflow_elements:
        legend = ax.legend(handles=workflow_elements, loc='center', 
                          frameon=True, fontsize=10, title='System',
                          borderpad=0.2, columnspacing=0.5, handletextpad=0.3,
                          handlelength=1.0)
        legend.get_title().set_fontweight('bold')
    
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()


def create_model_legend(output_path: Path):
    """Create a separate figure with the model legend (all models from MODEL_MARKERS)."""
    _fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')
    
    # Create legend entries for all models in MODEL_MARKERS, sorted by cost (most expensive first)
    model_elements = []
    
    # Get all models with their base costs and sort by cost (descending)
    models_with_costs = []
    for model in MODEL_MARKERS.keys():
        base_cost = get_model_base_cost(model, use_batch=False)
        if base_cost is not None:
            models_with_costs.append((model, base_cost))
    
    # Sort by cost (most expensive first)
    models_with_costs.sort(key=lambda x: x[1], reverse=True)
    
    for model, _ in models_with_costs:
        marker = MODEL_MARKERS[model]
        model_elements.append(
            plt.Line2D([0], [0], marker=marker, color='black', linestyle='None',
                      markersize=10, label=model, markerfacecolor='black',
                      markeredgecolor='black')
        )
    
    if model_elements:
        legend = ax.legend(handles=model_elements, loc='center', 
                          frameon=True, fontsize=10, title='Model',
                          borderpad=0.2, columnspacing=0.5, handletextpad=0.3,
                          handlelength=1.0)
        legend.get_title().set_fontweight('bold')
    
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()


def plot_dataset_lang_pair(
    dataset: str,
    lang_pair: str,
    reports: List[Dict],
    output_dir: Path
):
    """Create a plot for a specific dataset/lang_pair combination."""
    if not reports:
        return
    
    # Prepare data
    workflows = set()
    models = set()
    data_points = []
    
    for report in reports:
        # The report contains the full workflow name, convert to acronym
        workflow_name = report.get("workflow", "")
        workflow = get_workflow_acronym(workflow_name)
        model = report["model"]
        workflows.add(workflow)
        models.add(model)
        
        data_points.append({
            "workflow": workflow,
            "model": model,
            "chrf": report["chrf"],
            "total_tokens": report["total_tokens"],
            "tokens_input": report.get("tokens_input", 0),
            "tokens_output": report.get("tokens_output", 0),
            "workflow_name": report.get("workflow", "")  # Keep original for reference
        })
    
    if not data_points:
        return
    
    # Create figure - single column width for ACL paper (about 3.5 inches), 3:2 ratio
    _fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot each workflow with different colors
    for workflow in sorted(workflows):
        workflow_data = [d for d in data_points if d["workflow"] == workflow]
        
        if not workflow_data:
            continue
        
        color = WORKFLOW_COLORS.get(workflow, "#000000")
        
        # Plot each model for this workflow
        for model in sorted(models):
            model_data = [d for d in workflow_data if d["model"] == model]
            
            if not model_data:
                continue
            
            tokens = [d["total_tokens"] for d in model_data]
            chrfs = [d["chrf"] for d in model_data]
            
            marker = MODEL_MARKERS.get(model, DEFAULT_MARKER)
            # Make star markers slightly larger for better visibility
            marker_size = 70 if marker == "*" else 50
            ax.scatter(tokens, chrfs, c=color, marker=marker, s=marker_size, 
                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=3)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel('Total Tokens (log scale)', fontsize=10)
    ax.set_ylabel('chrF++', fontsize=10)
    
    # Set y-axis limits based on language pair
    if lang_pair == "en-zht":
        ax.set_ylim(25, 50)
    else:
        ax.set_ylim(55, 80)
    
    # Note: No title as per user request (captions will be in paper)
    
    # Grid (behind everything)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    safe_dataset = dataset.replace("/", "_")
    safe_lang_pair = lang_pair.replace("/", "_")
    output_path = output_dir / f"{safe_dataset}_{safe_lang_pair}_chrF_x_tokens.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created plot: {output_path}")


def plot_dataset_lang_pair_price(
    dataset: str,
    lang_pair: str,
    reports: List[Dict],
    output_dir: Path,
    use_batch: bool = False,
    is_term: bool = False
):
    """Create a price-based plot for a specific dataset/lang_pair combination."""
    if not reports:
        return
    
    # Prepare data
    workflows = set()
    models = set()
    data_points = []
    
    for report in reports:
        # The report contains the full workflow name, convert to acronym
        workflow_name = report.get("workflow", "")
        workflow = get_workflow_acronym(workflow_name)
        model = report["model"]
        workflows.add(workflow)
        models.add(model)
        
        # Calculate cost
        tokens_input = report.get("tokens_input", 0)
        tokens_output = report.get("tokens_output", 0)
        cost = calculate_cost(tokens_input, tokens_output, model, use_batch)
        
        if cost is None:
            continue  # Skip if pricing not available for this model
        
        data_points.append({
            "workflow": workflow,
            "model": model,
            "chrf": report["chrf"],
            "cost": cost,
            "workflow_name": report.get("workflow", "")  # Keep original for reference
        })
    
    if not data_points:
        return
    
    # Create figure - single column width for ACL paper (about 3.5 inches), 3:2 ratio
    _fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # First pass: collect all workflow points and draw lines (behind markers)
    workflow_points_dict = {}
    for workflow in sorted(workflows):
        workflow_data = [d for d in data_points if d["workflow"] == workflow]
        
        if not workflow_data:
            continue
        
        color = WORKFLOW_COLORS.get(workflow, "#000000")
        
        # Collect all points for this workflow
        workflow_points = []
        for model in models:
            model_data = [d for d in workflow_data if d["model"] == model]
            if model_data:
                # Take first point (should only be one per model per workflow)
                point = model_data[0]
                workflow_points.append((point["cost"], point["chrf"], model))
        
        # Sort by cost (cheapest to most expensive) for connecting lines
        workflow_points_sorted = sorted(workflow_points, key=lambda x: x[0])
        workflow_points_dict[workflow] = (workflow_points_sorted, color)
        
        # Draw dotted lines first (behind markers)
        if len(workflow_points_sorted) > 1:
            costs_line = [p[0] for p in workflow_points_sorted]
            chrfs_line = [p[1] for p in workflow_points_sorted]
            # Draw dotted line connecting points (thicker line, behind markers)
            ax.plot(costs_line, chrfs_line, color=color, linestyle=':', linewidth=1.5, alpha=0.6, zorder=0)
    
    # Second pass: plot all markers on top
    for workflow in sorted(workflows):
        if workflow not in workflow_points_dict:
            continue
        workflow_points_sorted, color = workflow_points_dict[workflow]
        
        # Plot each model for this workflow (on top of lines)
        for cost, chrf, model in workflow_points_sorted:
            marker = MODEL_MARKERS.get(model, DEFAULT_MARKER)
            # Make star markers slightly larger for better visibility
            marker_size = 70 if marker == "*" else 50
            ax.scatter(cost, chrf, c=color, marker=marker, s=marker_size, 
                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel('Cost ($, log scale)', fontsize=10)
    ax.set_ylabel('chrF++', fontsize=10)
    
    # Set y-axis limits based on language pair
    if lang_pair == "en-zht":
        ax.set_ylim(25, 50)
    else:
        ax.set_ylim(55, 80)
    
    # Note: No title as per user request (captions will be in paper)
    
    # Grid (behind everything)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with appropriate naming
    safe_dataset = dataset.replace("/", "_")
    safe_lang_pair = lang_pair.replace("/", "_")
    if is_term:
        # For term workflows: wmt25+T_{lang_pair}_chrF_x_price.pdf
        output_path = output_dir / f"{safe_dataset}+T_{safe_lang_pair}_chrF_x_price.pdf"
    else:
        output_path = output_dir / f"{safe_dataset}_{safe_lang_pair}_chrF_x_price.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created price plot: {output_path}")


def plot_dataset_lang_pair_term_acc(
    dataset: str,
    lang_pair: str,
    reports: List[Dict],
    output_dir: Path,
    use_batch: bool = False
):
    """Create a terminology accuracy plot for *.term workflows."""
    if not reports:
        return
    
    # Prepare data
    workflows = set()
    models = set()
    data_points = []
    
    for report in reports:
        # The report contains the full workflow name, convert to acronym
        workflow_name = report.get("workflow", "")
        workflow = get_workflow_acronym(workflow_name)
        model = report["model"]
        
        # Skip if term accuracy not available
        term_acc = report.get("term_acc")
        if term_acc is None:
            continue
        
        workflows.add(workflow)
        models.add(model)
        
        # Calculate cost
        tokens_input = report.get("tokens_input", 0)
        tokens_output = report.get("tokens_output", 0)
        cost = calculate_cost(tokens_input, tokens_output, model, use_batch)
        
        if cost is None:
            continue  # Skip if pricing not available for this model
        
        data_points.append({
            "workflow": workflow,
            "model": model,
            "term_acc": term_acc,
            "cost": cost,
            "workflow_name": report.get("workflow", "")  # Keep original for reference
        })
    
    if not data_points:
        return
    
    # Create figure - single column width for ACL paper (about 3.5 inches), 3:2 ratio
    _fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # First pass: collect all workflow points and draw lines (behind markers)
    workflow_points_dict = {}
    for workflow in sorted(workflows):
        workflow_data = [d for d in data_points if d["workflow"] == workflow]
        
        if not workflow_data:
            continue
        
        color = WORKFLOW_COLORS.get(workflow, "#000000")
        
        # Collect all points for this workflow
        workflow_points = []
        for model in models:
            model_data = [d for d in workflow_data if d["model"] == model]
            if model_data:
                # Take first point (should only be one per model per workflow)
                point = model_data[0]
                workflow_points.append((point["cost"], point["term_acc"], model))
        
        # Sort by cost (cheapest to most expensive) for connecting lines
        workflow_points_sorted = sorted(workflow_points, key=lambda x: x[0])
        workflow_points_dict[workflow] = (workflow_points_sorted, color)
        
        # Draw dotted lines first (behind markers)
        if len(workflow_points_sorted) > 1:
            costs_line = [p[0] for p in workflow_points_sorted]
            term_accs_line = [p[1] for p in workflow_points_sorted]
            # Draw dotted line connecting points (thicker line, behind markers)
            ax.plot(costs_line, term_accs_line, color=color, linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Second pass: plot all markers on top
    for workflow in sorted(workflows):
        if workflow not in workflow_points_dict:
            continue
        workflow_points_sorted, color = workflow_points_dict[workflow]
        
        # Plot each model for this workflow (on top of lines)
        for cost, term_acc, model in workflow_points_sorted:
            marker = MODEL_MARKERS.get(model, DEFAULT_MARKER)
            # Make star markers slightly larger for better visibility
            marker_size = 70 if marker == "*" else 50
            ax.scatter(cost, term_acc, c=color, marker=marker, s=marker_size, 
                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel('Cost ($, log scale)', fontsize=10)
    ax.set_ylabel('Terminology Accuracy', fontsize=10)
    
    # Set y-axis limits for TermAcc
    ax.set_ylim(0.5, 0.8)
    
    # Grid (behind everything)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with naming: wmt25+T_{lang_pair}_TAcc_x_price.pdf
    safe_dataset = dataset.replace("/", "_")
    safe_lang_pair = lang_pair.replace("/", "_")
    output_path = output_dir / f"{safe_dataset}+T_{safe_lang_pair}_TAcc_x_price.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created term accuracy plot: {output_path}")


def plot_dataset_avg_price(
    dataset: str,
    reports_by_lang_pair: Dict[str, List[Dict]],
    output_dir: Path,
    metric: str = "chrf",  # "chrf" or "termacc"
    use_batch: bool = False,
    is_term: bool = False
):
    """Create an average plot across all language pairs for a dataset."""
    if not reports_by_lang_pair:
        return
    
    # Aggregate data: for each workflow+model, compute average metric and total cost
    aggregated_data = defaultdict(lambda: {"values": [], "costs": []})
    
    for lang_pair, reports in reports_by_lang_pair.items():
        for report in reports:
            workflow_name = report.get("workflow", "")
            workflow = get_workflow_acronym(workflow_name)
            model = report["model"]
            key = (workflow, model)
            
            # Get metric value
            if metric == "chrf":
                value = report.get("chrf")
            elif metric == "termacc":
                value = report.get("term_acc")
            else:
                continue
            
            if value is None:
                continue
            
            # Calculate cost for this lang pair
            tokens_input = report.get("tokens_input", 0)
            tokens_output = report.get("tokens_output", 0)
            cost = calculate_cost(tokens_input, tokens_output, model, use_batch)
            
            if cost is None:
                continue
            
            aggregated_data[key]["values"].append(value)
            aggregated_data[key]["costs"].append(cost)
    
    if not aggregated_data:
        return
    
    # Prepare data points: average metric, total cost
    workflows = set()
    models = set()
    data_points = []
    
    for (workflow, model), data in aggregated_data.items():
        if not data["values"] or not data["costs"]:
            continue
        
        avg_value = sum(data["values"]) / len(data["values"])
        total_cost = sum(data["costs"])  # Total cost across all lang pairs
        
        workflows.add(workflow)
        models.add(model)
        
        data_points.append({
            "workflow": workflow,
            "model": model,
            "value": avg_value,
            "cost": total_cost
        })
    
    if not data_points:
        return
    
    # Create figure
    _fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # First pass: collect all workflow points and draw lines (behind markers)
    workflow_points_dict = {}
    for workflow in sorted(workflows):
        workflow_data = [d for d in data_points if d["workflow"] == workflow]
        
        if not workflow_data:
            continue
        
        color = WORKFLOW_COLORS.get(workflow, "#000000")
        
        # Collect all points for this workflow
        workflow_points = []
        for model in models:
            model_data = [d for d in workflow_data if d["model"] == model]
            if model_data:
                # Take first point (should only be one per model per workflow)
                point = model_data[0]
                workflow_points.append((point["cost"], point["value"], model))
        
        # Sort by cost (cheapest to most expensive) for connecting lines
        workflow_points_sorted = sorted(workflow_points, key=lambda x: x[0])
        workflow_points_dict[workflow] = (workflow_points_sorted, color)
        
        # Draw dotted lines first (behind markers)
        if len(workflow_points_sorted) > 1:
            costs_line = [p[0] for p in workflow_points_sorted]
            values_line = [p[1] for p in workflow_points_sorted]
            # Draw dotted line connecting points (thicker line, behind markers)
            ax.plot(costs_line, values_line, color=color, linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Second pass: plot all markers on top
    for workflow in sorted(workflows):
        if workflow not in workflow_points_dict:
            continue
        workflow_points_sorted, color = workflow_points_dict[workflow]
        
        # Plot each model for this workflow (on top of lines)
        for cost, value, model in workflow_points_sorted:
            marker = MODEL_MARKERS.get(model, DEFAULT_MARKER)
            # Make star markers slightly larger for better visibility
            marker_size = 70 if marker == "*" else 50
            ax.scatter(cost, value, c=color, marker=marker, s=marker_size, 
                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Labels
    ax.set_xlabel('Cost ($, log scale)', fontsize=10)
    if metric == "chrf":
        ax.set_ylabel('chrF++', fontsize=10)
        # Set y-axis limits based on dataset
        if dataset == "dolfin":
            ax.set_ylim(55, 80)
        else:  # wmt25
            ax.set_ylim(35, 60)
    elif metric == "termacc":
        ax.set_ylabel('Terminology Accuracy', fontsize=10)
        ax.set_ylim(0.5, 0.8)
    
    # Grid (behind everything)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    safe_dataset = dataset.replace("/", "_")
    if is_term:
        if metric == "chrf":
            output_path = output_dir / f"{safe_dataset}+T_AVG_chrF_x_price.pdf"
        else:  # termacc
            output_path = output_dir / f"{safe_dataset}+T_AVG_TAcc_x_price.pdf"
    else:
        output_path = output_dir / f"{safe_dataset}_AVG_chrF_x_price.pdf"
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created AVG plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cost-performance trade-off for agentic MT workflows"
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
        help="Directory to save output PDF plots"
    )
    parser.add_argument(
        "--include-tokens",
        action="store_true",
        help="Also create token-based plots (default: only price plots)"
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
    
    if len(existing_dirs) < len(outputs_dirs):
        print("Warning: Some outputs directories do not exist:")
        for d in outputs_dirs:
            if d not in existing_dirs:
                print(f"  - {d} (skipping)")
        print(f"Using {len(existing_dirs)} existing directory/directories:")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all reports from all directories
    print("Collecting reports...")
    print(f"Searching in {len(existing_dirs)} directory/directories:")
    for d in existing_dirs:
        print(f"  - {d}")
    reports_by_dataset_lang, reports_by_dataset_lang_term, incomplete_settings = collect_reports(existing_dirs)
    
    # Collect all workflows for legend (from both term and non-term)
    all_workflows = set()
    
    for reports in reports_by_dataset_lang.values():
        for report in reports:
            workflow_name = report.get("workflow", "")
            workflow_acronym = get_workflow_acronym(workflow_name)
            all_workflows.add(workflow_acronym)
    
    for reports in reports_by_dataset_lang_term.values():
        for report in reports:
            workflow_name = report.get("workflow", "")
            workflow_acronym = get_workflow_acronym(workflow_name)
            all_workflows.add(workflow_acronym)
    
    # Create separate legend figures
    if all_workflows:
        workflow_legend_path = output_dir / "legend_workflow.pdf"
        create_workflow_legend(all_workflows, workflow_legend_path)
        print(f"Created workflow legend: {workflow_legend_path}")
    
    model_legend_path = output_dir / "legend_model.pdf"
    create_model_legend(model_legend_path)
    print(f"Created model legend: {model_legend_path}")
    
    # Create plots for each dataset/lang_pair
    print("\nCreating plots...")
    
    # Create plots for non-term workflows
    for (dataset, lang_pair), reports in sorted(reports_by_dataset_lang.items()):
        # Create token-based plots (optional)
        if args.include_tokens:
            plot_dataset_lang_pair(dataset, lang_pair, reports, output_dir)
        # Create price-based plots (standard pricing)
        plot_dataset_lang_pair_price(dataset, lang_pair, reports, output_dir, use_batch=False)
    
    # Create plots for term workflows (WMT25 only)
    for (dataset, lang_pair), reports in sorted(reports_by_dataset_lang_term.items()):
        # Create chrF vs price plots for term workflows
        plot_dataset_lang_pair_price(dataset, lang_pair, reports, output_dir, use_batch=False, is_term=True)
        # Create Terminology Accuracy vs price plots
        plot_dataset_lang_pair_term_acc(dataset, lang_pair, reports, output_dir, use_batch=False)
    
    # Create AVG plots per dataset
    print("\nCreating AVG plots...")
    
    # DOLFIN AVG plot
    dolfin_reports = {lp: reports for (ds, lp), reports in reports_by_dataset_lang.items() if ds == "dolfin"}
    if dolfin_reports:
        plot_dataset_avg_price("dolfin", dolfin_reports, output_dir, metric="chrf", use_batch=False, is_term=False)
    
    # WMT25+Term AVG plots
    wmt25_term_reports = {lp: reports for (ds, lp), reports in reports_by_dataset_lang_term.items() if ds == "wmt25"}
    if wmt25_term_reports:
        plot_dataset_avg_price("wmt25", wmt25_term_reports, output_dir, metric="chrf", use_batch=False, is_term=True)
        plot_dataset_avg_price("wmt25", wmt25_term_reports, output_dir, metric="termacc", use_batch=False, is_term=True)
    
    # Print incomplete settings
    if incomplete_settings:
        print("\n" + "="*80)
        print("Incomplete Settings (still running or missing reports):")
        print("="*80)
        
        # Group by dataset/lang_pair and filter to only models in MODEL_MARKERS
        incomplete_by_dataset_lang = defaultdict(list)
        filtered_count = 0
        for dataset, lang_pair, workflow, model in sorted(incomplete_settings):
            # Only include models that are in MODEL_MARKERS (models in legend)
            if model in MODEL_MARKERS:
                incomplete_by_dataset_lang[(dataset, lang_pair)].append((workflow, model))
            else:
                filtered_count += 1
        
        if incomplete_by_dataset_lang:
            for (dataset, lang_pair), settings in sorted(incomplete_by_dataset_lang.items()):
                print(f"\n{dataset} / {lang_pair}:")
                for workflow, model in sorted(settings):
                    print(f"  - {workflow} / {model}")
        
        total_shown = sum(len(s) for s in incomplete_by_dataset_lang.values())
        if filtered_count > 0:
            print(f"\nTotal incomplete (shown): {total_shown} settings (filtered out {filtered_count} settings for models not in legend)")
        else:
            print(f"\nTotal incomplete: {total_shown} settings")
    else:
        print("\nAll experiments are complete!")
    
    print(f"\nPlots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())


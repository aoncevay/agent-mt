#!/usr/bin/env python3
"""
Generate LaTeX table with experimental results.

Creates separate tables for DOLFIN and WMT25+Term datasets with workflows as rows
(grouped by workflow type with models as sub-rows) and metrics/language pairs as columns.

Usage:
    python report/write_tables_paper.py --outputs_dirs zhijin/agent-mt-main/outputs outputs

Note: The generated LaTeX tables require the following packages in your .tex file preamble:
    \\usepackage[table]{xcolor}  % For \\cellcolor command
    \\usepackage{booktabs}        % For \\toprule, \\midrule, \\bottomrule, \\cmidrule
    \\usepackage{multirow}        % For \\multirow command
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

# Import from plot script
import importlib.util
plot_script_path = Path(__file__).parent / "plot_main_results.py"
spec = importlib.util.spec_from_file_location("plot_main_results", plot_script_path)
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)

MODEL_MARKERS = plot_module.MODEL_MARKERS
WORKFLOW_ACRONYMS = plot_module.WORKFLOW_ACRONYMS
get_workflow_acronym = plot_module.get_workflow_acronym
parse_report = plot_module.parse_report
collect_reports = plot_module.collect_reports
calculate_cost = plot_module.calculate_cost
get_model_base_cost = plot_module.get_model_base_cost
compute_pareto_ranks = plot_module.compute_pareto_ranks

# Model display names
MODEL_DISPLAY_NAMES = {
    "qwen3-32b": "Qwen 3 32B",
    "qwen3-235b": "Qwen 3 235B",
    "gpt-4-1": "GPT-4.1",
    "gpt-4-1-nano": "GPT-4.1 nano",
    "gpt-5": "GPT-5",
    "gpt-4-1-mini": "GPT-4.1 mini"
}

# Workflow order and display names
WORKFLOW_ORDER = ["ZS", "IRB", "MaMT", "SbS_chat", "MAATS_multi", "DeLTA"]
WORKFLOW_DISPLAY_NAMES = {
    "ZS": "Zero-shot",
    "IRB": "IRB",
    "MaMT": "MaMT",
    "SbS_chat": "Step-by-step",
    "MAATS_multi": "MAATS",
    "DeLTA": "DelTA",
}

# Model order (from larger to smaller) - will be sorted by cost in tables
MODEL_ORDER = ["gpt-4-1", "qwen3-235b", "qwen3-32b", "gpt-4-1-nano"]

def get_models_sorted_by_cost() -> List[str]:
    """Get models sorted by base API cost (most expensive first)."""
    models_with_costs = []
    for model in MODEL_ORDER:
        base_cost = get_model_base_cost(model, use_batch=False)
        if base_cost is not None:
            models_with_costs.append((model, base_cost))
    
    # Sort by cost (most expensive first)
    models_with_costs.sort(key=lambda x: x[1], reverse=True)
    return [model for model, _ in models_with_costs]

# Language pairs
DOLFIN_LANG_PAIRS = ["en_de", "en_es", "en_fr", "en_it"]
WMT25_LANG_PAIRS = ["en-zht", "zht-en"]

# LaTeX color commands (using xcolor package)
def get_color_for_value(value: float, min_val: float, max_val: float, metric_type: str = "chrf") -> str:
    """Get LaTeX color command for a value (green for high, red for low).
    
    Uses logarithmic normalization to better distinguish top systems when there are bottom outliers.
    """
    if value is None or min_val == max_val:
        return ""
    
    # Use logarithmic normalization for better distribution across wide ranges
    # Add small offset to avoid log(0) for metrics that might have values near 0
    offset = 0.01 if metric_type == "termacc" else 0.0
    
    # Check if log scale is applicable (min_val + offset > 0)
    if min_val + offset > 0:
        log_value = math.log(value + offset)
        log_min = math.log(min_val + offset)
        log_max = math.log(max_val + offset)
        normalized = (log_value - log_min) / (log_max - log_min)
    else:
        # Fall back to linear normalization if log scale not applicable
        normalized = (value - min_val) / (max_val - min_val)
    
    # Use different color schemes for different metrics
    if metric_type == "termacc":
        # For TermAcc, high is good (green), low is bad (red)
        if normalized >= 0.7:
            return "\\cellcolor{green!25}"
        elif normalized >= 0.5:
            return "\\cellcolor{green!10}"
        elif normalized >= 0.3:
            return "\\cellcolor{red!10}"
        else:
            return "\\cellcolor{red!25}"
    elif metric_type == "metricx":
        # For MetricX-24, low is good (green), high is bad (red) - inverse scale
        if normalized <= 0.3:
            return "\\cellcolor{green!25}"
        elif normalized <= 0.5:
            return "\\cellcolor{green!10}"
        elif normalized <= 0.7:
            return "\\cellcolor{red!10}"
        else:
            return "\\cellcolor{red!25}"
    else:
        # For chrF++, high is good (green), low is bad (red)
        if normalized >= 0.7:
            return "\\cellcolor{green!25}"
        elif normalized >= 0.5:
            return "\\cellcolor{green!10}"
        elif normalized >= 0.3:
            return "\\cellcolor{red!10}"
        else:
            return "\\cellcolor{red!25}"


def format_value(value: Optional[float], metric_type: str = "chrf") -> str:
    """Format a numeric value for LaTeX, returning empty string if None."""
    if value is None:
        return "---"
    # Use 2 decimals for TermAcc, MetricX, and Cost, 1 for others
    if metric_type == "termacc" or metric_type == "metricx" or metric_type == "cost":
        return f"{value:.2f}"
    return f"{value:.1f}"


def format_cost(value: Optional[float]) -> str:
    """Format cost value without dollar sign (dollar sign is in header)."""
    if value is None:
        return "---"
    return f"{value:.2f}"


def format_value_with_star(value: Optional[float], metric_type: str, rank: Optional[int]) -> str:
    """
    Format a numeric value for LaTeX with optional Pareto star.
    
    Args:
        value: The numeric value
        metric_type: Type of metric ("chrf", "termacc", etc.)
        rank: Pareto rank (1 for gold star, 2 for silver star, None for no star)
    
    Returns:
        LaTeX formatted string with value and optional star
    """
    formatted = format_value(value, metric_type)
    if formatted == "---":
        return formatted
    
    if rank == 1:
        # Gold star: use a more visible gold color and larger size (1.5x)
        # Use \rlap to overlay without taking space, \hspace to shift right, \raisebox to shift up
        # Use \scalebox{1.5} to make star 1.5x bigger
        return f"{formatted}\\rlap{{\\hspace{{0.15em}}\\raisebox{{0.3ex}}{{\\textcolor{{rgb,1:red,0.85;green,0.65;blue,0.13}}{{\\scalebox{{1.5}}{{$\\star$}}}}}}}}"
    elif rank == 2:
        # Silver star: same positioning but gray, larger size (1.5x)
        return f"{formatted}\\rlap{{\\hspace{{0.15em}}\\raisebox{{0.3ex}}{{\\textcolor{{gray!60}}{{\\scalebox{{1.5}}{{$\\star$}}}}}}}}"
    else:
        return formatted


def get_metricx_score(
    dataset: str,
    lang_pair: str,
    workflow_dir: str,
    model: str,
    metrics_base_dir: Path = Path("metrics/results")
) -> Optional[float]:
    """
    Read MetricX score from metrics.json file.
    
    Args:
        dataset: Dataset name (e.g., "dolfin", "wmt25")
        lang_pair: Language pair (e.g., "en_de", "en-zht")
        workflow_dir: Workflow directory name (e.g., "IRB", "IRB.term")
        model: Model name (e.g., "gpt-4-1")
        metrics_base_dir: Base directory for metrics results
    
    Returns:
        Average MetricX score if found, None otherwise
    """
    # Try both underscore and dash formats for language pairs
    # First try the original format
    metrics_file = metrics_base_dir / dataset / lang_pair / workflow_dir / model / "metrics.json"
    
    if not metrics_file.exists():
        # Try with normalized format (underscore to dash or vice versa)
        # For DOLFIN: try converting en_de -> en-de
        # For WMT25: try converting en-zht -> en_zht (though WMT25 uses dashes)
        if '_' in lang_pair:
            # Convert underscore to dash
            normalized_lang_pair = lang_pair.replace('_', '-')
        else:
            # Convert dash to underscore
            normalized_lang_pair = lang_pair.replace('-', '_')
        
        if normalized_lang_pair != lang_pair:
            metrics_file = metrics_base_dir / dataset / normalized_lang_pair / workflow_dir / model / "metrics.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        metricx_data = metrics_data.get("metricx", {})
        if metricx_data:
            avg_metricx = metricx_data.get("avg_metricx")
            if avg_metricx is not None:
                return float(avg_metricx)
    except (json.JSONDecodeError, IOError, ValueError, KeyError):
        # Skip if file doesn't exist, is invalid JSON, or missing data
        pass
    
    return None


def collect_data_by_workflow_model(
    reports_by_dataset_lang: Dict,
    reports_by_dataset_lang_term: Dict,
    metrics_base_dir: Path = Path("metrics/results")
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]]:
    """
    Organize data by workflow -> model -> dataset -> lang_pair -> metric -> value.
    
    Returns:
        {
            workflow_acronym: {
                model: {
                    "dolfin" or "wmt25_term": {
                        lang_pair: {
                            "chrf": value,
                            "termacc": value (only for wmt25_term),
                            "cost": value
                        }
                    }
                }
            }
        }
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # Process non-term reports (DOLFIN only - we skip WMT25 non-term per requirements)
    for (dataset, lang_pair), reports in reports_by_dataset_lang.items():
        if dataset != "dolfin":
            continue
        
        if lang_pair not in DOLFIN_LANG_PAIRS:
            continue
        
        for report in reports:
            workflow_name = report.get("workflow", "")
            workflow = get_workflow_acronym(workflow_name)
            model = report["model"]
            
            if workflow not in WORKFLOW_ORDER or model not in MODEL_MARKERS:
                continue
            
            # Exclude gpt-5
            if model == "gpt-5":
                continue
            
            data[workflow][model]["dolfin"][lang_pair]["chrf"] = report.get("chrf")
            # Calculate cost for this lang pair
            tokens_input = report.get("tokens_input", 0)
            tokens_output = report.get("tokens_output", 0)
            cost = calculate_cost(tokens_input, tokens_output, model, use_batch=False)
            data[workflow][model]["dolfin"][lang_pair]["cost"] = cost
            
            # Try to read MetricX score from metrics.json
            workflow_dir = workflow  # For DOLFIN, no .term suffix
            metricx_score = get_metricx_score(dataset, lang_pair, workflow_dir, model, metrics_base_dir)
            if metricx_score is not None:
                data[workflow][model]["dolfin"][lang_pair]["metricx"] = metricx_score
    
    # Process term reports (WMT25 only)
    for (dataset, lang_pair), reports in reports_by_dataset_lang_term.items():
        if dataset != "wmt25":
            continue
        
        if lang_pair not in WMT25_LANG_PAIRS:
            continue
        
        for report in reports:
            workflow_name = report.get("workflow", "")
            workflow = get_workflow_acronym(workflow_name)
            model = report["model"]
            
            if workflow not in WORKFLOW_ORDER or model not in MODEL_MARKERS:
                continue
            
            # Exclude gpt-5
            if model == "gpt-5":
                continue
            
            # Store both chrF and TermAcc for term workflows
            data[workflow][model]["wmt25_term"][lang_pair]["chrf"] = report.get("chrf")
            data[workflow][model]["wmt25_term"][lang_pair]["termacc"] = report.get("term_acc")
            # Calculate cost for this lang pair
            tokens_input = report.get("tokens_input", 0)
            tokens_output = report.get("tokens_output", 0)
            cost = calculate_cost(tokens_input, tokens_output, model, use_batch=False)
            data[workflow][model]["wmt25_term"][lang_pair]["cost"] = cost
            
            # Try to read MetricX score from metrics.json
            # For WMT25 term workflows, workflow_dir has .term suffix
            workflow_dir = f"{workflow}.term"
            metricx_score = get_metricx_score(dataset, lang_pair, workflow_dir, model, metrics_base_dir)
            if metricx_score is not None:
                data[workflow][model]["wmt25_term"][lang_pair]["metricx"] = metricx_score
    
    return data


def compute_averages(
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    dataset: str,
    metric: str,
    lang_pairs: List[str]
) -> Dict[Tuple[str, str], Optional[float]]:
    """Compute averages for each workflow-model combination."""
    averages = {}
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            key = (workflow, model)
            values = []
            
            for lang_pair in lang_pairs:
                value = data.get(workflow, {}).get(model, {}).get(dataset, {}).get(lang_pair, {}).get(metric)
                if value is not None:
                    values.append(value)
            
            if values:
                averages[key] = sum(values) / len(values)
            else:
                averages[key] = None
    
    return averages


def compute_total_costs(
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    dataset: str,
    lang_pairs: List[str]
) -> Dict[Tuple[str, str], Optional[float]]:
    """Compute total costs (sum) for each workflow-model combination across all language pairs."""
    totals = {}
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            key = (workflow, model)
            total_cost = 0.0
            has_data = False
            
            for lang_pair in lang_pairs:
                cost = data.get(workflow, {}).get(model, {}).get(dataset, {}).get(lang_pair, {}).get("cost")
                if cost is not None:
                    total_cost += cost
                    has_data = True
            
            if has_data:
                totals[key] = total_cost
            else:
                totals[key] = None
    
    return totals


def generate_latex_table_dolfin(data: Dict, output_path: Path) -> None:
    """Generate LaTeX table for DOLFIN dataset."""
    
    # Compute averages first
    dolfin_chrf_avg = compute_averages(data, "dolfin", "chrf", DOLFIN_LANG_PAIRS)
    dolfin_metricx_avg = compute_averages(data, "dolfin", "metricx", DOLFIN_LANG_PAIRS)
    
    # Compute min/max for color coding from Avg values (since we color the Avg column)
    chrf_values = [val for val in dolfin_chrf_avg.values() if val is not None]
    chrf_min = min(chrf_values) if chrf_values else 0
    chrf_max = max(chrf_values) if chrf_values else 100
    
    metricx_values = [val for val in dolfin_metricx_avg.values() if val is not None]
    metricx_min = min(metricx_values) if metricx_values else 0
    metricx_max = max(metricx_values) if metricx_values else 100
    
    # Compute total costs (sum across all lang pairs)
    dolfin_total_costs = compute_total_costs(data, "dolfin", DOLFIN_LANG_PAIRS)
    # Get min/max from total costs (not individual lang pair costs)
    cost_values = [cost for cost in dolfin_total_costs.values() if cost is not None]
    cost_min = min(cost_values) if cost_values else 0
    cost_max = max(cost_values) if cost_values else 1000
    
    # Compute Pareto ranks for chrF++ (cost vs avg_chrf)
    # Collect all (workflow, model) pairs with data
    chrf_costs = []
    chrf_values_list = []
    chrf_keys = []  # List of (workflow, model) tuples in same order
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            key = (workflow, model)
            avg_chrf = dolfin_chrf_avg.get(key)
            total_cost = dolfin_total_costs.get(key)
            if avg_chrf is not None and total_cost is not None:
                chrf_costs.append(total_cost)
                chrf_values_list.append(avg_chrf)
                chrf_keys.append(key)
    
    # Compute Pareto ranks with percentile-based quality threshold
    # For chrF++, exclude points below 75th percentile (keep top 25% quality)
    pareto_ranks_chrf = {}
    if chrf_costs and chrf_values_list:
        # Calculate 75th percentile threshold (keep systems above this)
        min_value = np.percentile(chrf_values_list, 75) if len(chrf_values_list) > 1 else None
        ranks = compute_pareto_ranks(chrf_costs, chrf_values_list, min_value=min_value)
        # Map ranks back to (workflow, model) keys
        for rank, indices in ranks.items():
            for idx in indices:
                pareto_ranks_chrf[chrf_keys[idx]] = rank
    
    # Start LaTeX table
    lines = []
    lines.append("% This table requires the following packages in your LaTeX preamble:")
    lines.append("% \\usepackage[table]{xcolor}  % For \\cellcolor command")
    lines.append("% \\usepackage{booktabs}        % For \\toprule, \\midrule, \\bottomrule, \\cmidrule")
    lines.append("% \\usepackage{multirow}        % For \\multirow command")
    lines.append("% \\usepackage{graphicx}        % For \\scalebox command (for star sizing)")
    lines.append("%")
    lines.append("% Color definitions for Pareto ranks:")
    lines.append("% \\definecolor{rank1}{RGB}{255,215,0}  % Gold for rank 1")
    lines.append("% \\definecolor{rank2}{RGB}{192,192,192}  % Silver for rank 2")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")  # Reduce column spacing
    lines.append("\\resizebox{\\textwidth}{!}{")  # Scale to fit textwidth
    # Columns: System (2: workflow + model) + chrF++ (5: Avg + 4 lang pairs) + MetricX-24 (5: Avg + 4 lang pairs) + Cost (1: Total) = 13
    lines.append("\\begin{tabular}{ll" + "c" * 11 + "}")
    lines.append("\\toprule")
    
    # First header row: chrF++ | MetricX-24 | Cost ($)
    header1 = "\\multirow{2}{*}{\\textbf{System}} & \\multirow{2}{*}{} & "
    header1 += "\\multicolumn{5}{c}{\\textbf{chrF++} $\\uparrow$} & "
    header1 += "\\multicolumn{5}{c}{\\textbf{MetricX-24} $\\downarrow$} & "
    header1 += "\\multicolumn{1}{c}{\\textbf{Cost (\\$)}} \\\\"
    lines.append(header1)
    lines.append("\\cmidrule(lr){3-7} \\cmidrule(lr){8-12} \\cmidrule(lr){13-13}")
    
    # Second header row: language pairs and Total
    header2 = "& & \\textbf{Avg} & En-De & En-Es & En-Fr & En-It & \\textbf{Avg} & En-De & En-Es & En-Fr & En-It & \\textbf{Total} \\\\"
    lines.append(header2)
    lines.append("\\midrule")
    
    # Data rows
    # Get models sorted by cost (most expensive first)
    sorted_models = get_models_sorted_by_cost()
    
    for workflow in WORKFLOW_ORDER:
        workflow_display = WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)
        
        # Workflow header row (if not first)
        if workflow != WORKFLOW_ORDER[0]:
            lines.append("\\midrule")
        
        # All models are shown for all workflows
        models_to_show = sorted_models
        
        # Add rows for all models (sorted by cost, most expensive first), even if they don't have data
        for i, model in enumerate(models_to_show):
            model_display = MODEL_DISPLAY_NAMES.get(model, model)
            
            row_parts = []
            
            # First column: workflow name (multirow)
            if i == 0:
                row_parts.append(f"\\multirow{{{len(models_to_show)}}}{{*}}{{{workflow_display}}}")
            else:
                row_parts.append("")
            
            # Second column: model name
            row_parts.append(model_display)
            
            # chrF++ columns (5 columns: Avg, En-De, En-Es, En-Fr, En-It)
            avg_val = dolfin_chrf_avg.get((workflow, model))
            color = get_color_for_value(avg_val, chrf_min, chrf_max, "chrf") if avg_val is not None else ""
            rank = pareto_ranks_chrf.get((workflow, model))
            formatted_with_star = format_value_with_star(avg_val, "chrf", rank)
            row_parts.append(f"{color}{formatted_with_star}")
            
            for lang_pair in DOLFIN_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("dolfin", {}).get(lang_pair, {}).get("chrf")
                row_parts.append(format_value(val, "chrf"))
            
            # MetricX-24 columns (5 columns: Avg, En-De, En-Es, En-Fr, En-It)
            avg_val = dolfin_metricx_avg.get((workflow, model))
            color = get_color_for_value(avg_val, metricx_min, metricx_max, "metricx") if avg_val is not None else ""
            formatted = format_value(avg_val, "metricx")
            row_parts.append(f"{color}{formatted}")
            
            for lang_pair in DOLFIN_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("dolfin", {}).get(lang_pair, {}).get("metricx")
                row_parts.append(format_value(val, "metricx"))
            
            # Cost column (Total) - inverse color scale (lower is better)
            # Use logarithmic normalization for better distribution across wide cost range
            total_cost = dolfin_total_costs.get((workflow, model))
            cost_color = ""
            if total_cost is not None and cost_min != cost_max and cost_min > 0:
                # Logarithmic normalization
                log_cost = math.log(total_cost)
                log_min = math.log(cost_min)
                log_max = math.log(cost_max)
                normalized = (log_cost - log_min) / (log_max - log_min)
                # Inverse: high cost = red, low cost = green
                if normalized >= 0.7:
                    cost_color = "\\cellcolor{red!25}"
                elif normalized >= 0.5:
                    cost_color = "\\cellcolor{red!10}"
                elif normalized >= 0.3:
                    cost_color = "\\cellcolor{green!10}"
                else:
                    cost_color = "\\cellcolor{green!25}"
            row_parts.append(f"{cost_color}{format_cost(total_cost)}")
            
            # Join row parts
            row = " & ".join(row_parts) + " \\\\"
            lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")  # End resizebox
    lines.append("\\caption{Main results for DOLFIN dataset. Top-performing systems are labelled as rank 1\\textcolor{rank1}{$\\bigstar$} and rank 2\\textcolor{rank2}{$\\bigstar$} according to Pareto optimality (only systems above the 75th percentile in performance are considered for Pareto ranking).}")
    lines.append("\\label{tab:main_results_dolfin}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"Generated LaTeX table: {output_path}")


def generate_latex_table_wmt25(data: Dict, output_path: Path) -> None:
    """Generate LaTeX table for WMT25+Term dataset."""
    
    # Compute averages first
    wmt25_chrf_avg = compute_averages(data, "wmt25_term", "chrf", WMT25_LANG_PAIRS)
    wmt25_metricx_avg = compute_averages(data, "wmt25_term", "metricx", WMT25_LANG_PAIRS)
    wmt25_termacc_avg = compute_averages(data, "wmt25_term", "termacc", WMT25_LANG_PAIRS)
    
    # Compute min/max for color coding from Avg values (since we color the Avg column)
    chrf_values = [val for val in wmt25_chrf_avg.values() if val is not None]
    chrf_min = min(chrf_values) if chrf_values else 0
    chrf_max = max(chrf_values) if chrf_values else 100
    
    metricx_values = [val for val in wmt25_metricx_avg.values() if val is not None]
    metricx_min = min(metricx_values) if metricx_values else 0
    metricx_max = max(metricx_values) if metricx_values else 100
    
    termacc_values = [val for val in wmt25_termacc_avg.values() if val is not None]
    termacc_min = min(termacc_values) if termacc_values else 0
    termacc_max = max(termacc_values) if termacc_values else 1
    
    # Compute total costs (sum across all lang pairs)
    wmt25_total_costs = compute_total_costs(data, "wmt25_term", WMT25_LANG_PAIRS)
    # Get min/max from total costs (not individual lang pair costs)
    cost_values = [cost for cost in wmt25_total_costs.values() if cost is not None]
    cost_min = min(cost_values) if cost_values else 0
    cost_max = max(cost_values) if cost_values else 1000
    
    # Compute Pareto ranks for chrF++ (cost vs avg_chrf)
    chrf_costs = []
    chrf_values_list = []
    chrf_keys = []
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            key = (workflow, model)
            avg_chrf = wmt25_chrf_avg.get(key)
            total_cost = wmt25_total_costs.get(key)
            if avg_chrf is not None and total_cost is not None:
                chrf_costs.append(total_cost)
                chrf_values_list.append(avg_chrf)
                chrf_keys.append(key)
    
    pareto_ranks_chrf = {}
    if chrf_costs and chrf_values_list:
        # For chrF++, exclude points below 75th percentile (keep top 25% quality)
        min_value = np.percentile(chrf_values_list, 75) if len(chrf_values_list) > 1 else None
        ranks = compute_pareto_ranks(chrf_costs, chrf_values_list, min_value=min_value)
        for rank, indices in ranks.items():
            for idx in indices:
                pareto_ranks_chrf[chrf_keys[idx]] = rank
    
    # Compute Pareto ranks for TermAcc (cost vs avg_termacc)
    termacc_costs = []
    termacc_values_list = []
    termacc_keys = []
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            key = (workflow, model)
            avg_termacc = wmt25_termacc_avg.get(key)
            total_cost = wmt25_total_costs.get(key)
            if avg_termacc is not None and total_cost is not None:
                termacc_costs.append(total_cost)
                termacc_values_list.append(avg_termacc)
                termacc_keys.append(key)
    
    pareto_ranks_termacc = {}
    if termacc_costs and termacc_values_list:
        # For TermAcc, exclude points below 75th percentile (keep top 25% quality)
        min_value = np.percentile(termacc_values_list, 75) if len(termacc_values_list) > 1 else None
        ranks = compute_pareto_ranks(termacc_costs, termacc_values_list, min_value=min_value)
        for rank, indices in ranks.items():
            for idx in indices:
                pareto_ranks_termacc[termacc_keys[idx]] = rank
    
    # Start LaTeX table
    lines = []
    lines.append("% This table requires the following packages in your LaTeX preamble:")
    lines.append("% \\usepackage[table]{xcolor}  % For \\cellcolor command")
    lines.append("% \\usepackage{booktabs}        % For \\toprule, \\midrule, \\bottomrule, \\cmidrule")
    lines.append("% \\usepackage{multirow}        % For \\multirow command")
    lines.append("% \\usepackage{graphicx}        % For \\scalebox command (for star sizing)")
    lines.append("%")
    lines.append("% Color definitions for Pareto ranks:")
    lines.append("% \\definecolor{rank1}{RGB}{255,215,0}  % Gold for rank 1")
    lines.append("% \\definecolor{rank2}{RGB}{192,192,192}  % Silver for rank 2")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")  # Reduce column spacing
    lines.append("\\resizebox{\\textwidth}{!}{")  # Scale to fit textwidth
    # Columns: System (2: workflow + model) + chrF++ (3: Avg + 2 lang pairs) + MetricX-24 (3: Avg + 2 lang pairs) + TermAcc (3: Avg + 2 lang pairs) + Cost (1: Total) = 12
    lines.append("\\begin{tabular}{ll" + "c" * 10 + "}")
    lines.append("\\toprule")
    
    # First header row: chrF++ | MetricX-24 | TermAcc | Cost ($)
    header1 = "\\multirow{2}{*}{\\textbf{System}} & \\multirow{2}{*}{} & "
    header1 += "\\multicolumn{3}{c}{\\textbf{chrF++} $\\uparrow$} & "
    header1 += "\\multicolumn{3}{c}{\\textbf{MetricX-24} $\\downarrow$} & "
    header1 += "\\multicolumn{3}{c}{\\textbf{TermAcc} $\\uparrow$} & "
    header1 += "\\multicolumn{1}{c}{\\textbf{Cost (\\$)}} \\\\"
    lines.append(header1)
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-12}")
    
    # Second header row: language pairs and Total
    header2 = "& & \\textbf{Avg} & En-Zht & Zht-En & \\textbf{Avg} & En-Zht & Zht-En & \\textbf{Avg} & En-Zht & Zht-En & \\textbf{Total} \\\\"
    lines.append(header2)
    lines.append("\\midrule")
    
    # Data rows
    # Get models sorted by cost (most expensive first)
    sorted_models = get_models_sorted_by_cost()
    
    for workflow in WORKFLOW_ORDER:
        workflow_display = WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)
        
        # Workflow header row (if not first)
        if workflow != WORKFLOW_ORDER[0]:
            lines.append("\\midrule")
        
        # All models are shown for all workflows
        models_to_show = sorted_models
        
        # Add rows for all models (sorted by cost, most expensive first), even if they don't have data
        for i, model in enumerate(models_to_show):
            model_display = MODEL_DISPLAY_NAMES.get(model, model)
            
            row_parts = []
            
            # First column: workflow name (multirow)
            if i == 0:
                row_parts.append(f"\\multirow{{{len(models_to_show)}}}{{*}}{{{workflow_display}}}")
            else:
                row_parts.append("")
            
            # Second column: model name
            row_parts.append(model_display)
            
            # chrF++ columns (3 columns: Avg, En-Zht, Zht-En)
            avg_val = wmt25_chrf_avg.get((workflow, model))
            color = get_color_for_value(avg_val, chrf_min, chrf_max, "chrf") if avg_val is not None else ""
            rank = pareto_ranks_chrf.get((workflow, model))
            formatted_with_star = format_value_with_star(avg_val, "chrf", rank)
            row_parts.append(f"{color}{formatted_with_star}")
            
            for lang_pair in WMT25_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("chrf")
                row_parts.append(format_value(val, "chrf"))
            
            # MetricX-24 columns (3 columns: Avg, En-Zht, Zht-En)
            avg_val = wmt25_metricx_avg.get((workflow, model))
            color = get_color_for_value(avg_val, metricx_min, metricx_max, "metricx") if avg_val is not None else ""
            formatted = format_value(avg_val, "metricx")
            row_parts.append(f"{color}{formatted}")
            
            for lang_pair in WMT25_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("metricx")
                row_parts.append(format_value(val, "metricx"))
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("metricx")
                row_parts.append(format_value(val, "metricx"))
            
            # TermAcc columns (3 columns: Avg, En-Zht, Zht-En)
            avg_val = wmt25_termacc_avg.get((workflow, model))
            color = get_color_for_value(avg_val, termacc_min, termacc_max, "termacc") if avg_val is not None else ""
            rank = pareto_ranks_termacc.get((workflow, model))
            formatted_with_star = format_value_with_star(avg_val, "termacc", rank)
            row_parts.append(f"{color}{formatted_with_star}")
            
            for lang_pair in WMT25_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("termacc")
                row_parts.append(format_value(val, "termacc"))
            
            # Cost column (Total) - inverse color scale (lower is better)
            # Use logarithmic normalization for better distribution across wide cost range
            total_cost = wmt25_total_costs.get((workflow, model))
            cost_color = ""
            if total_cost is not None and cost_min != cost_max and cost_min > 0:
                # Logarithmic normalization
                log_cost = math.log(total_cost)
                log_min = math.log(cost_min)
                log_max = math.log(cost_max)
                normalized = (log_cost - log_min) / (log_max - log_min)
                # Inverse: high cost = red, low cost = green
                if normalized >= 0.7:
                    cost_color = "\\cellcolor{red!25}"
                elif normalized >= 0.5:
                    cost_color = "\\cellcolor{red!10}"
                elif normalized >= 0.3:
                    cost_color = "\\cellcolor{green!10}"
                else:
                    cost_color = "\\cellcolor{green!25}"
            row_parts.append(f"{cost_color}{format_cost(total_cost)}")
            
            # Join row parts
            row = " & ".join(row_parts) + " \\\\"
            lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")  # End resizebox
    lines.append("\\caption{Main results for WMT25+Term dataset. Top-performing systems are labelled as rank 1\\textcolor{rank1}{$\\bigstar$} and rank 2\\textcolor{rank2}{$\\bigstar$} according to Pareto optimality (only systems above the 75th percentile in performance are considered for Pareto ranking).}")
    lines.append("\\label{tab:main_results_wmt25}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"Generated LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table with experimental results"
    )
    parser.add_argument(
        "--outputs_dirs",
        type=str,
        nargs='+',
        default=["outputs_qwen3", "outputs"],
        help="Paths to outputs directories containing report.json files (can specify multiple)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="report/tables",
        help="Output directory for LaTeX table files"
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
    
    # Collect all reports
    print("Collecting reports...")
    reports_by_dataset_lang, reports_by_dataset_lang_term, _ = collect_reports(existing_dirs)
    
    # Organize data
    print("Organizing data...")
    data = collect_data_by_workflow_model(reports_by_dataset_lang, reports_by_dataset_lang_term)
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_table_dolfin(data, output_dir / "main_results_dolfin.tex")
    generate_latex_table_wmt25(data, output_dir / "main_results_wmt25.tex")
    
    return 0


if __name__ == "__main__":
    exit(main())

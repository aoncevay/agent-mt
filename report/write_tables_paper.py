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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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

# Model display names
MODEL_DISPLAY_NAMES = {
    "qwen3-32b": "Qwen 3 32B",
    "qwen3-235b": "Qwen 3 235B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "claude-sonnet-4": "Claude Sonnet 4",
    "gpt-4-1": "GPT-4.1"
}

# Workflow order and display names
WORKFLOW_ORDER = ["ZS", "IRB", "MaMT", "SbS_chat", "MAATS_multi", "ADT", "DeLTA"]
WORKFLOW_DISPLAY_NAMES = {
    "ZS": "Zero-shot",
    "IRB": "IRB",
    "MaMT": "MaMT",
    "SbS_chat": "Step-by-step",
    "MAATS_multi": "MAATS",
    "ADT": "ADT",
    "DeLTA": "DeLTA",
}

# Model order (from larger to smaller)
MODEL_ORDER = ["gpt-4-1", "claude-sonnet-4", "qwen3-235b", "gpt-oss-120b", "qwen3-32b", "gpt-oss-20b"]

# Language pairs
DOLFIN_LANG_PAIRS = ["en_de", "en_es", "en_fr", "en_it"]
WMT25_LANG_PAIRS = ["en-zht", "zht-en"]

# LaTeX color commands (using xcolor package)
def get_color_for_value(value: float, min_val: float, max_val: float, metric_type: str = "chrf") -> str:
    """Get LaTeX color command for a value (green for high, red for low)."""
    if value is None or min_val == max_val:
        return ""
    
    # Normalize value to 0-1 range
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
    else:
        # For chrF++ and COMET, high is good (green), low is bad (red)
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
    # Use 2 decimals for TermAcc, 1 for others
    if metric_type == "termacc":
        return f"{value:.2f}"
    return f"{value:.1f}"


def collect_data_by_workflow_model(
    reports_by_dataset_lang: Dict,
    reports_by_dataset_lang_term: Dict
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
                            "termacc": value (only for wmt25_term)
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
            
            data[workflow][model]["dolfin"][lang_pair]["chrf"] = report.get("chrf")
    
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
            
            # Store both chrF and TermAcc for term workflows
            data[workflow][model]["wmt25_term"][lang_pair]["chrf"] = report.get("chrf")
            data[workflow][model]["wmt25_term"][lang_pair]["termacc"] = report.get("term_acc")
    
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


def generate_latex_table_dolfin(data: Dict, output_path: Path) -> None:
    """Generate LaTeX table for DOLFIN dataset."""
    
    # Compute min/max for color coding (chrF++ and COMET use same scale)
    chrf_values = []
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            for lang_pair in DOLFIN_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("dolfin", {}).get(lang_pair, {}).get("chrf")
                if val is not None:
                    chrf_values.append(val)
    
    chrf_min = min(chrf_values) if chrf_values else 0
    chrf_max = max(chrf_values) if chrf_values else 100
    
    # Compute averages
    dolfin_chrf_avg = compute_averages(data, "dolfin", "chrf", DOLFIN_LANG_PAIRS)
    
    # Start LaTeX table
    lines = []
    lines.append("% This table requires the following packages in your LaTeX preamble:")
    lines.append("% \\usepackage[table]{xcolor}  % For \\cellcolor command")
    lines.append("% \\usepackage{booktabs}        % For \\toprule, \\midrule, \\bottomrule, \\cmidrule")
    lines.append("% \\usepackage{multirow}        % For \\multirow command")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    # Columns: System (2: workflow + model) + chrF++ (5: Avg + 4 lang pairs) + COMET (5: Avg + 4 lang pairs) = 12
    lines.append("\\begin{tabular}{ll" + "c" * 10 + "}")
    lines.append("\\toprule")
    
    # First header row: chrF++ | COMET
    header1 = "\\multirow{2}{*}{\\textbf{System}} & \\multirow{2}{*}{} & "
    header1 += "\\multicolumn{5}{c}{\\textbf{chrF++}} & "
    header1 += "\\multicolumn{5}{c}{\\textbf{COMET}} \\\\"
    lines.append(header1)
    lines.append("\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}")
    
    # Second header row: language pairs
    header2 = "& & \\textbf{Avg} & En-De & En-Es & En-Fr & En-It & \\textbf{Avg} & En-De & En-Es & En-Fr & En-It \\\\"
    lines.append(header2)
    lines.append("\\midrule")
    
    # Data rows
    for workflow in WORKFLOW_ORDER:
        workflow_display = WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)
        
        # Workflow header row (if not first)
        if workflow != WORKFLOW_ORDER[0]:
            lines.append("\\midrule")
        
        # Add rows for all models (in order), even if they don't have data
        for i, model in enumerate(MODEL_ORDER):
            model_display = MODEL_DISPLAY_NAMES.get(model, model)
            
            row_parts = []
            
            # First column: workflow name (multirow)
            if i == 0:
                row_parts.append(f"\\multirow{{{len(MODEL_ORDER)}}}{{*}}{{{workflow_display}}}")
            else:
                row_parts.append("")
            
            # Second column: model name
            row_parts.append(model_display)
            
            # chrF++ columns (5 columns: Avg, En-De, En-Es, En-Fr, En-It)
            avg_val = dolfin_chrf_avg.get((workflow, model))
            color = get_color_for_value(avg_val, chrf_min, chrf_max, "chrf") if avg_val is not None else ""
            row_parts.append(f"{color}{format_value(avg_val, 'chrf')}")
            
            for lang_pair in DOLFIN_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("dolfin", {}).get(lang_pair, {}).get("chrf")
                row_parts.append(format_value(val, "chrf"))
            
            # COMET columns (5 columns: Avg, En-De, En-Es, En-Fr, En-It) - empty for now
            row_parts.append("---")  # Avg column (empty)
            for _ in DOLFIN_LANG_PAIRS:
                row_parts.append("---")  # Individual lang pair columns (empty)
            
            # Join row parts
            row = " & ".join(row_parts) + " \\\\"
            lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Main results for DOLFIN dataset.}")
    lines.append("\\label{tab:main_results_dolfin}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"Generated LaTeX table: {output_path}")


def generate_latex_table_wmt25(data: Dict, output_path: Path) -> None:
    """Generate LaTeX table for WMT25+Term dataset."""
    
    # Compute min/max for color coding (separate for chrF++ and TermAcc)
    chrf_values = []
    termacc_values = []
    
    for workflow in WORKFLOW_ORDER:
        for model in MODEL_ORDER:
            for lang_pair in WMT25_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("chrf")
                if val is not None:
                    chrf_values.append(val)
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("termacc")
                if val is not None:
                    termacc_values.append(val)
    
    chrf_min = min(chrf_values) if chrf_values else 0
    chrf_max = max(chrf_values) if chrf_values else 100
    termacc_min = min(termacc_values) if termacc_values else 0
    termacc_max = max(termacc_values) if termacc_values else 100
    
    # Compute averages
    wmt25_chrf_avg = compute_averages(data, "wmt25_term", "chrf", WMT25_LANG_PAIRS)
    wmt25_termacc_avg = compute_averages(data, "wmt25_term", "termacc", WMT25_LANG_PAIRS)
    
    # Start LaTeX table
    lines = []
    lines.append("% This table requires the following packages in your LaTeX preamble:")
    lines.append("% \\usepackage[table]{xcolor}  % For \\cellcolor command")
    lines.append("% \\usepackage{booktabs}        % For \\toprule, \\midrule, \\bottomrule, \\cmidrule")
    lines.append("% \\usepackage{multirow}        % For \\multirow command")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    # Columns: System (2: workflow + model) + chrF++ (3: Avg + 2 lang pairs) + COMET (3: Avg + 2 lang pairs) + TermAcc (3: Avg + 2 lang pairs) = 11
    lines.append("\\begin{tabular}{ll" + "c" * 9 + "}")
    lines.append("\\toprule")
    
    # First header row: chrF++ | COMET | TermAcc
    header1 = "\\multirow{2}{*}{\\textbf{System}} & \\multirow{2}{*}{} & "
    header1 += "\\multicolumn{3}{c}{\\textbf{chrF++}} & "
    header1 += "\\multicolumn{3}{c}{\\textbf{COMET}} & "
    header1 += "\\multicolumn{3}{c}{\\textbf{TermAcc}} \\\\"
    lines.append(header1)
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}")
    
    # Second header row: language pairs
    header2 = "& & \\textbf{Avg} & En-Zht & Zht-En & \\textbf{Avg} & En-Zht & Zht-En & \\textbf{Avg} & En-Zht & Zht-En \\\\"
    lines.append(header2)
    lines.append("\\midrule")
    
    # Data rows
    for workflow in WORKFLOW_ORDER:
        workflow_display = WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)
        
        # Workflow header row (if not first)
        if workflow != WORKFLOW_ORDER[0]:
            lines.append("\\midrule")
        
        # Add rows for all models (in order), even if they don't have data
        for i, model in enumerate(MODEL_ORDER):
            model_display = MODEL_DISPLAY_NAMES.get(model, model)
            
            row_parts = []
            
            # First column: workflow name (multirow)
            if i == 0:
                row_parts.append(f"\\multirow{{{len(MODEL_ORDER)}}}{{*}}{{{workflow_display}}}")
            else:
                row_parts.append("")
            
            # Second column: model name
            row_parts.append(model_display)
            
            # chrF++ columns (3 columns: Avg, En-Zht, Zht-En)
            avg_val = wmt25_chrf_avg.get((workflow, model))
            color = get_color_for_value(avg_val, chrf_min, chrf_max, "chrf") if avg_val is not None else ""
            row_parts.append(f"{color}{format_value(avg_val, 'chrf')}")
            
            for lang_pair in WMT25_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("chrf")
                row_parts.append(format_value(val, "chrf"))
            
            # COMET columns (3 columns: Avg, En-Zht, Zht-En) - empty for now
            row_parts.append("---")  # Avg column (empty)
            for _ in WMT25_LANG_PAIRS:
                row_parts.append("---")  # Individual lang pair columns (empty)
            
            # TermAcc columns (3 columns: Avg, En-Zht, Zht-En)
            avg_val = wmt25_termacc_avg.get((workflow, model))
            color = get_color_for_value(avg_val, termacc_min, termacc_max, "termacc") if avg_val is not None else ""
            row_parts.append(f"{color}{format_value(avg_val, 'termacc')}")
            
            for lang_pair in WMT25_LANG_PAIRS:
                val = data.get(workflow, {}).get(model, {}).get("wmt25_term", {}).get(lang_pair, {}).get("termacc")
                row_parts.append(format_value(val, "termacc"))
            
            # Join row parts
            row = " & ".join(row_parts) + " \\\\"
            lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Main results for WMT25+Term dataset.}")
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
        default=["zhijin/agent-mt-main/outputs", "outputs"],
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

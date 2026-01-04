#!/usr/bin/env python3
"""
Print LaTeX table with model costs (pricing per 1,000 tokens).

Usage:
    python report/cost_table.py

Note: The generated LaTeX table requires the following packages in your .tex file preamble:
    \\usepackage{booktabs}        % For \\toprule, \\midrule, \\bottomrule
"""

from pathlib import Path

# Import from plot script
import importlib.util
plot_script_path = Path(__file__).parent / "plot_main_results.py"
spec = importlib.util.spec_from_file_location("plot_main_results", plot_script_path)
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)

MODEL_PRICING_STANDARD = plot_module.MODEL_PRICING_STANDARD
MODEL_DISPLAY_NAMES = plot_module.MODEL_DISPLAY_NAMES
get_model_base_cost = plot_module.get_model_base_cost

# Model order (from most expensive to least expensive)
MODEL_ORDER = ["gpt-4-1", "qwen3-235b", "gpt-oss-120b", "qwen3-32b", "gpt-4-1-nano"]


def format_price(price: float) -> str:
    """Format price with appropriate precision, escaping $ for LaTeX."""
    if price >= 0.01:
        return f"\\${price:.3f}"
    elif price >= 0.001:
        return f"\\${price:.4f}"
    else:
        return f"\\${price:.5f}"


def get_models_sorted_by_cost(use_batch: bool = False) -> list:
    """Get models sorted by base API cost (most expensive first)."""
    models_with_costs = []
    for model in MODEL_ORDER:
        base_cost = get_model_base_cost(model, use_batch=use_batch)
        if base_cost is not None:
            models_with_costs.append((model, base_cost))
    
    # Sort by cost (most expensive first)
    models_with_costs.sort(key=lambda x: x[1], reverse=True)
    return [model for model, _ in models_with_costs]


def print_latex_table():
    """Print LaTeX table with model costs (standard tier, used in experiments)."""
    print("\n% Model Pricing Table (Standard Tier)")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{lrr}")
    print("\\toprule")
    print("Model & Input & Output \\\\")
    print("      & \\multicolumn{2}{c}{(\\$/1k tokens)} \\\\")
    print("\\midrule")
    
    sorted_models = get_models_sorted_by_cost(use_batch=False)
    
    for model in sorted_models:
        if model not in MODEL_PRICING_STANDARD:
            continue
        
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        prices = MODEL_PRICING_STANDARD[model]
        input_price = prices["input"]
        output_price = prices["output"]
        
        print(f"{display_name} & "
              f"{format_price(input_price)} & "
              f"{format_price(output_price)} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}%")
    print("\\caption{Model pricing per 1,000 tokens (standard tier, used in experiments).}")
    print("\\end{table}")


def main():
    """Main function to print cost tables."""
    print("=" * 80)
    print("Model Cost Tables")
    print("=" * 80)
    
    # Print standard tier table (used in experiments)
    print_latex_table()


if __name__ == "__main__":
    main()

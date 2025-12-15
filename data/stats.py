"""
Print concise dataset statistics for WMT and DOLFIN for the paper
MD and Latex format
Per language pair: #documents #tokens_src #tokens_tgt avg_tokens_src avg_tokens_tgt
"""

import json
from pathlib import Path
from typing import Dict
import tiktoken

# Initialize tokenizer (using cl100k_base encoding, same as GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in a text string, excluding tokens that are:
    - Numbers only
    - Markdown table separators (|)
    - Tokens without at least one alphabet character
    """
    if not isinstance(text, str) or not text:
        return 0
    
    # Encode text to tokens
    tokens = tokenizer.encode(text)
    
    # Count only tokens that contain at least one alphabet character
    count = 0
    for token_id in tokens:
        # Decode token to check its content
        try:
            token_str = tokenizer.decode([token_id])
            # Check if token contains at least one alphabet character (a-z or A-Z)
            if any(c.isalpha() for c in token_str):
                count += 1
        except Exception:
            # If decoding fails, skip this token
            continue
    
    return count


def count_sentences(text: str) -> int:
    """
    Count sentences by counting non-empty lines.
    This is a proxy for sentence counting appropriate for document-level statistics.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    
    # Split by newlines and count non-empty lines
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    return len(non_empty_lines)


def get_wmt25_stats(data_dir: Path) -> Dict[str, Dict[str, any]]:
    """Calculate statistics for WMT25 dataset."""
    stats = {}
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    # Group by language pair direction
    en_zht_stats = {"documents": 0, "tokens_src": 0, "tokens_tgt": 0, "sentences_src": 0, "sentences_tgt": 0}
    zht_en_stats = {"documents": 0, "tokens_src": 0, "tokens_tgt": 0, "sentences_src": 0, "sentences_tgt": 0}
    
    for year in years:
        file_path = data_dir / f"full_data_{year}.jsonl"
        if not file_path.exists():
            continue
        
        # Determine direction: odd=en->zht, even=zht->en
        if year % 2 == 1:  # Odd year: en->zht
            current_stats = en_zht_stats
            src_lang = "en"
            tgt_lang = "zh"  # Files use "zh" key
        else:  # Even year: zht->en
            current_stats = zht_en_stats
            src_lang = "zh"
            tgt_lang = "en"
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                sample = json.loads(line)
                source_text = sample.get(src_lang, "")
                target_text = sample.get(tgt_lang, "")
                
                if source_text and target_text:
                    current_stats["documents"] += 1
                    current_stats["tokens_src"] += count_tokens(source_text)
                    current_stats["tokens_tgt"] += count_tokens(target_text)
                    current_stats["sentences_src"] += count_sentences(source_text)
                    current_stats["sentences_tgt"] += count_sentences(target_text)
    
    # Calculate averages and store
    if en_zht_stats["documents"] > 0:
        stats["en-zht"] = {
            "documents": en_zht_stats["documents"],
            "tokens_src": en_zht_stats["tokens_src"],
            "tokens_tgt": en_zht_stats["tokens_tgt"],
            "sentences_src": en_zht_stats["sentences_src"],
            "sentences_tgt": en_zht_stats["sentences_tgt"],
            "avg_tokens_src": en_zht_stats["tokens_src"] / en_zht_stats["documents"],
            "avg_tokens_tgt": en_zht_stats["tokens_tgt"] / en_zht_stats["documents"],
        }
    
    if zht_en_stats["documents"] > 0:
        stats["zht-en"] = {
            "documents": zht_en_stats["documents"],
            "tokens_src": zht_en_stats["tokens_src"],
            "tokens_tgt": zht_en_stats["tokens_tgt"],
            "sentences_src": zht_en_stats["sentences_src"],
            "sentences_tgt": zht_en_stats["sentences_tgt"],
            "avg_tokens_src": zht_en_stats["tokens_src"] / zht_en_stats["documents"],
            "avg_tokens_tgt": zht_en_stats["tokens_tgt"] / zht_en_stats["documents"],
        }
    
    return stats


def get_dolfin_stats(data_dir: Path) -> Dict[str, Dict[str, any]]:
    """Calculate statistics for DOLFIN dataset."""
    stats = {}
    lang_pairs = ["en_de", "en_es", "en_fr", "en_it"]
    
    for lang_pair in lang_pairs:
        file_path = data_dir / f"dolfin_test_{lang_pair}.jsonl"
        if not file_path.exists():
            continue
        
        source_lang, target_lang = lang_pair.split("_")
        pair_stats = {"documents": 0, "tokens_src": 0, "tokens_tgt": 0, "sentences_src": 0, "sentences_tgt": 0}
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                sample = json.loads(line)
                source_text = sample.get(source_lang, "")
                target_text = sample.get(target_lang, "")
                
                if source_text and target_text:
                    pair_stats["documents"] += 1
                    pair_stats["tokens_src"] += count_tokens(source_text)
                    pair_stats["tokens_tgt"] += count_tokens(target_text)
                    pair_stats["sentences_src"] += count_sentences(source_text)
                    pair_stats["sentences_tgt"] += count_sentences(target_text)
        
        if pair_stats["documents"] > 0:
            stats[lang_pair] = {
                "documents": pair_stats["documents"],
                "tokens_src": pair_stats["tokens_src"],
                "tokens_tgt": pair_stats["tokens_tgt"],
                "sentences_src": pair_stats["sentences_src"],
                "sentences_tgt": pair_stats["sentences_tgt"],
                "avg_tokens_src": pair_stats["tokens_src"] / pair_stats["documents"],
                "avg_tokens_tgt": pair_stats["tokens_tgt"] / pair_stats["documents"],
            }
    
    return stats


def format_number(num: float, decimals: int = 1) -> str:
    """Format number with appropriate precision."""
    if decimals == 0:
        return f"{int(num):,}"
    return f"{num:,.{decimals}f}"


def print_markdown_table(stats: Dict[str, Dict[str, any]], dataset_name: str):
    """Print statistics in Markdown table format."""
    print(f"\n## {dataset_name.upper()} Dataset Statistics\n")
    print("| Language Pair | Documents | Tokens (Src) | Tokens (Tgt) | Avg Tokens (Src) | Avg Tokens (Tgt) | Sentences (Src) | Sentences (Tgt) |")
    print("|---------------|-----------|--------------|--------------|------------------|------------------|-----------------|-----------------|")
    
    for lang_pair, stat in sorted(stats.items()):
        # Replace underscores with hyphens for display
        lang_pair_display = lang_pair.replace("_", "-")
        print(f"| {lang_pair_display} | "
              f"{format_number(stat['documents'], 0)} | "
              f"{format_number(stat['tokens_src'], 0)} | "
              f"{format_number(stat['tokens_tgt'], 0)} | "
              f"{format_number(stat['avg_tokens_src'], 1)} | "
              f"{format_number(stat['avg_tokens_tgt'], 1)} | "
              f"{format_number(stat['sentences_src'], 0)} | "
              f"{format_number(stat['sentences_tgt'], 0)} |")


def print_latex_table(stats: Dict[str, Dict[str, any]], dataset_name: str):
    """Print statistics in LaTeX table format."""
    print(f"\n% {dataset_name.upper()} Dataset Statistics")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lrrrrrrr}")
    print("\\toprule")
    print("Lang Pair & Docs & Tokens (Src) & Tokens (Tgt) & Avg Tok (Src) & Avg Tok (Tgt) & Sents (Src) & Sents (Tgt) \\\\")
    print("\\midrule")
    
    for lang_pair, stat in sorted(stats.items()):
        # Replace underscores with hyphens for LaTeX
        lang_pair_display = lang_pair.replace("_", "-")
        print(f"{lang_pair_display} & "
              f"{format_number(stat['documents'], 0)} & "
              f"{format_number(stat['tokens_src'], 0)} & "
              f"{format_number(stat['tokens_tgt'], 0)} & "
              f"{format_number(stat['avg_tokens_src'], 1)} & "
              f"{format_number(stat['avg_tokens_tgt'], 1)} & "
              f"{format_number(stat['sentences_src'], 0)} & "
              f"{format_number(stat['sentences_tgt'], 0)} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Statistics for {dataset_name.upper()} dataset.}}")
    print("\\end{table}")


def print_combined_latex_table(wmt25_stats: Dict[str, Dict[str, any]], dolfin_stats: Dict[str, Dict[str, any]]):
    """Print combined LaTeX table with both datasets."""
    print("\n% Combined Dataset Statistics")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{llrrrrrrr}")
    print("\\toprule")
    print("Dataset & Lang Pair & Docs & Tokens (Src) & Tokens (Tgt) & Avg Tok (Src) & Avg Tok (Tgt) & Sents (Src) & Sents (Tgt) \\\\")
    print("\\midrule")
    
    # Add WMT25 entries
    for lang_pair, stat in sorted(wmt25_stats.items()):
        print(f"WMT25-Term & {lang_pair} & "
              f"{format_number(stat['documents'], 0)} & "
              f"{format_number(stat['tokens_src'], 0)} & "
              f"{format_number(stat['tokens_tgt'], 0)} & "
              f"{format_number(stat['avg_tokens_src'], 1)} & "
              f"{format_number(stat['avg_tokens_tgt'], 1)} & "
              f"{format_number(stat['sentences_src'], 0)} & "
              f"{format_number(stat['sentences_tgt'], 0)} \\\\")
    
    # Add DOLFIN entries
    for lang_pair, stat in sorted(dolfin_stats.items()):
        # Replace underscores with hyphens for DOLFIN
        lang_pair_display = lang_pair.replace("_", "-")
        print(f"DOLFIN & {lang_pair_display} & "
              f"{format_number(stat['documents'], 0)} & "
              f"{format_number(stat['tokens_src'], 0)} & "
              f"{format_number(stat['tokens_tgt'], 0)} & "
              f"{format_number(stat['avg_tokens_src'], 1)} & "
              f"{format_number(stat['avg_tokens_tgt'], 1)} & "
              f"{format_number(stat['sentences_src'], 0)} & "
              f"{format_number(stat['sentences_tgt'], 0)} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Statistics for WMT25-Term and DOLFIN datasets.}")
    print("\\end{table}")


def main():
    # Get data directory (assuming script is in data/ directory)
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir / "raw"
    
    # Calculate statistics
    print("=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    
    # WMT25 statistics
    wmt25_stats = {}
    wmt25_dir = raw_data_dir / "wmt25-terminology-track2"
    if wmt25_dir.exists():
        wmt25_stats = get_wmt25_stats(wmt25_dir)
        print_markdown_table(wmt25_stats, "WMT25")
        print_latex_table(wmt25_stats, "WMT25")
    else:
        print(f"\nWarning: WMT25 directory not found at {wmt25_dir}")
    
    # DOLFIN statistics
    dolfin_stats = {}
    dolfin_dir = raw_data_dir / "dolfin"
    if dolfin_dir.exists():
        dolfin_stats = get_dolfin_stats(dolfin_dir)
        print_markdown_table(dolfin_stats, "DOLFIN")
        print_latex_table(dolfin_stats, "DOLFIN")
    else:
        print(f"\nWarning: DOLFIN directory not found at {dolfin_dir}")
    
    # Combined LaTeX table
    if wmt25_stats and dolfin_stats:
        print_combined_latex_table(wmt25_stats, dolfin_stats)


if __name__ == "__main__":
    main()

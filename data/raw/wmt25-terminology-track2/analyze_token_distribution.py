"""
Script to analyze the token distribution of English segments in WMT25 terminology track2 test sets.
This helps decide on appropriate token length thresholds for filtering.
"""

import json
import tiktoken
import numpy as np
from pathlib import Path
from collections import defaultdict

# Initialize tokenizer for English (using cl100k_base encoding, same as GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    if not isinstance(text, str) or not text:
        return 0
    return len(tokenizer.encode(text))

def analyze_token_distribution(token_counts: list, label: str = "") -> None:
    """Analyze and print token length distribution statistics."""
    if not token_counts:
        print(f"No data for {label}")
        return
        
    token_counts = np.array(token_counts)
    
    print("\n" + "=" * 70)
    if label:
        print(f"TOKEN LENGTH DISTRIBUTION: {label}")
    else:
        print("TOKEN LENGTH DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Total samples: {len(token_counts):,}")
    print(f"Mean: {np.mean(token_counts):.2f} tokens")
    print(f"Median: {np.median(token_counts):.2f} tokens")
    print(f"Std Dev: {np.std(token_counts):.2f} tokens")
    print(f"Min: {np.min(token_counts)} tokens")
    print(f"Max: {np.max(token_counts)} tokens")
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th percentile: {np.percentile(token_counts, p):8.2f} tokens")
    
    # Histogram bins
    print(f"\nDistribution by ranges:")
    ranges = [
        (0, 50, "< 50"),
        (50, 100, "50-100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 500, "300-500"),
        (500, 1000, "500-1000"),
        (1000, 2000, "1000-2000"),
        (2000, 3000, "2000-3000"),
        (3000, float('inf'), "> 3000")
    ]
    for min_val, max_val, label_range in ranges:
        count = np.sum((token_counts >= min_val) & (token_counts < max_val))
        pct = (count / len(token_counts)) * 100 if len(token_counts) > 0 else 0
        print(f"  {label_range:12s}: {count:8,} ({pct:5.2f}%)")
    print("=" * 70 + "\n")

def main():
    # Paths
    script_dir = Path(__file__).parent
    jsonl_files = sorted(script_dir.glob("full_data_*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {script_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to analyze")
    
    all_token_counts = []
    year_token_counts = defaultdict(list)
    
    # Process each file
    for jsonl_file in jsonl_files:
        year = jsonl_file.stem.replace("full_data_", "")
        print(f"\nProcessing {jsonl_file.name}...")
        
        token_counts_year = []
        total_lines = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'en' in data and isinstance(data['en'], str):
                        token_count = count_tokens(data['en'])
                        token_counts_year.append(token_count)
                        all_token_counts.append(token_count)
                        total_lines += 1
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error at line {line_num}: {e}")
                    continue
        
        year_token_counts[year] = token_counts_year
        print(f"  Processed {total_lines:,} English segments from {year}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS (All Years Combined)")
    print("=" * 70)
    analyze_token_distribution(all_token_counts, "All Years")
    
    # Per-year statistics
    print("\n" + "=" * 70)
    print("PER-YEAR STATISTICS")
    print("=" * 70)
    for year in sorted(year_token_counts.keys()):
        analyze_token_distribution(year_token_counts[year], f"Year {year}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Year':<8} {'Count':<10} {'Mean':<10} {'Median':<10} {'P25':<10} {'P75':<10} {'P95':<10}")
    print("-" * 70)
    for year in sorted(year_token_counts.keys()):
        counts = np.array(year_token_counts[year])
        if len(counts) > 0:
            print(f"{year:<8} {len(counts):<10,} {np.mean(counts):<10.1f} {np.median(counts):<10.1f} "
                  f"{np.percentile(counts, 25):<10.1f} {np.percentile(counts, 75):<10.1f} "
                  f"{np.percentile(counts, 95):<10.1f}")
    print("=" * 70)
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR TOKEN THRESHOLDS")
    print("=" * 70)
    overall_counts = np.array(all_token_counts)
    if len(overall_counts) > 0:
        p5 = np.percentile(overall_counts, 5)
        p10 = np.percentile(overall_counts, 10)
        p25 = np.percentile(overall_counts, 25)
        median = np.median(overall_counts)
        
        print(f"\nBased on overall distribution:")
        print(f"  - 5th percentile: {p5:.1f} tokens (removes shortest 5%)")
        print(f"  - 10th percentile: {p10:.1f} tokens (removes shortest 10%)")
        print(f"  - 25th percentile: {p25:.1f} tokens (removes shortest 25%)")
        print(f"  - Median: {median:.1f} tokens")
        print(f"\nSuggested lower thresholds:")
        print(f"  - Conservative (keep 95%): {max(50, int(p5))} tokens")
        print(f"  - Moderate (keep 90%): {max(100, int(p10))} tokens")
        print(f"  - Aggressive (keep 75%): {max(200, int(p25))} tokens")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()


"""
Python script to process the Dolfin dataset. 
The parquet file "test-00000-of-00001.parquet" contains the following columns:

source_text: the segment in source language
target_text: the segment in target language
src_lang: source language
trg_lang: target language
sub_domain: document type referring to the sub-domain of finance
date: date of publication of the document
comet_slide: Comet-kiwi-slide score
Annotation: annotations of context-sensitive phenomena (obtained by CTXPRO and Llama-3-70b)
id: unique id of the segment

Filtering steps:
1. We want to extract the src_lang == "en" only, and any tgt_lang.
2. We want to keep the entries where source_text length (English) is over 500 tokens, but less than 3000 tokens.
   Token counting only includes tokens that contain at least one alphabet character (excludes numbers-only and markdown separators).
   For the second point, it would be good to analyse the distribution of the token length of the source text. 
3. Let's filter out entries that are markdown tables. We can look if the source_text starts with | and ends with |.
4. Filter out entries that contain markdown table rows (contains \n| or |\n) to remove embedded tables in longer texts. 

Save the result data in a jsonl file per language pair (e.g. dolfin_test_en_es.jsonl)), using the structure: 
{{src_lang}: {source_text}, {tgt_lang}: {target_text}, sub_domain: ..., id: ..., annotation: ...}}
"""

import json
import pandas as pd
from pathlib import Path
import tiktoken
import numpy as np

# Initialize tokenizer for English (using cl100k_base encoding, same as GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """
    Count tokens in a text string, excluding tokens that are:
    - Numbers only
    - Markdown table separators (|)
    - Tokens without at least one alphabet character
    """
    if pd.isna(text) or not isinstance(text, str):
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

def is_markdown_table(text: str) -> bool:
    """Check if text is a markdown table (starts with | and ends with |)."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    text_stripped = text.strip()
    return text_stripped.startswith('|') and text_stripped.endswith('|')

def contains_markdown_table_rows(text: str) -> bool:
    """Check if text contains markdown table rows (contains \n| or |\n)."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    return '\n|' in text or '|\n' in text

def analyze_token_distribution(token_counts: list) -> None:
    """Analyze and print token length distribution statistics."""
    token_counts = np.array(token_counts)
    
    print("\n" + "=" * 60)
    print("TOKEN LENGTH DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(token_counts)}")
    print(f"Mean: {np.mean(token_counts):.2f} tokens")
    print(f"Median: {np.median(token_counts):.2f} tokens")
    print(f"Std Dev: {np.std(token_counts):.2f} tokens")
    print(f"Min: {np.min(token_counts)} tokens")
    print(f"Max: {np.max(token_counts)} tokens")
    print("\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(token_counts, p):.2f} tokens")
    
    # Histogram bins
    print("\nDistribution by ranges:")
    ranges = [
        (0, 50, "< 50"),
        (50, 100, "50-100"),
        (100, 200, "100-200"),
        (200, 500, "200-500"),
        (500, 1000, "500-1000"),
        (1000, 2000, "1000-2000"),
        (2000, float('inf'), "> 2000")
    ]
    for min_val, max_val, label in ranges:
        count = np.sum((token_counts >= min_val) & (token_counts < max_val))
        pct = (count / len(token_counts)) * 100
        print(f"  {label:12s}: {count:6d} ({pct:5.2f}%)")
    print("=" * 60 + "\n")

def main():
    # Paths
    script_dir = Path(__file__).parent
    parquet_file = script_dir / "test-00000-of-00001.parquet"
    output_dir = script_dir
    
    # Read parquet file
    print(f"Reading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Total rows: {len(df)}")
    
    # Filter: src_lang == "en"
    print("\nFiltering for src_lang == 'en'...")
    df_en = df[df['src_lang'] == 'en'].copy()
    print(f"Rows with src_lang == 'en': {len(df_en)}")
    
    # Count tokens for all English source texts (before filtering)
    print("\nCounting tokens for source texts...")
    df_en['source_token_count'] = df_en['source_text'].apply(count_tokens)
    
    # Analyze token distribution before filtering
    print("\nToken distribution BEFORE filtering:")
    analyze_token_distribution(df_en['source_token_count'].tolist())
    
    # Filter: 500 < tokens < 3000
    print("\nFiltering for 500 < token_count < 3000 (alphabet tokens only)...")
    df_filtered = df_en[
        (df_en['source_token_count'] > 500) & 
        (df_en['source_token_count'] < 3000)
    ].copy()
    print(f"Rows after token length filtering: {len(df_filtered)}")
    
    # Filter: Remove markdown tables (starts with | and ends with |)
    print("\nFiltering out markdown tables (starts with | and ends with |)...")
    markdown_table_mask = df_filtered['source_text'].apply(is_markdown_table)
    markdown_table_count = markdown_table_mask.sum()
    print(f"Found {markdown_table_count} markdown table entries to remove")
    
    df_filtered = df_filtered[~markdown_table_mask].copy()
    print(f"Rows after markdown table filtering: {len(df_filtered)}")
    
    # Filter: Remove entries containing markdown table rows (contains \n| or |\n)
    print("\nFiltering out entries containing markdown table rows (contains \\n| or |\\n)...")
    markdown_table_rows_mask = df_filtered['source_text'].apply(contains_markdown_table_rows)
    markdown_table_rows_count = markdown_table_rows_mask.sum()
    print(f"Found {markdown_table_rows_count} entries with markdown table rows to remove")
    
    df_filtered = df_filtered[~markdown_table_rows_mask].copy()
    print(f"Rows after markdown table rows filtering: {len(df_filtered)}")
    
    # Analyze token distribution after filtering
    print("\nToken distribution AFTER filtering (500-3000 alphabet tokens, no markdown tables):")
    analyze_token_distribution(df_filtered['source_token_count'].tolist())
    
    # Group by language pair and save
    print("\nGrouping by language pairs...")
    lang_pairs = df_filtered.groupby('trg_lang')
    
    for trg_lang, group_df in lang_pairs:
        output_file = output_dir / f"dolfin_test_en_{trg_lang}.jsonl"
        
        records = []
        for _, row in group_df.iterrows():
            record = {
                row['src_lang']: row['source_text'],
                row['trg_lang']: row['target_text'],
                'sub_domain': row.get('sub_domain', None),
                'id': row.get('id', None)
                #'annotation': row.get('Annotation', None)
            }
            # Remove None values
            record = {k: v for k, v in record.items() if v is not None}
            records.append(record)
        
        # Save to jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(records)} records to {output_file.name}")
    
    print(f"\n✓ Processing complete! Generated {len(lang_pairs)} language pair files.")

if __name__ == "__main__":
    main()

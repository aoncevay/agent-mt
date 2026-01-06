"""
Simplified MetricX Analysis

This script performs a simplified MetricX evaluation by:
1. Splitting source, target, and reference into chunks based on token count (512 tokens per chunk)
2. Computing MetricX scores for each chunk pair
3. Averaging scores across all chunks

This approach skips paragraph alignment and works directly with chunked text.

Usage:
    python metrics/my_metricx_analysis.py --dataset wmt25 --workflow IRB --model gpt-4-1
    python metrics/my_metricx_analysis.py --dataset dolfin --target_language es
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tiktoken

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.evaluate_experiments import (
    find_experiments,
    get_latest_agent_output,
    load_sample_data,
    get_metrics_file_path,
    BASE_DATA_DIR,
    get_workflows_and_models_to_process
)
from metrics.metricx_evaluator import compute_metricx_scores


def _log_with_time(msg: str):
    """Log message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: "gpt-4")
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: approximate using character count (rough estimate: 1 token ≈ 4 chars)
        return len(text) // 4


def split_into_chunks(text: str, num_chunks: int) -> List[str]:
    """
    Split text into approximately equal chunks by lines.
    
    Args:
        text: Text to split
        num_chunks: Number of chunks to create
    
    Returns:
        List of text chunks
    """
    if not text or num_chunks <= 1:
        return [text] if text else []
    
    lines = text.split('\n')
    if not lines:
        return [text]
    
    # Filter out empty lines
    lines = [line for line in lines if line.strip()]
    if not lines:
        return [text]
    
    # Calculate lines per chunk
    lines_per_chunk = max(1, len(lines) // num_chunks)
    
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = '\n'.join(lines[i:i + lines_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    
    # Ensure we have exactly num_chunks (merge last chunks if needed)
    while len(chunks) > num_chunks:
        # Merge last two chunks
        chunks[-2] = chunks[-2] + '\n' + chunks[-1]
        chunks.pop()
    
    # If we have fewer chunks, split the last one
    while len(chunks) < num_chunks and len(chunks) > 0:
        last_chunk = chunks[-1]
        lines = last_chunk.split('\n')
        if len(lines) > 1:
            mid = len(lines) // 2
            chunks[-1] = '\n'.join(lines[:mid])
            chunks.append('\n'.join(lines[mid:]))
        else:
            break
    
    return chunks if chunks else [text]


def process_experiment_simplified(
    output_dir: Path,
    report_data: Dict[str, Any],
    dataset: str,
    metric_model=None,
    tokenizer=None
) -> Optional[Dict[str, Any]]:
    """
    Process a single experiment with simplified MetricX analysis.
    
    Args:
        output_dir: Output directory for the experiment
        report_data: Report data from report.json
        dataset: Dataset name
        metric_model: Pre-loaded MetricX model (optional)
        tokenizer: Pre-loaded MetricX tokenizer (optional)
    
    Returns:
        Dictionary with MetricX results or None if processing failed
    """
    from metrics.metricx_evaluator import _find_metricx_model, _find_mt5_tokenizer
    
    # Check if experiment is complete
    if report_data.get('total_samples', 0) != report_data.get('successful_samples', 0):
        print(f"  ⚠ Skipping incomplete experiment: {report_data.get('successful_samples', 0)}/{report_data.get('total_samples', 0)} samples")
        return None
    
    workflow_name = report_data.get('workflow', '')
    model_name = report_data.get('model', '')
    lang_pair = report_data.get('lang_pair', '')
    
    print(f"\n{'='*80}")
    print(f"Processing: {workflow_name} + {model_name}")
    print(f"Dataset: {dataset}, Lang pair: {lang_pair}")
    print(f"{'='*80}")
    
    # Load MetricX model if not provided
    if metric_model is None or tokenizer is None:
        _log_with_time("  Loading MetricX model...")
        metricx_model_path = _find_metricx_model()
        mt5_tokenizer_path = _find_mt5_tokenizer()
        
        if not metricx_model_path or not mt5_tokenizer_path:
            print(f"  ✗ Could not find MetricX model or tokenizer")
            return None
        
        # Load model and tokenizer
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        _log_with_time("    Loading MetricX model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(mt5_tokenizer_path, local_files_only=True)
        metric_model = AutoModelForSeq2SeqLM.from_pretrained(metricx_model_path, local_files_only=True)
        
        if torch.cuda.is_available():
            metric_model = metric_model.to('cuda')
            _log_with_time("    Moved MetricX model to GPU")
    
    # Load samples from report
    samples = report_data.get('samples', [])
    if not samples:
        print(f"  ✗ No samples found in report")
        return None
    
    # Collect source/reference/translation for all samples
    sample_data = []
    for sample in samples:
        if sample.get('error'):
            continue
        
        sample_idx = sample.get('sample_idx') or sample.get('index')
        sample_id = sample.get('sample_id') or sample.get('id') or str(sample_idx)
        
        # Load source/reference from data files
        data_tuple = load_sample_data(dataset, lang_pair, sample_idx, sample_id, BASE_DATA_DIR)
        if not data_tuple:
            print(f"  ⚠ Warning: Could not load data for sample {sample_idx}, skipping")
            continue
        
        source_text, reference_text, _ = data_tuple
        
        # Load translation from agent output file
        translation = get_latest_agent_output(output_dir, sample_id, sample_idx)
        if not translation:
            print(f"  ⚠ Warning: Could not load translation for sample {sample_idx}, skipping")
            continue
        
        sample_data.append({
            'sample_idx': sample_idx,
            'sample_id': sample_id,
            'source': source_text,
            'target': translation,
            'reference': reference_text
        })
    
    if not sample_data:
        print(f"  ✗ No valid samples found")
        return None
    
    print(f"  Processing {len(sample_data)} samples...")
    
    # Process each sample: split into chunks and compute MetricX
    all_scores = []
    per_sample_results = []
    
    for sample_info in sample_data:
        sample_idx = sample_info['sample_idx']
        sample_id = sample_info['sample_id']
        src_text = sample_info['source']
        tgt_text = sample_info['target']
        ref_text = sample_info['reference']
        
        # Count tokens in source (use this to determine number of chunks)
        # MetricX input format: "source: {src} target: {tgt} reference: {ref}"
        # MetricX accepts up to 512 tokens total for the entire input
        # Strategy: Divide 512 tokens among src, tgt, ref (roughly equal)
        # Each part gets ~512/3 ≈ 170 tokens (leaving ~2 tokens for prompt overhead)
        # The corresponding tgt/ref chunks will be compared even if they differ in length
        # (That's the translation's problem - mismatches will be penalized)
        src_tokens = count_tokens(src_text)
        tokens_per_src_chunk = 512 // 3  # ~170 tokens per part (src, tgt, ref), total ~510 tokens + overhead
        num_chunks = max(1, (src_tokens + tokens_per_src_chunk - 1) // tokens_per_src_chunk)  # Round up
        
        # Split into chunks
        src_chunks = split_into_chunks(src_text, num_chunks)
        tgt_chunks = split_into_chunks(tgt_text, num_chunks)  # Same number of chunks
        ref_chunks = split_into_chunks(ref_text, num_chunks)  # Same number of chunks
        
        # Ensure all have the same number of chunks
        max_chunks = max(len(src_chunks), len(tgt_chunks), len(ref_chunks))
        while len(src_chunks) < max_chunks:
            src_chunks.append("")
        while len(tgt_chunks) < max_chunks:
            tgt_chunks.append("")
        while len(ref_chunks) < max_chunks:
            ref_chunks.append("")
        
        # Prepare segments for MetricX (format: (src, tgt, ref))
        segments = list(zip(src_chunks, tgt_chunks, ref_chunks))
        
        # Filter out segments where source is empty (can't evaluate without source)
        # If source is empty but target exists, that's over-translation (will be penalized with score 25.0)
        # But we still need source to evaluate, so skip segments without source
        valid_segments = [(s, t, r) for s, t, r in segments if s.strip()]
        
        if not valid_segments:
            print(f"    ⚠ Sample {sample_idx}: No valid segments with source text after chunking")
            continue
        
        # Compute MetricX scores for this sample
        try:
            metric_result = compute_metricx_scores(
                valid_segments,
                metricx_model_path=None,  # Not needed if metric_model is provided
                mt5_tokenizer_path=None,  # Not needed if tokenizer is provided
                metric_model=metric_model,
                tokenizer=tokenizer
            )
            
            sample_scores = metric_result.get('scores', [])
            if sample_scores:
                all_scores.extend(sample_scores)
                per_sample_results.append({
                    'sample_idx': sample_idx,
                    'sample_id': sample_id,
                    'num_chunks': len(valid_segments),
                    'src_tokens': src_tokens,
                    'avg_metricx': sum(sample_scores) / len(sample_scores) if sample_scores else None,
                    'min_metricx': min(sample_scores) if sample_scores else None,
                    'max_metricx': max(sample_scores) if sample_scores else None,
                    'scores': sample_scores
                })
        except Exception as e:
            print(f"    ⚠ Sample {sample_idx}: Error computing MetricX: {e}")
            continue
    
    if not all_scores:
        print(f"  ✗ No MetricX scores computed")
        return None
    
    # Aggregate results
    results = {
        'workflow': workflow_name,
        'model': model_name,
        'dataset': dataset,
        'lang_pair': lang_pair,
        'num_samples': len(per_sample_results),
        'num_segments': len(all_scores),
        'avg_metricx': sum(all_scores) / len(all_scores),
        'min_metricx': min(all_scores),
        'max_metricx': max(all_scores),
        'per_sample': per_sample_results
    }
    
    return results


def save_results(results: Dict[str, Any], output_dir: Path, metrics_base_dir: Path = Path("metrics/results")):
    """
    Save results to my_metricx.json file.
    
    Args:
        results: Results dictionary
        output_dir: Original output directory
        metrics_base_dir: Base directory for metrics results
    """
    # Get metrics file path (same structure as evaluate_experiments.py)
    metrics_file = get_metrics_file_path(output_dir, metrics_base_dir)
    
    # Change filename from metrics.json to my_metricx.json
    metrics_file = metrics_file.parent / "my_metricx.json"
    
    # Ensure directory exists
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved results to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Simplified MetricX Analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wmt25", "dolfin"],
        help="Dataset name: 'wmt25' or 'dolfin'"
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default=None,
        help="Target language filter (e.g., 'zht' for WMT25, 'es' for DOLFIN)"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Filter by workflow (e.g., 'IRB', 'MaMT'). For WMT25, will automatically append '.term'"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model (e.g., 'gpt-4-1', 'qwen3-32b')"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        nargs='+',
        default=["outputs", "outputs_qwen3"],
        help="Output directories to scan (default: outputs outputs_qwen3)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already have my_metricx.json"
    )
    
    args = parser.parse_args()
    
    # For WMT25, append .term to workflow if not already present
    if args.dataset == "wmt25" and args.workflow and not args.workflow.endswith('.term'):
        args.workflow = args.workflow + '.term'
    
    # Get workflows and models to process
    workflows, models = get_workflows_and_models_to_process()
    
    # Find experiments
    outputs_dirs = [Path(d) for d in args.outputs_dir]
    experiments = find_experiments(
        dataset=args.dataset,
        target_language=args.target_language,
        workflow=args.workflow,
        model=args.model,
        outputs_dirs=outputs_dirs
    )
    
    print(f"\nFound {len(experiments)} experiments to process")
    
    # Load MetricX model once
    _log_with_time("Loading MetricX model...")
    from metrics.metricx_evaluator import _find_metricx_model, _find_mt5_tokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    
    metricx_model_path = _find_metricx_model()
    mt5_tokenizer_path = _find_mt5_tokenizer()
    
    if not metricx_model_path or not mt5_tokenizer_path:
        print("✗ Could not find MetricX model or tokenizer")
        return
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(mt5_tokenizer_path),
        local_files_only=True
    )
    metric_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(metricx_model_path),
        local_files_only=True
    )
    
    if torch.cuda.is_available():
        metric_model = metric_model.to('cuda')
        print("  ✓ Loaded MetricX model on GPU")
    else:
        print("  ✓ Loaded MetricX model on CPU")
    
    # Process experiments
    processed = 0
    skipped_resume = 0
    failed = 0
    
    for output_dir, report_data in experiments:
        # Check if we should skip due to --resume
        if args.resume:
            metrics_file = get_metrics_file_path(output_dir, Path("metrics/results"))
            metrics_file = metrics_file.parent / "my_metricx.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        existing = json.load(f)
                    if existing.get('num_samples') and existing.get('avg_metricx') is not None:
                        skipped_resume += 1
                        continue
                except:
                    pass
        
        try:
            results = process_experiment_simplified(
                output_dir,
                report_data,
                args.dataset,
                metric_model=metric_model,
                tokenizer=tokenizer
            )
            
            if results:
                save_results(results, output_dir)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error processing {output_dir}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and processed % 5 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (resume): {skipped_resume}")
    print(f"  Failed: {failed}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


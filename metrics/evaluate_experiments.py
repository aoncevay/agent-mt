#!/usr/bin/env python3
"""
Main orchestration script for document-level metrics evaluation.

Evaluates experiments using term-consistency approach:
1. Split documents into aligned segments (using LaBSE)
2. Align terms (using TermBasedMetric for WMT25-Term)
3. Compute COMET scores per segment
4. Aggregate and save results

Usage:
    python metrics/evaluate_experiments.py --dataset wmt25
    python metrics/evaluate_experiments.py --dataset wmt25 --target_language zht
    python metrics/evaluate_experiments.py --dataset dolfin --target_language es
"""

import argparse
import json
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


def _log_with_time(msg: str):
    """Print message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import from our codebase
from src.data_loaders import get_data_loader, get_available_dolfin_lang_pairs
from src.workflow_acronyms import get_workflow_acronym
from report.write_tables_paper import WORKFLOW_ORDER, MODEL_ORDER

# Import metrics utilities
from metrics.utils import get_latest_agent_output, load_report, parse_system_name_from_path

# Configuration
OUTPUTS_DIRS = [PROJECT_ROOT / "outputs", PROJECT_ROOT / "outputs_qwen3"]
BASE_DATA_DIR = PROJECT_ROOT / "data" / "raw"
METRICS_OUTPUT_DIR = PROJECT_ROOT / "metrics" / "results"


def find_experiments(
    dataset: str,
    target_language: Optional[str] = None,
    workflow: Optional[str] = None,
    model: Optional[str] = None,
    outputs_dirs: List[Path] = None
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Find all experiments matching the dataset and filters.
    
    Args:
        dataset: Dataset name ("wmt25" or "dolfin")
        target_language: Optional target language filter (e.g., "zht", "es")
        workflow: Optional workflow filter (e.g., "IRB", "IRB.term")
        model: Optional model filter (e.g., "gpt-4-1", "qwen3-32b")
        outputs_dirs: List of output directories to scan
    
    Returns:
        List of (output_dir, report_data) tuples
    """
    if outputs_dirs is None:
        outputs_dirs = OUTPUTS_DIRS
    
    experiments = []
    
    for outputs_dir in outputs_dirs:
        if not outputs_dir.exists():
            continue
        
        # Look for report.json files
        for report_file in outputs_dir.rglob("report.json"):
            # Parse path: outputs/{dataset}/{lang_pair}/{workflow_dir}/{model}/report.json
            parts = report_file.parts
            try:
                outputs_idx = parts.index(outputs_dir.name)
                if len(parts) < outputs_idx + 5:
                    continue
                
                dataset_name = parts[outputs_idx + 1]
                lang_pair = parts[outputs_idx + 2]
                
                # Filter by dataset
                if dataset_name != dataset:
                    continue
                
                # Extract workflow_dir and model
                workflow_dir = parts[outputs_idx + 3]
                model_name = parts[outputs_idx + 4] if len(parts) > outputs_idx + 4 else None
                
                # For WMT25, only process .term experiments (terminology experiments)
                if dataset == "wmt25":
                    if not workflow_dir.endswith('.term'):
                        continue  # Skip non-terminology experiments for WMT25
                
                # Filter by workflow if specified
                if workflow:
                    # workflow can be "IRB" or "IRB.term" - check both
                    workflow_base = workflow_dir.replace('.term', '')
                    workflow_filter_base = workflow.replace('.term', '')
                    if workflow_base != workflow_filter_base:
                        continue
                
                # Filter by model if specified
                if model and model_name != model:
                    continue
                
                # Filter by target_language if specified
                if target_language:
                    # For WMT25: lang_pair is "en-zht" or "zht-en"
                    # For DOLFIN: lang_pair is "en_es", "en_de", etc.
                    if dataset == "wmt25":
                        # Check if target_language matches either direction
                        if target_language not in lang_pair:
                            continue
                    elif dataset == "dolfin":
                        # Check if target_language matches the target part
                        if f"_{target_language}" not in lang_pair and not lang_pair.endswith(f"_{target_language}"):
                            continue
                
                # Load report
                report_data = load_report(report_file.parent)
                if not report_data:
                    continue
                
                # Verify experiment is complete
                total = report_data.get('total_samples', 0)
                successful = report_data.get('successful_samples', 0)
                if total == 0 or total != successful:
                    continue  # Skip incomplete experiments
                
                # Add metadata
                report_data['_output_dir'] = report_file.parent
                report_data['_lang_pair'] = lang_pair
                report_data['_dataset'] = dataset_name
                
                experiments.append((report_file.parent, report_data))
                
            except (ValueError, IndexError):
                continue
    
    return experiments


def get_workflows_and_models_to_process() -> Tuple[List[str], List[str]]:
    """
    Get list of workflows and models to process (from write_tables_paper.py).
    
    Returns:
        (workflows, models) - Lists of workflow acronyms and model names
    """
    # Workflows from write_tables_paper.py
    workflows = WORKFLOW_ORDER  # ["ZS", "IRB", "MaMT", "SbS_chat", "MAATS_multi", "ADT", "DeLTA"]
    
    # Models from write_tables_paper.py (excluding GPT-5 and gpt-4-1-mini for non-zero-shot)
    models = [m for m in MODEL_ORDER if m not in ["gpt-5", "gpt-4-1-mini"]]  # These are zero-shot only
    
    return workflows, models


def should_process_experiment(
    report_data: Dict[str, Any],
    workflows: List[str],
    models: List[str]
) -> bool:
    """
    Check if an experiment should be processed based on workflow and model filters.
    
    Args:
        report_data: Report data dictionary
        workflows: List of workflow acronyms to process
        models: List of model names to process
    
    Returns:
        True if experiment should be processed
    """
    workflow_name = report_data.get('workflow', '')
    workflow_acronym = get_workflow_acronym(workflow_name)
    
    model_name = report_data.get('model', '')
    
    # Check if workflow matches
    if workflow_acronym not in workflows:
        return False
    
    # Check if model matches (handle base_model combinations)
    if '+' in model_name:
        # For base_model combinations, check if the main model matches
        main_model = model_name.split('+')[-1]
        if main_model not in models:
            return False
    else:
        if model_name not in models:
            return False
    
    return True


def load_sample_data(
    dataset: str,
    lang_pair: str,
    sample_idx: int,
    sample_id: str,
    data_dir: Path
) -> Optional[Tuple[str, str, Optional[Dict[str, list]]]]:
    """
    Load source text, reference text, and terminology for a sample.
    
    Args:
        dataset: Dataset name
        lang_pair: Language pair
        sample_idx: Sample index
        sample_id: Sample ID
        data_dir: Base data directory
    
    Returns:
        (source_text, reference_text, terminology) or None if not found
    """
    try:
        # Get data loader
        if dataset == "wmt25":
            data_loader = get_data_loader(dataset, data_dir, target_languages=None)
        elif dataset == "dolfin":
            # For DOLFIN, we need to create loader with specific lang_pair
            from src.data_loaders import DOLFINDataLoader
            data_loader = DOLFINDataLoader(data_dir / "dolfin", lang_pair=lang_pair)
        else:
            return None
        
        # Load all samples (we'll find the one we need)
        all_samples = data_loader.load_samples()
        
        # Find the sample by index or ID
        for sample in all_samples:
            sample_idx_in_data = all_samples.index(sample)
            sample_id_in_data = sample.get("id") or sample.get("_id") or str(sample_idx_in_data)
            
            if sample_idx_in_data == sample_idx or str(sample_id_in_data) == str(sample_id):
                # Found the sample
                source_lang, target_lang = data_loader.get_translation_direction(sample)
                source_text, reference_text, terminology = data_loader.extract_texts(
                    sample, source_lang, target_lang
                )
                return source_text, reference_text, terminology
        
        return None
        
    except Exception as e:
        print(f"  ⚠ Warning: Could not load data for sample {sample_idx}: {e}")
        return None


def process_experiment(
    output_dir: Path,
    report_data: Dict[str, Any],
    dataset: str,
    lang_pair: str,
    workflows: List[str],
    models: List[str],
    labse_model=None,  # Optional pre-loaded SentenceTransformer model
    polyfuzz_model=None,  # Optional pre-loaded PolyFuzz model
    metric_model=None,  # Optional pre-loaded MetricX or COMET model
    metric_tokenizer=None,  # Optional pre-loaded tokenizer (for MetricX)
    labse_only: bool = False,  # If True, only do alignment and save to tmp
    metricx_only: bool = False,  # If True, skip TBM, load from tmp if available
    tbm_only: bool = False,  # If True, only compute TBM, load from tmp
    comet_only: bool = False,  # If True, skip TBM computation
    use_metric: str = "metricx"  # "metricx" or "comet"
) -> Optional[Dict[str, Any]]:
    """
    Process a single experiment and compute metrics.
    
    Args:
        output_dir: Output directory for the experiment
        report_data: Report data from report.json
        dataset: Dataset name
        lang_pair: Language pair
        workflows: List of workflows to process
        models: List of models to process
    
    Returns:
        Dictionary with metrics results, or None if experiment should be skipped
    """
    # Note: Filtering by workflows/models is now done in main() before calling process_experiment()
    # This check is kept as a safety guard, but should always pass now
    if not should_process_experiment(report_data, workflows, models):
        return None
    
    import time
    
    def _log_with_time(msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
    
    workflow_name = report_data.get('workflow', '')
    model_name = report_data.get('model', '')
    use_terminology = '.term' in str(output_dir)
    
    experiment_start = time.time()
    print(f"\n{'='*80}")
    print(f"Processing: {workflow_name} + {model_name}")
    print(f"  Dataset: {dataset}, Lang pair: {lang_pair}")
    print(f"  Output dir: {output_dir}")
    _log_with_time("  Starting experiment processing...")
    print(f"{'='*80}")
    
    # Get system name (workflow+model)
    system_name = parse_system_name_from_path(output_dir)
    if not system_name:
        # Fallback: construct from report data
        workflow_acronym = get_workflow_acronym(workflow_name)
        system_name = f"{workflow_acronym}+{model_name}"
    
    # Load samples from report
    samples = report_data.get('samples', [])
    if not samples:
        print(f"  ⚠ No samples found in report")
        return None
    
    print(f"  Found {len(samples)} samples")
    
    # Collect source/reference/translation/terminology for all samples
    sample_data = []
    for sample in samples:
        if sample.get('error'):
            continue  # Skip samples with errors
        
        sample_idx = sample.get('sample_idx', 0)
        sample_id = sample.get('sample_id', str(sample_idx))
        
        # Load source/reference/terminology from data files
        data_tuple = load_sample_data(dataset, lang_pair, sample_idx, sample_id, BASE_DATA_DIR)
        if not data_tuple:
            print(f"  ⚠ Warning: Could not load data for sample {sample_idx}, skipping")
            continue
        
        source_text, reference_text, terminology = data_tuple
        
        # Load translation from agent output file
        translation = get_latest_agent_output(output_dir, sample_id, sample_idx)
        if not translation:
            print(f"  ⚠ Warning: Could not load translation for sample {sample_idx}, skipping")
            continue
        
        sample_data.append({
            'sample_idx': sample_idx,
            'sample_id': sample_id,
            'source_text': source_text,
            'reference_text': reference_text,
            'translation': translation,
            'terminology': terminology
        })
    
    if not sample_data:
        print(f"  ✗ No valid samples to process")
        return None
    
    print(f"  Processing {len(sample_data)} valid samples...")
    
    # Extract source and target language from lang_pair
    if '-' in lang_pair:
        src_lang, tgt_lang = lang_pair.split('-', 1)
    elif '_' in lang_pair:
        src_lang, tgt_lang = lang_pair.split('_', 1)
    else:
        # Default: assume English source
        src_lang = 'en'
        tgt_lang = lang_pair.replace('en', '').replace('-', '').replace('_', '')
    
    # Prepare documents for docpreprocessor
    documents = [(s['source_text'], s['translation']) for s in sample_data]
    references = [s['reference_text'] for s in sample_data]
    
    # Get terminology (if available)
    terminology = None
    if dataset == "wmt25" and sample_data:
        # For WMT25, terminology is per-sample, but we'll use the first one as global
        # (in practice, all samples should have the same terminology)
        terminology = sample_data[0].get('terminology')
    
    # 1. Run docpreprocessor (split + align segments) OR load from tmp
    from metrics.tmp_utils import save_aligned_df, load_aligned_df, has_aligned_df
    import torch
    
    skipped_count = 0  # Track segments skipped due to missing src-ref alignment
    
    # Check if we should load from tmp (metricx_only or tbm_only)
    if (metricx_only or tbm_only) and has_aligned_df(output_dir, dataset, lang_pair):
        _log_with_time(f"  Step 1: Loading aligned segments from tmp file...")
        tmp_result = load_aligned_df(output_dir, dataset, lang_pair)
        if tmp_result:
            aligned_df, sample_data = tmp_result
            print(f"    ✓ Loaded {len(aligned_df)} aligned segments from tmp")
            # Note: skipped_count is not available from tmp files (would need to save it separately)
        else:
            print(f"    ✗ Could not load tmp file, need to run --labse-only first")
            return None
    elif labse_only:
        # Only do alignment and save to tmp
        _log_with_time(f"  Step 1: Splitting and aligning documents (--labse-only mode)...")
        from metrics.docpreprocessor import DocPreprocessor
        
        use_gpu = torch.cuda.is_available()
        if labse_model is None or polyfuzz_model is None:
            from metrics.docpreprocessor import (
                load_labse_model_once,
                load_polyfuzz_model_once,
                load_embeddings_wrapper_once,
                find_labse_model_path
            )
            _log_with_time("  Loading models (fallback - should be provided from main)...")
            labse_model_path = find_labse_model_path()
            if labse_model is None:
                labse_model = load_labse_model_once(labse_model_path, use_gpu=use_gpu)
            if polyfuzz_model is None:
                embeddings = load_embeddings_wrapper_once(labse_model_path, labse_model)
                polyfuzz_model = load_polyfuzz_model_once(labse_model_path, embeddings)
        
        preprocessor = DocPreprocessor(
            src_lang, 
            tgt_lang, 
            labse_model=labse_model,
            polyfuzz_model=polyfuzz_model,
            use_gpu=use_gpu
        )
        aligned_df = preprocessor.process_documents(
            documents,
            references=references,  # Pass references for alignment
            terminology=terminology,
            similarity_threshold=0.4,
            separator='\n\n'
        )
        skipped_count = getattr(preprocessor, 'skipped_segments_count', 0)
        print(f"    ✓ Aligned {len(aligned_df)} segments" + (f" (skipped {skipped_count} segments without src-ref alignment)" if skipped_count > 0 else ""))
        
        # Save to tmp file
        tmp_file = save_aligned_df(aligned_df, output_dir, dataset, lang_pair, sample_data)
        print(f"    ✓ Saved aligned segments to: {tmp_file}")
        
        # Clear GPU memory after LaBSE
        if use_gpu:
            torch.cuda.empty_cache()
            print(f"    ✓ Cleared GPU memory")
        
        # Return early (only alignment, no metrics)
        return {
            'system_name': system_name,
            'workflow': workflow_name,
            'model': model_name,
            'dataset': dataset,
            'lang_pair': lang_pair,
            'num_samples': len(sample_data),
            'num_segments': len(aligned_df),
            'tmp_file': str(tmp_file),
            'labse_only': True
        }
    else:
        # Normal flow: do alignment
        _log_with_time(f"  Step 1: Splitting and aligning documents...")
        from metrics.docpreprocessor import DocPreprocessor
        
        use_gpu = torch.cuda.is_available()
        if labse_model is None or polyfuzz_model is None:
            from metrics.docpreprocessor import (
                load_labse_model_once,
                load_polyfuzz_model_once,
                load_embeddings_wrapper_once,
                find_labse_model_path
            )
            _log_with_time("  Loading models (fallback - should be provided from main)...")
            labse_model_path = find_labse_model_path()
            if labse_model is None:
                labse_model = load_labse_model_once(labse_model_path, use_gpu=use_gpu)
            if polyfuzz_model is None:
                embeddings = load_embeddings_wrapper_once(labse_model_path, labse_model)
                polyfuzz_model = load_polyfuzz_model_once(labse_model_path, embeddings)
        
        preprocessor = DocPreprocessor(
            src_lang, 
            tgt_lang, 
            labse_model=labse_model,
            polyfuzz_model=polyfuzz_model,
            use_gpu=use_gpu
        )
        aligned_df = preprocessor.process_documents(
            documents,
            references=references,  # Pass references for alignment
            terminology=terminology,
            similarity_threshold=0.4,
            separator='\n\n'
        )
        skipped_count = getattr(preprocessor, 'skipped_segments_count', 0)
        print(f"    ✓ Aligned {len(aligned_df)} segments" + (f" (skipped {skipped_count} segments without src-ref alignment)" if skipped_count > 0 else ""))
        
        # Clear GPU memory after LaBSE (before loading MetricX/COMET)
        if use_gpu:
            torch.cuda.empty_cache()
            print(f"    ✓ Cleared GPU memory after LaBSE alignment")
    
    # 2. Run termbasedmetric (WMT25-Term only, and only if not comet_only)
    tbm_results = {
        'first': {'micro': None, 'macro': None},
        'frequent': {'micro': None, 'macro': None},
        'predefined': {'micro': None, 'macro': None}
    }
    
    # Skip TBM for DOLFIN (no terminology) or if --comet-only or --metricx-only flag is set
    # But compute if --tbm-only flag is set
    should_compute_tbm = (tbm_only or (dataset == "wmt25" and terminology and not comet_only and not metricx_only))
    
    if should_compute_tbm:
        print(f"  Step 2: Computing term-based metrics...")
        try:
            from metrics.termbasedmetric import TermBasedMetric
            
            # Initialize term-based metric
            tbm = TermBasedMetric(src_lang, tgt_lang, keyword_extractor='predefined', aligner='llm')
            
            # Prepare data for termbasedmetric
            # We need to convert aligned_df to the format termbasedmetric expects
            # For now, we'll create a simplified version that works with aligned segments
            # TODO: Full integration may require adapting termbasedmetric to work with our format
            
            print(f"    ⚠ Term-based metrics computation requires full integration (TODO)")
            print(f"    ⚠ Skipping TBM for now - will be implemented in next iteration")
            
        except Exception as e:
            print(f"    ⚠ Warning: Could not compute term-based metrics: {e}")
    elif dataset == "dolfin":
        print(f"  Step 2: Skipping term-based metrics (DOLFIN has no terminology)")
    elif comet_only:
        print(f"  Step 2: Skipping term-based metrics (--comet-only flag set)")
    elif metricx_only:
        print(f"  Step 2: Skipping term-based metrics (--metricx-only flag set)")
    
    # 3. Run metric evaluator (COMET or MetricX) - per sample
    # Skip if tbm_only mode
    if tbm_only:
        print(f"  Step 3: Skipping metric evaluation (--tbm-only mode)")
        metric_results = {
            f'avg_{use_metric}': None,
            f'min_{use_metric}': None,
            f'max_{use_metric}': None,
            'per_sample': []
        }
    else:
        import time
        metric_name = "MetricX" if use_metric == "metricx" else "COMET"
        print(f"  Step 3: Computing {metric_name} scores per sample...")
        step3_start = time.time()
        
        metric_results = {
            f'avg_{use_metric}': None,
            f'min_{use_metric}': None,
            f'max_{use_metric}': None,
            'per_sample': []
        }
        
        # Track alignment statistics per sample
        alignment_stats = []
        
        try:
            if use_metric == "metricx":
                from metrics.metricx_evaluator import compute_metricx_scores
                # Use pre-loaded model and tokenizer if available
                if metric_model is not None and metric_tokenizer is not None:
                    # Create a wrapper that uses the pre-loaded model
                    def compute_with_preloaded(segments):
                        return compute_metricx_scores(
                            segments,
                            metric_model=metric_model,
                            tokenizer=metric_tokenizer
                        )
                    compute_metric_fn = compute_with_preloaded
                else:
                    compute_metric_fn = compute_metricx_scores
            else:
                from metrics.comet_evaluator import compute_comet_scores
                compute_metric_fn = compute_comet_scores
            
            _log_with_time = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if metric_model is None:
                _log_with_time(f"  Loading {metric_name} evaluator...")
            else:
                _log_with_time(f"  Using pre-loaded {metric_name} model...")
            # Group segments by sample (paragraph column indicates document index)
            try:
                from tqdm import tqdm
                sample_iterator = tqdm(range(len(sample_data)), desc=f"  Computing {metric_name} per sample")
            except ImportError:
                sample_iterator = range(len(sample_data))
            
            for sample_idx in sample_iterator:
                sample_info = sample_data[sample_idx]
                # Filter by document column (which corresponds to sample index)
                sample_segments_df = aligned_df[aligned_df['document'] == sample_idx]
                
                if len(sample_segments_df) == 0:
                    # No aligned segments for this sample
                    alignment_stats.append({
                        'sample_idx': sample_idx,
                        'sample_id': sample_info['sample_id'],
                        'under_translated_segments': 0,  # We'll compute this below
                        'over_translated_segments': 0,
                        f'{use_metric}_scores': [],
                        f'avg_{use_metric}': None
                    })
                    continue
                
                # Prepare segments for this sample: (source, translation, reference)
                # IMPORTANT: Every source segment should have a reference segment
                # - Under-translation (empty target): (src, "", ref) → MetricX will penalize (low score)
                # - Over-translation (empty source): ("", tgt, None) → May not have reference, but we still compute
                segments = []
                for idx, row in sample_segments_df.iterrows():
                    # DataFrame columns are 'src_segment', 'tgt_segment', and 'ref_segment'
                    src = str(row['src_segment']) if pd.notna(row['src_segment']) else ""
                    tgt = str(row['tgt_segment']) if pd.notna(row['tgt_segment']) else ""
                    ref = row.get('ref_segment')
                    
                    # Handle reference segment
                    # Check if reference was successfully aligned (has_ref_alignment flag)
                    has_ref_alignment = row.get('has_ref_alignment', True)  # Default True for backward compatibility
                    
                    if ref is None or (isinstance(ref, float) and pd.isna(ref)) or (isinstance(ref, str) and not ref.strip()):
                        # If we have a source segment but no reference, this is a problem
                        # (every source should have a reference)
                        if src and src.strip():  # Source exists but no reference
                            # DO NOT use full reference text - this should not happen
                            # If alignment failed, we should flag it and skip or use empty
                            print(f"      ⚠ WARNING: Source segment has no aligned reference for sample {sample_idx}, segment {idx} (alignment failed)")
                            ref = ""  # Use empty string, not full reference
                        else:
                            # No source (over-translation) - ref can be empty, this is OK
                            ref = ""
                    elif not has_ref_alignment:
                        # Reference exists but alignment flag indicates it's not properly aligned
                        print(f"      ⚠ WARNING: Source segment {idx} in sample {sample_idx} has reference but alignment flag is False")
                    else:
                        ref = str(ref)
                    
                    # Only add segments that have at least source or target
                    # (we need at least one to compute MetricX)
                    # For over-translation (empty src, tgt exists): we include it but MetricX might handle it
                    # For under-translation (src exists, empty tgt): we include it to penalize (MetricX will give low score)
                    if src or tgt:
                        segments.append((src, tgt, ref))
                
                # Compute metric scores for this sample's segments
                _log_with_time(f"    Sample {sample_idx+1}/{len(sample_data)}: Computing {metric_name} for {len(segments)} segments...")
                sample_metric_start = time.time()
                sample_metric = compute_metric_fn(segments)
                metric_scores = sample_metric.get('scores', [])
                avg_key = f'avg_{use_metric}'
                _log_with_time(f"      ✓ {metric_name} computed in {time.time() - sample_metric_start:.2f}s (avg: {sample_metric.get(avg_key, 0):.4f})")
                
                # Count alignment statistics
                # Get source and target paragraphs for this sample (same splitting as docpreprocessor)
                src_text = sample_info['source_text']
                tgt_text = sample_info['translation']
                separator = '\n\n'
                
                src_paragraphs = [p.strip() for p in src_text.split(separator) if p.strip()]
                tgt_paragraphs = [p.strip() for p in tgt_text.split(separator) if p.strip()]
                
                # Track which source and target segments have alignments
                # We'll check if each paragraph has at least one aligned segment
                aligned_src_paragraphs = set()
                aligned_tgt_paragraphs = set()
                
                # For each aligned segment, find which source/target paragraph it belongs to
                for _, row in sample_segments_df.iterrows():
                    # DataFrame columns are 'src_segment' and 'tgt_segment', not language codes
                    src_seg = str(row['src_segment']).strip()
                    tgt_seg = str(row['tgt_segment']).strip()
                    
                    # Find which source paragraph contains this segment
                    for para_idx, src_para in enumerate(src_paragraphs):
                        # Check if segment is part of this paragraph (substring match)
                        if src_seg in src_para or (len(src_seg) > 20 and src_para in src_seg):
                            aligned_src_paragraphs.add(para_idx)
                            break
                    
                    # Find which target paragraph contains this segment
                    for para_idx, tgt_para in enumerate(tgt_paragraphs):
                        # Check if segment is part of this paragraph (substring match)
                        if tgt_seg in tgt_para or (len(tgt_seg) > 20 and tgt_para in tgt_seg):
                            aligned_tgt_paragraphs.add(para_idx)
                            break
                
                # Count unaligned segments
                # Under-translated: source paragraphs without any aligned output
                under_translated = len(src_paragraphs) - len(aligned_src_paragraphs)
                # Over-translated: target paragraphs without any aligned source
                over_translated = len(tgt_paragraphs) - len(aligned_tgt_paragraphs)
                
                alignment_stats.append({
                    'sample_idx': sample_idx,
                    'sample_id': sample_info['sample_id'],
                    'under_translated_segments': under_translated,
                    'over_translated_segments': over_translated,
                    f'{use_metric}_scores': metric_scores,
                    f'avg_{use_metric}': sample_metric.get(avg_key)
                })
            
            # Aggregate metric scores across all samples (outside the loop)
            all_metric_scores = []
            for stat in alignment_stats:
                all_metric_scores.extend(stat[f'{use_metric}_scores'])
            
            if all_metric_scores:
                metric_results[f'avg_{use_metric}'] = sum(all_metric_scores) / len(all_metric_scores)
                metric_results[f'min_{use_metric}'] = min(all_metric_scores)
                metric_results[f'max_{use_metric}'] = max(all_metric_scores)
            
            # Aggregate alignment statistics
            total_under = sum(s['under_translated_segments'] for s in alignment_stats)
            total_over = sum(s['over_translated_segments'] for s in alignment_stats)
            avg_under = total_under / len(alignment_stats) if alignment_stats else 0
            avg_over = total_over / len(alignment_stats) if alignment_stats else 0
            
            metric_results['per_sample'] = alignment_stats
            metric_results['alignment_stats'] = {
                'total_under_translated_segments': total_under,
                'total_over_translated_segments': total_over,
                'avg_under_translated_segments': avg_under,
                'avg_over_translated_segments': avg_over,
                'skipped_segments_no_ref_alignment': skipped_count  # Segments skipped due to missing src-ref alignment
            }
            
            step3_time = time.time() - step3_start
            _log_with_time(f"  ✓ Step 3 complete in {step3_time:.2f}s")
            avg_key = f'avg_{use_metric}'
            min_key = f'min_{use_metric}'
            max_key = f'max_{use_metric}'
            print(f"    ✓ {metric_name}: avg={metric_results[avg_key]:.4f}, "
                  f"min={metric_results[min_key]:.4f}, "
                  f"max={metric_results[max_key]:.4f}")
            print(f"    ✓ Alignment: avg_under={avg_under:.2f}, avg_over={avg_over:.2f}")
            
        except Exception as e:
            print(f"    ⚠ Warning: Could not compute {metric_name} scores: {e}")
        import traceback
        traceback.print_exc()
        
        # Clear GPU cache on metric error (both metrics use GPU)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    # Build results
    # Note: experiment_start is defined at the beginning of process_experiment
    # If it's not defined, we'll skip the timing
    try:
        experiment_time = time.time() - experiment_start
        _log_with_time(f"  ✓ Experiment processing complete in {experiment_time:.2f}s")
    except (NameError, UnboundLocalError):
        pass  # experiment_start not defined, skip timing
    
    results = {
        'system_name': system_name,
        'workflow': workflow_name,
        'model': model_name,
        'dataset': dataset,
        'lang_pair': lang_pair,
        'num_samples': len(sample_data),
        'num_segments': len(aligned_df),
        'first': tbm_results['first'],
        'frequent': tbm_results['frequent'],
        'predefined': tbm_results['predefined'],
        use_metric: metric_results  # Dynamic key based on use_metric
    }
    
    print(f"  ✓ Processed successfully")
    
    return results


def get_metrics_file_path(
    output_dir: Path,
    metrics_base_dir: Path
) -> Path:
    """
    Get the path to the metrics.json file for an experiment.
    
    Args:
        output_dir: Original output directory (e.g., outputs/wmt25/en-zht/IRB.term/gpt-4-1)
        metrics_base_dir: Base directory for metrics results (e.g., metrics/results)
    
    Returns:
        Path to metrics.json file
    """
    # Parse the output directory structure: {base}/{dataset}/{lang_pair}/{workflow_dir}/{model}
    parts = output_dir.parts
    
    # Find the base outputs directory name (outputs or outputs_qwen3)
    base_name = None
    for part in parts:
        if part in ['outputs', 'outputs_qwen3']:
            base_name = part
            break
    
    if not base_name:
        raise ValueError(f"Could not determine base directory from path: {output_dir}")
    
    base_idx = parts.index(base_name)
    
    # Extract: dataset, lang_pair, workflow_dir, model
    if len(parts) < base_idx + 5:
        raise ValueError(f"Invalid output directory structure: {output_dir}")
    
    dataset = parts[base_idx + 1]
    lang_pair = parts[base_idx + 2]
    workflow_dir = parts[base_idx + 3]  # e.g., "IRB.term" or "IRB"
    model = parts[base_idx + 4]
    
    # Build metrics directory structure matching outputs
    metrics_output_dir = metrics_base_dir / dataset / lang_pair / workflow_dir / model
    return metrics_output_dir / "metrics.json"


def has_complete_metrics(
    metrics_file: Path,
    dataset: str,
    comet_only: bool = False,
    metricx_only: bool = False,
    use_metric: str = "metricx"
) -> bool:
    """
    Check if a metrics.json file exists and contains all required metrics.
    
    For WMT25: requires 'first', 'frequent', 'predefined' (TBM) and 'comet'
    For DOLFIN: requires 'comet' only
    
    Args:
        metrics_file: Path to metrics.json file
        dataset: Dataset name ('wmt25' or 'dolfin')
    
    Returns:
        True if file exists and has all required metrics, False otherwise
    """
    if not metrics_file.exists():
        return False
    
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False
    
    # Check for metric (COMET or MetricX) - required for both datasets
    # Use the specified metric, or try to detect from data
    metric_key = use_metric
    if metric_key not in metrics_data:
        # Try to detect from data if use_metric not found
        if 'metricx' in metrics_data:
            metric_key = 'metricx'
        elif 'comet' in metrics_data:
            metric_key = 'comet'
        else:
            return False
    
    metric_data = metrics_data.get(metric_key, {})
    avg_key = f'avg_{metric_key}'
    if not metric_data or metric_data.get(avg_key) is None:
        return False
    
    # For WMT25, also check for TBM metrics (unless comet_only or metricx_only mode)
    if dataset == "wmt25" and not comet_only and not metricx_only:
        first = metrics_data.get('first', {})
        frequent = metrics_data.get('frequent', {})
        predefined = metrics_data.get('predefined', {})
        
        # Check if at least one TBM metric has been computed
        # TBM metrics are dicts with 'micro' and 'macro' keys
        # If they're None, TBM was not computed yet
        has_tbm = False
        for tbm_dict in [first, frequent, predefined]:
            if tbm_dict and isinstance(tbm_dict, dict):
                # Check if it has non-None values (meaning TBM was computed)
                micro = tbm_dict.get('micro')
                macro = tbm_dict.get('macro')
                if micro is not None or macro is not None:
                    has_tbm = True
                    break
        
        if not has_tbm:
            return False
    
    # All required metrics are present
    return True


def save_metrics_results(
    results: Dict[str, Any],
    output_dir: Path,
    metrics_base_dir: Path
) -> None:
    """
    Save metrics results to a file matching the output directory structure.
    
    Structure: metrics/results/{dataset}/{lang_pair}/{workflow_dir}/{model}/metrics.json
    
    Args:
        results: Results dictionary for one experiment
        output_dir: Original output directory (e.g., outputs/wmt25/en-zht/IRB.term/gpt-4-1)
        metrics_base_dir: Base directory for metrics results (e.g., metrics/results)
    """
    # Get metrics file path using shared function
    output_file = get_metrics_file_path(output_dir, metrics_base_dir)
    
    # Extract dataset and workflow_dir for metrics_data
    parts = output_dir.parts
    base_name = None
    for part in parts:
        if part in ['outputs', 'outputs_qwen3']:
            base_name = part
            break
    
    if not base_name:
        raise ValueError(f"Could not determine base directory from path: {output_dir}")
    
    base_idx = parts.index(base_name)
    dataset = parts[base_idx + 1]
    lang_pair = parts[base_idx + 2]
    workflow_dir = parts[base_idx + 3]  # e.g., "IRB.term" or "IRB"
    model = parts[base_idx + 4]
    
    # Ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results (overwrite - each experiment gets its own file)
    metrics_data = {
        'dataset': dataset,
        'lang_pair': lang_pair,
        'workflow': workflow_dir,
        'model': model,
        'system_name': results.get('system_name', f"{workflow_dir}+{model}"),
        'num_samples': results.get('num_samples', 0),
        'num_segments': results.get('num_segments', 0),
        'first': results.get('first', {}),
        'frequent': results.get('frequent', {}),
        'predefined': results.get('predefined', {}),
        'comet': results.get('comet', {}),
        'metricx': results.get('metricx', {})
    }
    
    # Atomic write
    temp_file = output_file.with_suffix('.json.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    temp_file.replace(output_file)
    
    print(f"  ✓ Saved results to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate experiments using term-consistency metrics"
    )
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
        help="Target language filter (e.g., 'zht' for WMT25, 'es' for DOLFIN). "
             "If not specified, processes all language pairs."
    )
    parser.add_argument(
        "--outputs_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Custom output directories to scan (default: outputs/ and outputs_qwen3/)"
    )
    parser.add_argument(
        "--metrics_output_dir",
        type=str,
        default=None,
        help="Directory to save metrics results (default: metrics/results/)"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Filter by workflow (e.g., 'IRB', 'MaMT', 'IRB.term'). "
             "If not specified, processes all workflows."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model (e.g., 'gpt-4-1', 'qwen3-32b'). "
             "If not specified, processes all models."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already have complete metrics computed. "
             "For WMT25: requires TBM (first/frequent/predefined) and COMET. "
             "For DOLFIN: requires COMET only."
    )
    parser.add_argument(
        "--comet-only",
        action="store_true",
        help="Compute only COMET scores, skip term-based metrics (TBM). "
             "Useful for faster evaluation or when TBM dependencies are unavailable. "
             "For DOLFIN, this is the default behavior."
    )
    parser.add_argument(
        "--metricx-only",
        action="store_true",
        help="Compute only MetricX scores, skip term-based metrics (TBM). "
             "Loads aligned segments from tmp files if available. "
             "For WMT25 only. Useful for faster evaluation."
    )
    parser.add_argument(
        "--labse-only",
        action="store_true",
        help="Only perform LaBSE alignment and save to tmp files. "
             "Useful for splitting GPU memory usage. Run this first, then use --metricx-only or --tbm-only."
    )
    parser.add_argument(
        "--tbm-only",
        action="store_true",
        help="Compute only term-based metrics (TBM), skip metric evaluation. "
             "Loads aligned segments from tmp files if available. "
             "For WMT25 only."
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["metricx", "comet"],
        default="metricx",
        help="Which metric to use for evaluation: 'metricx' (default) or 'comet'"
    )
    
    args = parser.parse_args()
    
    # Set up output directories
    if args.outputs_dirs:
        outputs_dirs = [Path(d) for d in args.outputs_dirs]
    else:
        outputs_dirs = OUTPUTS_DIRS
    
    if args.metrics_output_dir:
        metrics_dir = Path(args.metrics_output_dir)
    else:
        metrics_dir = METRICS_OUTPUT_DIR
    
    # Get workflows and models to process
    workflows, models = get_workflows_and_models_to_process()
    
    print("="*80)
    print("Document-Level Metrics Evaluation")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    if args.target_language:
        print(f"Target language filter: {args.target_language}")
    else:
        print(f"Target language filter: all")
    print(f"Workflows to process: {workflows}")
    print(f"Models to process: {models}")
    if args.workflow:
        print(f"Workflow filter: {args.workflow}")
    if args.model:
        print(f"Model filter: {args.model}")
    if args.comet_only:
        print(f"Mode: COMET-only (skipping term-based metrics)")
    if args.metricx_only:
        print(f"Mode: MetricX-only (skipping term-based metrics)")
    if args.labse_only:
        print(f"Mode: LaBSE-only (alignment only, saving to tmp)")
    if args.tbm_only:
        print(f"Mode: TBM-only (term-based metrics only, loading from tmp)")
    print(f"Using metric: {args.metric.upper()}")
    print(f"Output directories: {[str(d) for d in outputs_dirs]}")
    print(f"Metrics output: {metrics_dir}")
    print("="*80)
    
    # Find experiments
    print("\nFinding experiments...")
    all_experiments = find_experiments(
        args.dataset, 
        args.target_language, 
        args.workflow,
        args.model,
        outputs_dirs
    )
    print(f"Found {len(all_experiments)} completed experiments")
    
    # Filter by workflows and models (from write_tables_paper.py)
    print(f"\nFiltering by workflows: {workflows}")
    print(f"Filtering by models: {models}")
    experiments = []
    skipped_resume = 0
    for output_dir, report_data in all_experiments:
        if not should_process_experiment(report_data, workflows, models):
            continue
        
        # Check if we should skip due to --resume
        if args.resume:
            try:
                metrics_file = get_metrics_file_path(output_dir, metrics_dir)
                if has_complete_metrics(metrics_file, args.dataset, comet_only=args.comet_only, metricx_only=args.metricx_only, use_metric=args.metric):
                    skipped_resume += 1
                    continue
            except (ValueError, Exception):
                # If we can't determine the path, process it anyway
                pass
        
        experiments.append((output_dir, report_data))
    
    print(f"After filtering: {len(experiments)} experiments to process")
    if args.resume and skipped_resume > 0:
        print(f"Skipped {skipped_resume} experiments with complete metrics (--resume)")
    
    if not experiments:
        print("No experiments found matching the workflow/model filters. Exiting.")
        return 1
    
    # Group by lang_pair for processing
    experiments_by_lang_pair = defaultdict(list)
    for output_dir, report_data in experiments:
        lang_pair = report_data['_lang_pair']
        experiments_by_lang_pair[lang_pair].append((output_dir, report_data))
    
    print(f"\nProcessing {len(experiments_by_lang_pair)} language pair(s)...")
    
    # Load models ONCE at the start (reused for all experiments)
    labse_model = None
    polyfuzz_model = None
    metric_model = None
    
    # Load LaBSE models only if not in metricx_only or tbm_only mode (those load from tmp)
    if not args.metricx_only and not args.tbm_only:
        _log_with_time("="*80)
        _log_with_time("Loading LaBSE model and embeddings (once, will be reused for all experiments)...")
        from metrics.docpreprocessor import (
            load_labse_model_once,
            load_embeddings_wrapper_once,
            load_polyfuzz_model_once,
            find_labse_model_path
        )
        import torch
        use_gpu = torch.cuda.is_available()
        
        # Find model path first
        labse_model_path = find_labse_model_path()
        
        # Load LaBSE model
        labse_model = load_labse_model_once(labse_model_path, use_gpu=use_gpu)
        
        # Load embeddings wrapper (needs path, not model object)
        embeddings = load_embeddings_wrapper_once(labse_model_path, labse_model)
        
        # Load PolyFuzz model
        polyfuzz_model = load_polyfuzz_model_once(labse_model_path, embeddings)
        
        _log_with_time("  ✓ All LaBSE models loaded and ready")
        _log_with_time("="*80)
    
    # Load MetricX/COMET model ONCE at the start (only if not labse_only mode)
    metric_tokenizer = None  # Store tokenizer separately
    if not args.labse_only and args.metric == "metricx":
        _log_with_time("="*80)
        _log_with_time("Loading MetricX-24 model and tokenizer (once, will be reused for all experiments)...")
        import torch
        use_gpu = torch.cuda.is_available()
        
        from metrics.metricx_evaluator import _find_metricx_model, _find_mt5_tokenizer
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import os
        
        # Find model paths
        metricx_model_path = _find_metricx_model()
        mt5_tokenizer_path = _find_mt5_tokenizer()
        
        if metricx_model_path and mt5_tokenizer_path:
            # Set environment variables for offline mode
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # Load tokenizer
            _log_with_time("  Loading mT5 tokenizer...")
            metric_tokenizer = AutoTokenizer.from_pretrained(
                str(mt5_tokenizer_path),
                local_files_only=True
            )
            
            # Load model
            _log_with_time("  Loading MetricX-24 model...")
            metric_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(metricx_model_path),
                local_files_only=True,
                torch_dtype=torch.bfloat16 if use_gpu else torch.float32
            )
            
            # Move to GPU if available
            if use_gpu:
                metric_model = metric_model.to('cuda')
                _log_with_time(f"  ✓ MetricX-24 loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                _log_with_time("  ✓ MetricX-24 loaded on CPU")
            
            _log_with_time("="*80)
        else:
            _log_with_time("  ⚠ Warning: Could not find MetricX model, will load on-the-fly")
            metric_model = None
            metric_tokenizer = None
    
    # Process each language pair
    for lang_pair, lang_experiments in experiments_by_lang_pair.items():
        print(f"\n{'='*80}")
        print(f"Language Pair: {lang_pair}")
        print(f"  Experiments: {len(lang_experiments)}")
        print(f"{'='*80}")
        
        # Process each experiment
        for output_dir, report_data in lang_experiments:
            try:
                results = process_experiment(
                    output_dir,
                    report_data,
                    args.dataset,
                    lang_pair,
                    workflows,
                    models,
                    labse_model=labse_model,  # Pass the pre-loaded model
                    polyfuzz_model=polyfuzz_model,  # Pass the pre-loaded PolyFuzz model
                    metric_model=metric_model,  # Pass the pre-loaded metric model
                    metric_tokenizer=metric_tokenizer,  # Pass the pre-loaded tokenizer
                    comet_only=args.comet_only,  # Pass the comet_only flag
                    metricx_only=args.metricx_only,  # Pass the metricx_only flag
                    tbm_only=args.tbm_only,  # Pass the tbm_only flag
                    labse_only=args.labse_only,  # Pass the labse_only flag
                    use_metric=args.metric  # Pass the metric choice
                )
                
                if results:
                    # Save results to file matching output directory structure
                    save_metrics_results(results, output_dir, metrics_dir)
                    
                    # Clear GPU cache after each experiment to prevent memory buildup
                    if args.metric == "metricx" and metric_model is not None:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    
            except Exception as e:
                print(f"  ✗ Error processing experiment: {e}")
                import traceback
                traceback.print_exc()
                
                # Clear GPU cache on error to prevent memory buildup
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"  ✓ Cleared GPU cache after error")
                except Exception:
                    pass
                
                continue
    
    # Final GPU cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log_with_time("Cleared GPU cache after evaluation")
    except Exception:
        pass
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print(f"Results saved to: {metrics_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


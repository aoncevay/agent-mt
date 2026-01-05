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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

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
    models: List[str]
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
    # Check if we should process this experiment
    if not should_process_experiment(report_data, workflows, models):
        return None
    
    workflow_name = report_data.get('workflow', '')
    model_name = report_data.get('model', '')
    use_terminology = '.term' in str(output_dir)
    
    print(f"\n{'='*80}")
    print(f"Processing: {workflow_name} + {model_name}")
    print(f"  Dataset: {dataset}, Lang pair: {lang_pair}")
    print(f"  Output dir: {output_dir}")
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
    
    # 1. Run docpreprocessor (split + align segments)
    print(f"  Step 1: Splitting and aligning documents...")
    from metrics.docpreprocessor import DocPreprocessor
    preprocessor = DocPreprocessor(src_lang, tgt_lang)
    aligned_df = preprocessor.process_documents(
        documents,
        terminology=terminology,
        similarity_threshold=0.4,
        separator='\n\n'
    )
    print(f"    ✓ Aligned {len(aligned_df)} segments")
    
    # 2. Run termbasedmetric (WMT25-Term only)
    tbm_results = {
        'first': {'micro': None, 'macro': None},
        'frequent': {'micro': None, 'macro': None},
        'predefined': {'micro': None, 'macro': None}
    }
    
    if dataset == "wmt25" and terminology:
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
    
    # 3. Run comet_evaluator (all datasets)
    print(f"  Step 3: Computing COMET scores...")
    comet_results = {
        'avg_comet': None,
        'min_comet': None,
        'max_comet': None
    }
    
    try:
        from metrics.comet_evaluator import compute_comet_scores
        
        # Prepare segments: (source, translation, reference)
        # For now, use document-level references (we'll align them to segments later)
        segments = []
        for idx, row in aligned_df.iterrows():
            src = row[src_lang]
            tgt = row[tgt_lang]
            # Use first reference for now (document-level)
            # TODO: Align references to segments properly
            ref = references[0] if references else ""
            segments.append((src, tgt, ref))
        
        comet_results = compute_comet_scores(segments)
        print(f"    ✓ COMET: avg={comet_results['avg_comet']:.4f}, "
              f"min={comet_results['min_comet']:.4f}, "
              f"max={comet_results['max_comet']:.4f}")
        
    except Exception as e:
        print(f"    ⚠ Warning: Could not compute COMET scores: {e}")
        import traceback
        traceback.print_exc()
    
    # Build results
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
        'comet': comet_results
    }
    
    print(f"  ✓ Processed successfully")
    
    return results


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
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to metrics.json
    output_file = metrics_output_dir / "metrics.json"
    
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
        'comet': results.get('comet', {})
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
    print(f"Output directories: {[str(d) for d in outputs_dirs]}")
    print(f"Metrics output: {metrics_dir}")
    print("="*80)
    
    # Find experiments
    print("\nFinding experiments...")
    experiments = find_experiments(
        args.dataset, 
        args.target_language, 
        args.workflow,
        args.model,
        outputs_dirs
    )
    print(f"Found {len(experiments)} completed experiments")
    
    if not experiments:
        print("No experiments found. Exiting.")
        return 1
    
    # Group by lang_pair for processing
    experiments_by_lang_pair = defaultdict(list)
    for output_dir, report_data in experiments:
        lang_pair = report_data['_lang_pair']
        experiments_by_lang_pair[lang_pair].append((output_dir, report_data))
    
    print(f"\nProcessing {len(experiments_by_lang_pair)} language pair(s)...")
    
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
                    models
                )
                
                if results:
                    # Save results to file matching output directory structure
                    save_metrics_results(results, output_dir, metrics_dir)
                    
            except Exception as e:
                print(f"  ✗ Error processing experiment: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print(f"Results saved to: {metrics_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


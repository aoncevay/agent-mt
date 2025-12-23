#!/usr/bin/env python3
"""
Main runner script for translation experiments.

Usage:
    python src/run.py --dataset wmt25 --workflow zero_shot --model qwen3-235b
    python src/run.py --dataset dolfin_en_es --workflow zero_shot --model qwen3-235b --max_samples 10
    python src/run.py --dataset wmt25 --workflow MaMT_translate_postedit --model qwen3-235b
    python src/run.py --dataset wmt25 --workflow MaMT_translate_postedit_proofread --model qwen3-235b
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loaders import get_data_loader
from workflows import WORKFLOW_REGISTRY
from evaluation import compute_chrf, compute_bleu, compute_term_success_rate
from vars import model_name2bedrock_id, model_name2bedrock_arn, model_name2provider, model_name2openai_id
from workflow_acronyms import build_output_dir

# Load environment variables
load_dotenv("config.env")

# Configuration
BASE_DATA_DIR = (Path(__file__).parent.parent / "data" / "raw").resolve()
OUTPUT_BASE_DIR = (Path(__file__).parent.parent / "outputs").resolve()
MAX_RETRIES = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.getenv("BEDROCK_INITIAL_BACKOFF", "2.0"))
# AWS_REGION will be set after parsing args (default: us-east-2, but us-east-1 for ARNs)
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
# AWS_REGION will be set after parsing args (depends on model vs ARN)


def get_workflow_module(workflow_name: str):
    """Import and return workflow module."""
    if workflow_name not in WORKFLOW_REGISTRY:
        raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(WORKFLOW_REGISTRY.keys())}")
    
    module_name = WORKFLOW_REGISTRY[workflow_name]
    
    # Import workflow module
    if workflow_name == "zero_shot":
        from workflows import zero_shot
        return zero_shot
    else:
        # For all other workflows, use dynamic import
        import importlib
        return importlib.import_module(f"workflows.{module_name}")


def process_sample(
    sample: Dict[str, Any],
    data_loader,
    workflow_module,
    model_id: str,
    sample_idx: int,
    lang_pair: Optional[str] = None,
    use_terminology: bool = False,
    model_provider: Optional[str] = None,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single sample through the workflow."""
    # Get translation direction
    source_lang, target_lang = data_loader.get_translation_direction(sample)
    
    # Skip if filtered out (None, None)
    if source_lang is None or target_lang is None:
        return None
    
    # Extract sample ID if available (for DOLFIN, WMT25, etc.)
    sample_id = sample.get("id") or sample.get("_id") or str(sample_idx)
    
    # Extract texts
    source_text, reference_text, terminology = data_loader.extract_texts(
        sample, source_lang, target_lang
    )
    
    if not source_text or not reference_text:
        return None
    
    # Build display name for sample
    sample_display = f"ID:{sample_id}" if sample_id != str(sample_idx) else f"#{sample_idx + 1}"
    print(f"  Sample {sample_display} ({source_lang}->{target_lang}): "
          f"{len(source_text)} chars source, {len(reference_text)} chars reference")
    
    try:
        # Run workflow
        # Check if workflow supports reference parameter
        import inspect
        sig = inspect.signature(workflow_module.run_workflow)
        workflow_kwargs = {
            "source_text": source_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model_id": model_id,
            "terminology": terminology if use_terminology else None,
            "use_terminology": use_terminology,
            "max_retries": MAX_RETRIES,
            "initial_backoff": INITIAL_BACKOFF
        }
        
        # Add region only for Bedrock models
        if "region" in sig.parameters:
            if model_type == "bedrock" and AWS_REGION:
                workflow_kwargs["region"] = AWS_REGION
            else:
                workflow_kwargs["region"] = None  # OpenAI models don't need region
        
        # Add reference if workflow supports it
        if "reference" in sig.parameters:
            workflow_kwargs["reference"] = reference_text
        
        # Add model_provider if workflow supports it and we have one (Bedrock ARNs only)
        if model_provider and "model_provider" in sig.parameters:
            workflow_kwargs["model_provider"] = model_provider
        
        # Add model_type if workflow supports it
        if model_type and "model_type" in sig.parameters:
            workflow_kwargs["model_type"] = model_type
        
        result = workflow_module.run_workflow(**workflow_kwargs)
        
        # Evaluate each output
        outputs = result["outputs"]
        evaluations = []
        
        for i, output in enumerate(outputs):
            chrf_result = compute_chrf(output, reference_text)
            bleu_result = compute_bleu(output, reference_text, target_lang)
            
            # Compute term success rate if terminology is available
            # For WMT25, always compute it (dataset always has terminology)
            # For other datasets, only compute if use_terminology is True
            term_success_rate = -1.0
            if terminology:  # If terminology exists in the data
                dataset_name = data_loader.get_dataset_name()
                # WMT25 always has terminology, so compute term success rate even without --use_terminology
                # For DOLFIN and other datasets, only compute if explicitly requested
                # Check if dataset name starts with "wmt25" (exact match) or use_terminology flag
                if dataset_name == "wmt25" or use_terminology:
                    term_success_rate = compute_term_success_rate(
                        source_text, output, reference_text, terminology, lowercase=True
                    )
            
            evaluations.append({
                "agent_id": i,
                "chrf_score": chrf_result["score"],
                "bleu_score": bleu_result["score"],
                "term_success_rate": term_success_rate,
                "translation": output
            })
        
        return {
            "sample_idx": sample_idx,
            "sample_id": sample_id,  # Original ID from data
            "source_lang": source_lang,
            "target_lang": target_lang,
            "lang_pair": lang_pair,  # Language pair (e.g., "en_es")
            "source_text": source_text,
            "reference_text": reference_text,
            "outputs": outputs,
            "evaluations": evaluations,
            "tokens_input": result["tokens_input"],
            "tokens_output": result["tokens_output"],
            "latency": result["latency"],
            "error": None
        }
    
    except Exception as e:
        print(f"    ✗ Error: {e}")
        sample_id = sample.get("id") or sample.get("_id") or str(sample_idx)
        return {
            "sample_idx": sample_idx,
            "sample_id": sample_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "lang_pair": lang_pair,
            "source_text": source_text,
            "reference_text": reference_text,
            "outputs": [],
            "evaluations": [],
            "tokens_input": 0,
            "tokens_output": 0,
            "latency": None,
            "error": str(e)
        }


def load_existing_report(report_file: Path) -> Optional[Dict[str, Any]]:
    """Load existing report if it exists."""
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠ Warning: Could not load existing report: {e}")
            return None
    return None


def get_processed_sample_ids(existing_report: Optional[Dict[str, Any]]) -> set:
    """Extract set of processed sample IDs from existing report."""
    if not existing_report:
        return set()
    
    processed = set()
    for sample in existing_report.get("samples", []):
        sample_id = sample.get("sample_id")
        if sample_id:
            processed.add(str(sample_id))
        # Also track by sample_idx + lang_pair for cases without IDs
        sample_idx = sample.get("sample_idx")
        lang_pair = sample.get("lang_pair")
        if sample_idx is not None and lang_pair:
            processed.add(f"{lang_pair}_{sample_idx}")
    
    return processed


def save_outputs(
    results: List[Dict[str, Any]],
    output_dir: Path,
    dataset_name: str,
    workflow_name: str,
    model_name: str,
    max_samples: Optional[int] = None,
    resume: bool = False
):
    """Save translation outputs and generate report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine report filename
    if max_samples is not None:
        report_filename = f"report_{max_samples}_samples.json"
    else:
        report_filename = "report.json"
    
    report_file = output_dir / report_filename
    
    # Load existing report if resuming
    existing_report = None
    if resume:
        existing_report = load_existing_report(report_file)
        if existing_report:
            print(f"  Resuming: Found existing report with {existing_report.get('successful_samples', 0)} successful samples")
    
    # Save individual outputs
    for result in results:
        if result is None or result.get("error"):
            continue
        
        sample_idx = result["sample_idx"]
        sample_id = result.get("sample_id", str(sample_idx))
        outputs = result["outputs"]
        
        # Use sample_id in filename if it's different from sample_idx, otherwise use sample_idx
        # For unique IDs (like DOLFIN), use the ID; for enumerated indices, use the index
        if sample_id != str(sample_idx) and sample_id:
            # Use sample_id but sanitize it for filename
            safe_id = str(sample_id).replace("/", "_").replace("\\", "_")[:50]  # Limit length
            file_prefix = f"sample_{safe_id}"
        else:
            file_prefix = f"sample_{sample_idx:05d}"
        
        # Save each agent's output
        for agent_id, output in enumerate(outputs):
            output_file = output_dir / f"{file_prefix}_agent_{agent_id}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
    
    # Start with existing report if resuming, otherwise create new
    if resume and existing_report:
        report = existing_report.copy()
        # Get existing sample IDs to avoid duplicates
        existing_sample_ids = get_processed_sample_ids(existing_report)
        
        # Start with existing totals
        total_tokens_input = existing_report.get("summary", {}).get("total_tokens_input", 0)
        total_tokens_output = existing_report.get("summary", {}).get("total_tokens_output", 0)
        total_latency = existing_report.get("summary", {}).get("total_latency_seconds", 0.0)
        
        # Collect existing scores
        chrf_scores = []
        bleu_scores = []
        term_success_rates = []
        for s in existing_report.get("samples", []):
            if not s.get("error"):
                if s.get("chrf_scores"):
                    chrf_scores.append(s["chrf_scores"][-1])  # Use last agent's score
                if s.get("bleu_scores"):
                    last_bleu = s["bleu_scores"][-1]
                    if last_bleu is not None:
                        bleu_scores.append(last_bleu)
                if s.get("term_success_rates"):
                    last_term = s["term_success_rates"][-1]
                    if last_term >= 0:
                        term_success_rates.append(last_term)
    else:
        report = {
            "dataset": dataset_name,
            "workflow": workflow_name,
            "model": model_name,
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "samples": []
        }
        existing_sample_ids = set()
        total_tokens_input = 0
        total_tokens_output = 0
        total_latency = 0.0
        chrf_scores = []
        bleu_scores = []
        term_success_rates = []
    
    # Process new results
    new_samples_count = 0
    skipped_count = 0
    
    for result in results:
        if result is None:
            continue
        
        sample_id = result.get("sample_id", str(result["sample_idx"]))
        lang_pair = result.get("lang_pair", "")
        sample_key = str(sample_id) if sample_id != str(result["sample_idx"]) else f"{lang_pair}_{result['sample_idx']}"
        
        # Skip if already processed
        if sample_key in existing_sample_ids:
            skipped_count += 1
            continue
        
        new_samples_count += 1
        
        sample_data = {
            "sample_idx": result["sample_idx"],
            "sample_id": sample_id,
            "source_lang": result["source_lang"],
            "target_lang": result["target_lang"],
            "lang_pair": lang_pair,
            "error": result.get("error"),
        }
        
        if not result.get("error"):
            sample_data.update({
                "chrf_scores": [e["chrf_score"] for e in result["evaluations"]],
                "bleu_scores": [e.get("bleu_score") for e in result["evaluations"]],
                "term_success_rates": [e.get("term_success_rate", -1.0) for e in result["evaluations"]],
                "tokens_input": result["tokens_input"],
                "tokens_output": result["tokens_output"],
                "latency": result["latency"]
            })
            
            total_tokens_input += result["tokens_input"]
            total_tokens_output += result["tokens_output"]
            if result["latency"]:
                total_latency += result["latency"]
            
            # Collect scores (use last agent's score, which is typically the final translation)
            if result["evaluations"]:
                last_eval = result["evaluations"][-1]
                chrf_scores.append(last_eval["chrf_score"])
                if last_eval.get("bleu_score") is not None:
                    bleu_scores.append(last_eval["bleu_score"])
                if last_eval.get("term_success_rate", -1.0) >= 0:
                    term_success_rates.append(last_eval["term_success_rate"])
        
        report["samples"].append(sample_data)
    
    # Update report statistics
    report["total_samples"] = len(report["samples"])
    report["successful_samples"] = sum(1 for s in report["samples"] if not s.get("error"))
    report["failed_samples"] = sum(1 for s in report["samples"] if s.get("error"))
    
    successful_count = report["successful_samples"]
    report["summary"] = {
        "total_tokens_input": total_tokens_input,
        "total_tokens_output": total_tokens_output,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": total_latency / successful_count if successful_count > 0 else 0,
        "avg_chrf_score": sum(chrf_scores) / len(chrf_scores) if chrf_scores else None,
        "min_chrf_score": min(chrf_scores) if chrf_scores else None,
        "max_chrf_score": max(chrf_scores) if chrf_scores else None,
        "avg_bleu_score": sum(bleu_scores) / len(bleu_scores) if bleu_scores else None,
        "min_bleu_score": min(bleu_scores) if bleu_scores else None,
        "max_bleu_score": max(bleu_scores) if bleu_scores else None,
        "avg_term_success_rate": sum(term_success_rates) / len(term_success_rates) if term_success_rates else None,
        "min_term_success_rate": min(term_success_rates) if term_success_rates else None,
        "max_term_success_rate": max(term_success_rates) if term_success_rates else None,
    }
    
    # Save report
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    if resume and skipped_count > 0:
        print(f"  Skipped {skipped_count} already processed samples")
    if new_samples_count > 0:
        print(f"  Added {new_samples_count} new samples")
    
    print(f"\n✓ Report saved to {report_file}")
    print(f"  Successful: {report['successful_samples']}/{report['total_samples']}")
    if chrf_scores:
        print(f"  Average chrF++: {report['summary']['avg_chrf_score']:.2f}")
    if bleu_scores:
        print(f"  Average BLEU-4: {report['summary']['avg_bleu_score']:.2f}")
    if term_success_rates:
        print(f"  Average Term Success Rate: {report['summary']['avg_term_success_rate']:.4f}")
    print(f"  Total tokens (input/output): {total_tokens_input}/{total_tokens_output}")
    print(f"  Average latency: {report['summary']['avg_latency_seconds']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Run translation experiments")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: 'wmt25' or 'dolfin'")
    parser.add_argument("--workflow", type=str, required=True,
                        help=f"Workflow name (available: {list(WORKFLOW_REGISTRY.keys())})")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, default=None,
                        help=f"Model name (available: {list(model_name2bedrock_id.keys())})")
    model_group.add_argument("--model_arn", type=str, default=None,
                        help=f"Model ARN name (available: {list(model_name2bedrock_arn.keys())})")
    parser.add_argument("--target_languages", type=str, nargs="+", default=None,
                        help="Target languages to process (e.g., 'es de' for dolfin, 'zht' for wmt25). "
                             "If not specified, processes all available.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process per language pair (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (default: outputs/{dataset}_{workflow}_{model})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume experiment: skip already processed samples and merge with existing report")
    parser.add_argument("--use_terminology", action="store_true",
                        help="Use terminology dictionary if available (only for datasets that support it, e.g., wmt25)")
    
    args = parser.parse_args()
    
    # Validate and get model_id/model_arn
    model_id = None
    model_provider = None
    model_name = None
    model_type = "bedrock"  # "bedrock" or "openai"
    
    if args.model:
        # Check if it's an OpenAI model first
        if args.model in model_name2openai_id:
            model_id = model_name2openai_id[args.model]
            model_name = args.model
            model_type = "openai"
            # Check if cdao is available
            try:
                import cdao  # noqa: F401
            except ImportError:
                print(f"Error: OpenAI model '{args.model}' requires 'cdao' library.")
                print(f"  This library is only available in internal environments.")
                print(f"  Please use a Bedrock model instead, or ensure cdao is installed.")
                return 1
        elif args.model in model_name2bedrock_id:
            model_id = model_name2bedrock_id[args.model]
            model_name = args.model
            model_type = "bedrock"
        else:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available Bedrock models: {list(model_name2bedrock_id.keys())}")
            print(f"Available OpenAI models: {list(model_name2openai_id.keys())}")
            return 1
    elif args.model_arn:
        if args.model_arn not in model_name2bedrock_arn:
            print(f"Error: Unknown model ARN '{args.model_arn}'")
            print(f"Available model ARNs: {list(model_name2bedrock_arn.keys())}")
            return 1
        model_id = model_name2bedrock_arn[args.model_arn]
        model_provider = model_name2provider.get(args.model_arn)
        model_name = args.model_arn
        model_type = "bedrock"
    
    # Set AWS_REGION: us-east-1 for ARNs, us-east-2 for model IDs (unless env var is set)
    # Only relevant for Bedrock models
    if model_type == "bedrock":
        if not os.getenv("AWS_REGION"):
            if model_provider:  # ARN case
                AWS_REGION = "us-east-1"
            else:  # Model ID case
                AWS_REGION = "us-east-2"
        else:
            AWS_REGION = os.getenv("AWS_REGION")
    else:
        # OpenAI models don't need AWS_REGION
        AWS_REGION = None
    
    # Validate workflow
    try:
        workflow_module = get_workflow_module(args.workflow)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Determine language pairs/directions to process
    if args.dataset == "wmt25":
        # For WMT25, we process all samples and group by language pair
        # target_languages can filter which pairs to process: "zht" = en->zht, "en" = zht->en
        target_langs = args.target_languages if args.target_languages else None
        
        # Get data loader (without filtering - we'll filter during processing)
        try:
            data_loader = get_data_loader(args.dataset, BASE_DATA_DIR, target_languages=None)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1
        
        print("=" * 80)
        print("Translation Experiment")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Workflow: {args.workflow}")
        print(f"Model: {model_name} ({model_id})")
        if model_provider:
            print(f"Model Provider: {model_provider}")
        print(f"Target languages: {target_langs if target_langs else 'all (both directions)'}")
        print("=" * 80)
        
        # Load samples
        print(f"\nLoading samples from {data_loader.get_dataset_name()}...")
        samples = data_loader.load_samples(max_samples=args.max_samples)
        print(f"Loaded {len(samples)} samples")
        
        if not samples:
            print("No samples to process!")
            return 1
        
        # Group samples by language pair
        samples_by_pair = {}
        for sample in samples:
            source_lang, target_lang = data_loader.get_translation_direction(sample)
            
            # Filter by target language if specified
            if target_langs and target_lang not in target_langs:
                continue
            
            lang_pair = f"{source_lang}-{target_lang}"
            if lang_pair not in samples_by_pair:
                samples_by_pair[lang_pair] = []
            samples_by_pair[lang_pair].append(sample)
        
        print(f"\nFound {len(samples_by_pair)} language pair(s): {list(samples_by_pair.keys())}")
        
        # Process each language pair
        all_results = []
        
        for lang_pair, pair_samples in samples_by_pair.items():
            print(f"\n{'='*80}")
            print(f"Processing language pair: {lang_pair} ({len(pair_samples)} samples)")
            print(f"{'='*80}")
            
            # Determine output directory for this pair
            if args.output_dir:
                pair_output_dir = Path(args.output_dir) / lang_pair
            else:
                pair_output_dir = build_output_dir(
                    dataset=args.dataset,
                    lang_pair=lang_pair,
                    workflow_name=args.workflow,
                    model_name=model_name,
                    use_terminology=args.use_terminology
                )
            print(f"Output directory: {pair_output_dir}")
            
            # Check for existing processed samples if resuming
            processed_sample_ids = set()
            if args.resume:
                report_filename = f"report_{args.max_samples}_samples.json" if args.max_samples else "report.json"
                report_file = pair_output_dir / report_filename
                existing_report = load_existing_report(report_file)
                if existing_report:
                    processed_sample_ids = get_processed_sample_ids(existing_report)
                    print(f"  Found {len(processed_sample_ids)} already processed samples")
            
            # Process samples
            print("\nProcessing samples...")
            results = []
            skipped = 0
            
            for i, sample in enumerate(pair_samples):
                # Check if already processed
                sample_id = sample.get("id") or sample.get("_id") or str(i)
                sample_key = str(sample_id) if sample_id != str(i) else f"{lang_pair}_{i}"
                
                if args.resume and sample_key in processed_sample_ids:
                    skipped += 1
                    continue
                
                result = process_sample(
                    sample=sample,
                    data_loader=data_loader,
                    workflow_module=workflow_module,
                    model_id=model_id,
                    sample_idx=i,
                    lang_pair=lang_pair,
                    use_terminology=args.use_terminology,
                    model_provider=model_provider
                )
                if result:
                    results.append(result)
            
            if skipped > 0:
                print(f"  Skipped {skipped} already processed samples")
            
            # Save outputs and generate report for this pair
            print(f"\nSaving outputs for {lang_pair}...")
            save_outputs(
                results=results,
                output_dir=pair_output_dir,
                dataset_name=f"{args.dataset}_{lang_pair}",
                workflow_name=args.workflow,
                model_name=model_name,
                max_samples=args.max_samples,
                resume=args.resume
            )
            
            all_results.extend(results)
        
        print(f"\n{'='*80}")
        print("All language pairs processed!")
        print(f"{'='*80}")
    
    elif args.dataset == "dolfin":
        # For DOLFIN, get available language pairs
        from data_loaders import get_available_dolfin_lang_pairs
        
        # DOLFIN doesn't support terminology - disable if set
        if args.use_terminology:
            print(f"\n⚠ Warning: DOLFIN dataset does not support terminology dictionaries.")
            print(f"  Disabling --use_terminology flag for this dataset.")
            args.use_terminology = False
        
        dolfin_data_dir = BASE_DATA_DIR / "dolfin"
        print(f"Looking for DOLFIN files in: {dolfin_data_dir}")
        
        available_pairs = get_available_dolfin_lang_pairs(BASE_DATA_DIR)
        
        if not available_pairs:
            print(f"\nError: No DOLFIN language pair files found!")
            print(f"  Searched in: {dolfin_data_dir}")
            print(f"  Expected files matching pattern: dolfin_test_*.jsonl")
            if dolfin_data_dir.exists():
                print(f"  Directory exists. Files found:")
                for f in dolfin_data_dir.glob("*.jsonl"):
                    print(f"    - {f.name}")
            else:
                print(f"  Directory does not exist!")
            return 1
        
        print(f"Found {len(available_pairs)} language pair(s): {available_pairs}")
        
        if args.target_languages:
            # Filter to requested target languages
            target_langs = args.target_languages
            lang_pairs = [f"en_{lang}" for lang in target_langs if f"en_{lang}" in available_pairs]
            if not lang_pairs:
                print(f"Error: No available language pairs for target languages: {target_langs}")
                print(f"Available pairs: {available_pairs}")
                return 1
        else:
            # Process all available pairs
            lang_pairs = available_pairs
            target_langs = [pair.split("_")[1] for pair in lang_pairs]
        
        print("=" * 80)
        print("Translation Experiment")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Workflow: {args.workflow}")
        print(f"Model: {model_name} ({model_id})")
        if model_provider:
            print(f"Model Provider: {model_provider}")
        print(f"Language pairs: {lang_pairs}")
        print("=" * 80)
        
        # Process each language pair
        all_results = []
        
        for lang_pair in lang_pairs:
            print(f"\n{'='*80}")
            print(f"Processing language pair: {lang_pair}")
            print(f"{'='*80}")
            
            # Create loader for this language pair
            from data_loaders import DOLFINDataLoader
            data_loader = DOLFINDataLoader(BASE_DATA_DIR / "dolfin", lang_pair)
            
            # Determine output directory for this pair
            if args.output_dir:
                pair_output_dir = Path(args.output_dir) / lang_pair
            else:
                pair_output_dir = build_output_dir(
                    dataset=args.dataset,
                    lang_pair=lang_pair,
                    workflow_name=args.workflow,
                    model_name=model_name,
                    use_terminology=args.use_terminology
                )
            print(f"Output directory: {pair_output_dir}")
            
            # Check for existing processed samples if resuming
            processed_sample_ids = set()
            if args.resume:
                report_filename = f"report_{args.max_samples}_samples.json" if args.max_samples else "report.json"
                report_file = pair_output_dir / report_filename
                existing_report = load_existing_report(report_file)
                if existing_report:
                    processed_sample_ids = get_processed_sample_ids(existing_report)
                    print(f"  Found {len(processed_sample_ids)} already processed samples")
            
            # Load samples
            print(f"\nLoading samples from {lang_pair}...")
            samples = data_loader.load_samples(max_samples=args.max_samples)
            print(f"Loaded {len(samples)} samples")
            
            if not samples:
                print(f"No samples for {lang_pair}, skipping...")
                continue
            
            # Process samples
            print("\nProcessing samples...")
            results = []
            skipped = 0
            
            for i, sample in enumerate(samples):
                # Check if already processed
                sample_id = sample.get("id") or sample.get("_id") or str(i)
                sample_key = str(sample_id) if sample_id != str(i) else f"{lang_pair}_{i}"
                
                if args.resume and sample_key in processed_sample_ids:
                    skipped += 1
                    continue
                
                result = process_sample(
                    sample=sample,
                    data_loader=data_loader,
                    workflow_module=workflow_module,
                    model_id=model_id,
                    sample_idx=i,
                    lang_pair=lang_pair,
                    use_terminology=args.use_terminology,
                    model_provider=model_provider
                )
                if result:
                    results.append(result)
            
            if skipped > 0:
                print(f"  Skipped {skipped} already processed samples")
            
            # Save outputs and generate report for this pair
            print(f"\nSaving outputs for {lang_pair}...")
            save_outputs(
                results=results,
                output_dir=pair_output_dir,
                dataset_name=f"{args.dataset}_{lang_pair}",
                workflow_name=args.workflow,
                model_name=model_name,
                max_samples=args.max_samples,
                resume=args.resume
            )
            
            all_results.extend(results)
        
        print(f"\n{'='*80}")
        print("All language pairs processed!")
        print(f"{'='*80}")
    
    else:
        print(f"Error: Unknown dataset '{args.dataset}'. Available: 'wmt25', 'dolfin'")
        return 1
    
    if args.dataset == "wmt25":
        print("\n" + "=" * 80)
        print("Experiment completed!")
        print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


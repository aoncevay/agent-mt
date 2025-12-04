#!/usr/bin/env python3
"""
Main runner script for translation experiments.

Usage:
    python src/run.py --dataset wmt25 --workflow single_agent --model qwen3-235b
    python src/run.py --dataset dolfin_en_es --workflow single_agent_term --model qwen3-235b --max_samples 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loaders import get_data_loader
from workflows import WORKFLOW_REGISTRY
from evaluation import compute_chrf
from vars import model_name2bedrock_id

# Load environment variables
load_dotenv("config.env")

# Configuration
BASE_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "outputs"
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MAX_RETRIES = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.getenv("BEDROCK_INITIAL_BACKOFF", "2.0"))


def get_workflow_module(workflow_name: str):
    """Import and return workflow module."""
    if workflow_name not in WORKFLOW_REGISTRY:
        raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(WORKFLOW_REGISTRY.keys())}")
    
    module_name = WORKFLOW_REGISTRY[workflow_name]
    
    # Import workflow module
    if workflow_name == "single_agent":
        from workflows import single_agent
        return single_agent
    elif workflow_name == "single_agent_term":
        from workflows import single_agent_term
        return single_agent_term
    else:
        # For future workflows
        return __import__(f"workflows.{module_name}", fromlist=[module_name])


def process_sample(
    sample: Dict[str, Any],
    data_loader,
    workflow_module,
    model_id: str,
    sample_idx: int
) -> Dict[str, Any]:
    """Process a single sample through the workflow."""
    # Get translation direction
    source_lang, target_lang = data_loader.get_translation_direction(sample)
    
    # Skip if filtered out (None, None)
    if source_lang is None or target_lang is None:
        return None
    
    # Extract texts
    source_text, reference_text, terminology = data_loader.extract_texts(
        sample, source_lang, target_lang
    )
    
    if not source_text or not reference_text:
        return None
    
    print(f"  Sample {sample_idx + 1}: {source_lang}->{target_lang} "
          f"({len(source_text)} chars source, {len(reference_text)} chars reference)")
    
    try:
        # Run workflow
        result = workflow_module.run_workflow(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_id=model_id,
            terminology=terminology,
            region=AWS_REGION,
            max_retries=MAX_RETRIES,
            initial_backoff=INITIAL_BACKOFF
        )
        
        # Evaluate each output
        outputs = result["outputs"]
        evaluations = []
        
        for i, output in enumerate(outputs):
            chrf_result = compute_chrf(output, reference_text)
            evaluations.append({
                "agent_id": i,
                "chrf_score": chrf_result["score"],
                "translation": output
            })
        
        return {
            "sample_idx": sample_idx,
            "source_lang": source_lang,
            "target_lang": target_lang,
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
        return {
            "sample_idx": sample_idx,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_text": source_text,
            "reference_text": reference_text,
            "outputs": [],
            "evaluations": [],
            "tokens_input": 0,
            "tokens_output": 0,
            "latency": None,
            "error": str(e)
        }


def save_outputs(
    results: List[Dict[str, Any]],
    output_dir: Path,
    dataset_name: str,
    workflow_name: str,
    model_name: str
):
    """Save translation outputs and generate report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual outputs
    for result in results:
        if result is None or result.get("error"):
            continue
        
        sample_idx = result["sample_idx"]
        outputs = result["outputs"]
        
        # Save each agent's output
        for agent_id, output in enumerate(outputs):
            output_file = output_dir / f"sample_{sample_idx:05d}_agent_{agent_id}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
    
    # Generate summary report
    report = {
        "dataset": dataset_name,
        "workflow": workflow_name,
        "model": model_name,
        "total_samples": len(results),
        "successful_samples": sum(1 for r in results if r and not r.get("error")),
        "failed_samples": sum(1 for r in results if r and r.get("error")),
        "samples": []
    }
    
    total_tokens_input = 0
    total_tokens_output = 0
    total_latency = 0.0
    chrf_scores = []
    
    for result in results:
        if result is None:
            continue
        
        sample_data = {
            "sample_idx": result["sample_idx"],
            "source_lang": result["source_lang"],
            "target_lang": result["target_lang"],
            "error": result.get("error"),
        }
        
        if not result.get("error"):
            sample_data.update({
                "chrf_scores": [e["chrf_score"] for e in result["evaluations"]],
                "tokens_input": result["tokens_input"],
                "tokens_output": result["tokens_output"],
                "latency": result["latency"]
            })
            
            total_tokens_input += result["tokens_input"]
            total_tokens_output += result["tokens_output"]
            if result["latency"]:
                total_latency += result["latency"]
            
            # Collect chrF scores (use first agent's score for now)
            if result["evaluations"]:
                chrf_scores.append(result["evaluations"][0]["chrf_score"])
        
        report["samples"].append(sample_data)
    
    # Add summary statistics
    report["summary"] = {
        "total_tokens_input": total_tokens_input,
        "total_tokens_output": total_tokens_output,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": total_latency / len([r for r in results if r and not r.get("error")]) if report["successful_samples"] > 0 else 0,
        "avg_chrf_score": sum(chrf_scores) / len(chrf_scores) if chrf_scores else None,
        "min_chrf_score": min(chrf_scores) if chrf_scores else None,
        "max_chrf_score": max(chrf_scores) if chrf_scores else None,
    }
    
    # Save report
    report_file = output_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved to {report_file}")
    print(f"  Successful: {report['successful_samples']}/{report['total_samples']}")
    if chrf_scores:
        print(f"  Average chrF++: {report['summary']['avg_chrf_score']:.2f}")
    print(f"  Total tokens (input/output): {total_tokens_input}/{total_tokens_output}")
    print(f"  Average latency: {report['summary']['avg_latency_seconds']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Run translation experiments")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: 'wmt25' or 'dolfin'")
    parser.add_argument("--workflow", type=str, required=True,
                        help=f"Workflow name (available: {list(WORKFLOW_REGISTRY.keys())})")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name (available: {list(model_name2bedrock_id.keys())})")
    parser.add_argument("--target_languages", type=str, nargs="+", default=None,
                        help="Target languages to process (e.g., 'es de' for dolfin, 'zht' for wmt25). "
                             "If not specified, processes all available.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process per language pair (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (default: outputs/{dataset}_{workflow}_{model})")
    
    args = parser.parse_args()
    
    # Validate model
    if args.model not in model_name2bedrock_id:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(model_name2bedrock_id.keys())}")
        return 1
    
    model_id = model_name2bedrock_id[args.model]
    
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
        
        # Determine base output directory
        if args.output_dir:
            base_output_dir = Path(args.output_dir)
        else:
            base_output_dir = OUTPUT_BASE_DIR / f"{args.dataset}.{args.workflow}.{args.model}"
        
        print("=" * 80)
        print("Translation Experiment")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Workflow: {args.workflow}")
        print(f"Model: {args.model} ({model_id})")
        print(f"Target languages: {target_langs if target_langs else 'all (both directions)'}")
        print(f"Output directory: {base_output_dir}")
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
            pair_output_dir = base_output_dir / lang_pair
            
            # Process samples
            print("\nProcessing samples...")
            results = []
            
            for i, sample in enumerate(pair_samples):
                result = process_sample(
                    sample=sample,
                    data_loader=data_loader,
                    workflow_module=workflow_module,
                    model_id=model_id,
                    sample_idx=i
                )
                if result:
                    results.append(result)
            
            # Save outputs and generate report for this pair
            print(f"\nSaving outputs for {lang_pair}...")
            save_outputs(
                results=results,
                output_dir=pair_output_dir,
                dataset_name=f"{args.dataset}_{lang_pair}",
                workflow_name=args.workflow,
                model_name=args.model
            )
            
            all_results.extend(results)
        
        print(f"\n{'='*80}")
        print("All language pairs processed!")
        print(f"{'='*80}")
    
    elif args.dataset == "dolfin":
        # For DOLFIN, get available language pairs
        from data_loaders import get_available_dolfin_lang_pairs
        
        available_pairs = get_available_dolfin_lang_pairs(BASE_DATA_DIR)
        
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
        print(f"Model: {args.model} ({model_id})")
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
                pair_output_dir = OUTPUT_BASE_DIR / f"{args.dataset}.{args.workflow}.{args.model}" / lang_pair
            
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
            
            for i, sample in enumerate(samples):
                result = process_sample(
                    sample=sample,
                    data_loader=data_loader,
                    workflow_module=workflow_module,
                    model_id=model_id,
                    sample_idx=i
                )
                if result:
                    results.append(result)
            
            # Save outputs and generate report for this pair
            print(f"\nSaving outputs for {lang_pair}...")
            save_outputs(
                results=results,
                output_dir=pair_output_dir,
                dataset_name=f"{args.dataset}_{lang_pair}",
                workflow_name=args.workflow,
                model_name=args.model
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


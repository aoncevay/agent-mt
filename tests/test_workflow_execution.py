#!/usr/bin/env python3
"""
Test script to run actual workflow executions on a small test sample.
This verifies that workflows can execute end-to-end and produce outputs.

Usage:
    python tests/test_workflow_execution.py --workflow ADT_multi_agents --model qwen3-32b
    python tests/test_workflow_execution.py --workflow all --model qwen3-32b
"""

import argparse
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv("config.env")

from workflows import WORKFLOW_REGISTRY
from translation import create_bedrock_llm
from vars import model_name2bedrock_id

# Test data
TEST_SOURCE_TEXT = "The United Nations (UN) is an international organization. It was founded in 1945. The UN promotes peace and security worldwide."
TEST_TERMINOLOGY = {
    "United Nations": ["联合国", "UN"],
    "international organization": ["国际组织"],
    "peace and security": ["和平与安全"]
}
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")


def test_workflow(workflow_name: str, model_id: str, use_terminology: bool = False):
    """Test a single workflow execution."""
    print("\n" + "="*80)
    print(f"Testing workflow: {workflow_name}")
    print(f"Model: {model_id}")
    print(f"Use terminology: {use_terminology}")
    print("="*80)
    
    if workflow_name not in WORKFLOW_REGISTRY:
        print(f"✗ Unknown workflow: {workflow_name}")
        return False
    
    try:
        # Import workflow module
        import importlib
        module_name = WORKFLOW_REGISTRY[workflow_name]
        workflow_module = importlib.import_module(f"workflows.{module_name}")
        
        # Prepare arguments
        workflow_args = {
            "source_text": TEST_SOURCE_TEXT,
            "source_lang": "en",
            "target_lang": "zh",
            "model_id": model_id,
            "terminology": TEST_TERMINOLOGY if use_terminology else None,
            "use_terminology": use_terminology,
            "region": AWS_REGION,
            "max_retries": 2,  # Reduced for testing
            "initial_backoff": 1.0
        }
        
        # Check if workflow supports reference parameter
        import inspect
        sig = inspect.signature(workflow_module.run_workflow)
        if "reference" in sig.parameters:
            workflow_args["reference"] = None
        
        # Run workflow
        print(f"\nRunning workflow...")
        result = workflow_module.run_workflow(**workflow_args)
        
        # Validate result structure
        assert "outputs" in result, "Result missing 'outputs' key"
        assert "tokens_input" in result, "Result missing 'tokens_input' key"
        assert "tokens_output" in result, "Result missing 'tokens_output' key"
        assert "latency" in result, "Result missing 'latency' key"
        
        # Validate outputs
        assert isinstance(result["outputs"], list), "Outputs should be a list"
        assert len(result["outputs"]) > 0, "Outputs list should not be empty"
        
        # Check that outputs are non-empty strings
        for i, output in enumerate(result["outputs"]):
            assert isinstance(output, str), f"Output {i} should be a string"
            assert len(output.strip()) > 0, f"Output {i} should not be empty"
        
        # Validate metrics
        assert result["tokens_input"] >= 0, "tokens_input should be non-negative"
        assert result["tokens_output"] >= 0, "tokens_output should be non-negative"
        assert result["latency"] >= 0, "latency should be non-negative"
        
        print(f"\n✓ Workflow execution successful!")
        print(f"  Outputs: {len(result['outputs'])} agent(s)")
        print(f"  Tokens: {result['tokens_input']} in, {result['tokens_output']} out")
        print(f"  Latency: {result['latency']:.2f}s")
        
        # Print first 200 chars of each output
        for i, output in enumerate(result["outputs"]):
            preview = output[:200].replace("\n", " ")
            print(f"  Output {i+1}: {preview}...")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Validation failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test workflow executions")
    parser.add_argument("--workflow", type=str, required=True,
                        help=f"Workflow name to test (or 'all' for all workflows). Available: {list(WORKFLOW_REGISTRY.keys())}")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name. Available: {list(model_name2bedrock_id.keys())}")
    parser.add_argument("--use_terminology", action="store_true",
                        help="Test with terminology enabled")
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                        help="Workflows to skip (e.g., --skip zero_shot zero_shot_term)")
    
    args = parser.parse_args()
    
    # Validate model
    if args.model not in model_name2bedrock_id:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(model_name2bedrock_id.keys())}")
        return 1
    
    model_id = model_name2bedrock_id[args.model]
    
    # Determine workflows to test
    if args.workflow.lower() == "all":
        workflows_to_test = [w for w in WORKFLOW_REGISTRY.keys() if w not in args.skip]
    else:
        workflows_to_test = [args.workflow]
    
    print("="*80)
    print("WORKFLOW EXECUTION TESTS")
    print("="*80)
    print(f"\nTesting {len(workflows_to_test)} workflow(s)")
    print(f"Model: {args.model} ({model_id})")
    print(f"Use terminology: {args.use_terminology}")
    print(f"Test source text: {TEST_SOURCE_TEXT[:100]}...")
    
    results = {}
    for workflow_name in workflows_to_test:
        success = test_workflow(workflow_name, model_id, args.use_terminology)
        results[workflow_name] = success
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    for workflow_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {workflow_name}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())


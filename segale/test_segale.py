#!/usr/bin/env python3
"""
Test script for SEGALE evaluation framework using COMET metrics.

This script:
1. Creates sample JSONL files in SEGALE format from our experiment outputs
2. Tests segale-align functionality
3. Tests segale-eval functionality (COMET only)

Usage:
    python segale/test_segale.py --output-dir outputs/wmt25/en-es/IRB/gpt-4-1
    python segale/test_segale.py --output-dir outputs/wmt25/en-es/IRB/gpt-4-1 --use-docker
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Get project root (segale folder is at project root level)
PROJECT_ROOT = Path(__file__).parent.parent
SEGALE_DIR = PROJECT_ROOT / "other_repos" / "SEGALE"


def load_report(report_file: Path) -> Dict[str, Any]:
    """Load a report.json file."""
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_jsonl(file_path: Path) -> bool:
    """Validate that a JSONL file is properly formatted."""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Check required fields
                    if 'doc_id' not in entry or 'src' not in entry or 'tgt' not in entry:
                        print(f"⚠ Warning: Line {line_num} missing required fields")
                        return False
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"✗ Error: Invalid JSON on line {line_num}: {e}")
                    return False
        print(f"  Validated {count} entries")
        return True
    except Exception as e:
        print(f"✗ Error validating JSONL: {e}")
        return False


def get_latest_agent_output(output_dir: Path, sample_id: str, sample_idx: int) -> Optional[str]:
    """
    Get the final translation output from the latest agent's text file.
    
    Args:
        output_dir: Directory containing the sample output files
        sample_id: Sample ID (may be different from sample_idx)
        sample_idx: Sample index
    
    Returns:
        Content of the latest agent's output file, or None if not found
    """
    # Determine file prefix (same logic as in save_outputs)
    if sample_id != str(sample_idx) and sample_id:
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")[:50]
        file_prefix = f"sample_{safe_id}"
    else:
        file_prefix = f"sample_{sample_idx:05d}"
    
    # Find all agent files for this sample
    pattern = f"{file_prefix}_agent_*.txt"
    agent_files = list(output_dir.glob(pattern))
    
    if not agent_files:
        return None
    
    # Extract agent IDs and find the latest one
    agent_ids = []
    for file in agent_files:
        # Extract agent ID from filename: sample_XXXXX_agent_N.txt -> N
        try:
            parts = file.stem.split('_agent_')
            if len(parts) == 2:
                agent_id = int(parts[1])
                agent_ids.append((agent_id, file))
        except (ValueError, IndexError):
            continue
    
    if not agent_ids:
        return None
    
    # Get the file with the highest agent_id (latest agent)
    latest_agent_id, latest_file = max(agent_ids, key=lambda x: x[0])
    
    # Read the content
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except (IOError, UnicodeDecodeError) as e:
        print(f"⚠ Warning: Could not read {latest_file}: {e}")
        return None


def create_segale_jsonl(
    report_data: Dict[str, Any],
    output_file: Path,
    output_dir: Path,
    file_type: str = "system"
) -> None:
    """
    Create a SEGALE-compatible JSONL file from report data.
    
    Args:
        report_data: Report data from report.json
        output_file: Path to output JSONL file
        output_dir: Directory containing sample output text files
        file_type: "system" or "reference"
    """
    samples = report_data.get('samples', [])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Skip samples with errors
            if sample.get('error'):
                continue
            
            sample_id = sample.get('sample_id', sample.get('sample_idx', 'unknown'))
            sample_idx = sample.get('sample_idx', 0)
            doc_id = f"doc_{sample_id}"
            
            if file_type == "system":
                # Get the final output from the latest agent's text file
                tgt_text = get_latest_agent_output(output_dir, str(sample_id), sample_idx)
                if tgt_text is None:
                    print(f"⚠ Warning: Could not find output file for sample {sample_id}, skipping")
                    continue
                
                src_text = sample.get('source_text', '')
                
                entry = {
                    "doc_id": doc_id,
                    "sys_id": report_data.get('model', 'unknown'),
                    "src": src_text,
                    "tgt": tgt_text,
                    "seg_id": 0  # Single document, no segmentation
                }
            else:  # reference
                ref_text = sample.get('reference_text', '')
                src_text = sample.get('source_text', '')
                
                entry = {
                    "doc_id": doc_id,
                    "src": src_text,
                    "tgt": ref_text,  # In ref files, tgt is the reference
                    "seg_id": 0
                }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Created {file_type} JSONL file: {output_file}")
    valid_samples = len([s for s in samples if not s.get('error')])
    print(f"  Samples: {valid_samples}")
    
    # Validate the created file
    if not validate_jsonl(output_file):
        print(f"✗ Warning: {file_type} JSONL file validation failed")


def run_segale_align(
    system_file: Path,
    ref_file: Path,
    output_dir: Path,
    segmenter: str = "spacy",
    task_lang: str = "en",
    proc_device: str = "cpu",
    use_docker: bool = False,
    embedding_model: Optional[str] = None
) -> Path:
    """
    Run segale-align command.
    
    Returns:
        Path to aligned output file
    """
    # SEGALE creates output in a folder named after the system file (without extension)
    system_name = system_file.stem
    aligned_dir = system_file.parent / system_name
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    if use_docker:
        # Run in Docker
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{SEGALE_DIR}:/workspace",
            "-v", f"{system_file.parent}:/data",
            "-w", "/workspace",
            "segale:latest",  # Assuming image is built
            "segale-align",
            "--system_file", f"/data/{system_file.name}",
            "--ref_file", f"/data/{ref_file.name}",
            "--segmenter", segmenter,
            "--task_lang", task_lang,
            "--proc_device", proc_device,
            "-v"
        ]
        print(f"Running: {' '.join(docker_cmd)}")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
    else:
        # Run directly
        cmd = [
            "segale-align",
            "--system_file", str(system_file),
            "--ref_file", str(ref_file),
            "--segmenter", segmenter,
            "--task_lang", task_lang,
            "--proc_device", proc_device,
            "-v"
        ]
        # Add embedding model if specified (allows working without laser_encoders)
        if embedding_model:
            cmd.extend(["--embedding_model", embedding_model])
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=SEGALE_DIR)
    
    if result.returncode != 0:
        print(f"✗ Error running segale-align:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Find the aligned output file
    # SEGALE creates: aligned_{segmenter}_{system_name}.jsonl
    aligned_file = aligned_dir / f"aligned_{segmenter}_{system_name}.jsonl"
    if not aligned_file.exists():
        # Try alternative naming patterns
        aligned_files = list(aligned_dir.glob("aligned_*.jsonl"))
        if aligned_files:
            aligned_file = aligned_files[0]
            print(f"⚠ Found alternative aligned file: {aligned_file}")
        else:
            print(f"✗ Could not find aligned output file in {aligned_dir}")
            print(f"  Looking for: aligned_{segmenter}_{system_name}.jsonl")
            print(f"  Available files: {list(aligned_dir.glob('*'))}")
            return None
    
    print(f"✓ Alignment complete: {aligned_file}")
    return aligned_file


def run_segale_eval(
    aligned_file: Path,
    use_docker: bool = False
) -> Dict[str, Path]:
    """
    Run segale-eval command (COMET metrics only).
    
    Returns:
        Dictionary with paths to eval and result files
    """
    if use_docker:
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{SEGALE_DIR}:/workspace",
            "-v", f"{aligned_file.parent}:/data",
            "-w", "/workspace",
            "segale:latest",
            "segale-eval",
            "--input_file", f"/data/{aligned_file.name}"
        ]
        print(f"Running: {' '.join(docker_cmd)}")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
    else:
        cmd = [
            "segale-eval",
            "--input_file", str(aligned_file)
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=SEGALE_DIR)
    
    if result.returncode != 0:
        print(f"✗ Error running segale-eval:")
        print(result.stderr)
        return {}
    
    print(result.stdout)
    
    # Find output files
    eval_file = aligned_file.parent / f"eval_{aligned_file.name}"
    result_file = aligned_file.parent / f"result_{aligned_file.name}"
    
    output_files = {}
    if eval_file.exists():
        output_files['eval'] = eval_file
        print(f"✓ Evaluation file: {eval_file}")
    if result_file.exists():
        output_files['result'] = result_file
        print(f"✓ Result file: {result_file}")
    
    return output_files


def get_target_lang_code(lang_pair: str) -> str:
    """Get target language code for spaCy segmentation."""
    # Map common language pairs to spaCy language codes
    lang_map = {
        'es': 'es',  # Spanish
        'zh': 'zh',  # Chinese
        'zht': 'zh',  # Traditional Chinese
        'ja': 'ja',  # Japanese
        'de': 'de',  # German
        'fr': 'fr',  # French
    }
    
    # Extract target language from lang_pair (e.g., "en-es" -> "es")
    if '-' in lang_pair:
        target = lang_pair.split('-')[1]
        return lang_map.get(target, 'en')  # Default to English
    
    return 'en'


def check_docker_setup() -> bool:
    """Check if Docker is available and the SEGALE image exists."""
    # Check if docker command exists
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("✗ Docker is not installed or not in PATH")
            return False
    except FileNotFoundError:
        print("✗ Docker is not installed. Please install Docker Desktop (macOS) or docker.io (Linux)")
        return False
    
    # Check if SEGALE image exists
    result = subprocess.run(
        ["docker", "images", "-q", "segale:latest"],
        capture_output=True,
        text=True
    )
    if not result.stdout.strip():
        print("✗ SEGALE Docker image not found. Please build it first:")
        print("  cd other_repos/SEGALE")
        print("  docker build -t segale:latest .")
        return False
    
    print("✓ Docker is available and SEGALE image exists")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test SEGALE evaluation framework with COMET metrics"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to experiment output directory (e.g., outputs/wmt25/en-es/IRB/gpt-4-1)"
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="spacy",
        choices=["spacy", "ersatz"],
        help="Sentence segmenter to use (default: spacy)"
    )
    parser.add_argument(
        "--proc-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Processing device (default: cpu)"
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Run SEGALE in Docker container"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Alternative embedding model (HuggingFace model name) to use instead of laser_encoders. "
             "Examples: 'BAAI/bge-m3', 'intfloat/multilingual-e5-large'. "
             "If not provided, will try to use laser_encoders (requires installation)."
    )
    
    args = parser.parse_args()
    
    # Check Docker setup if requested
    if args.use_docker:
        if not check_docker_setup():
            print("\nSee segale/DOCKER_SETUP.md for detailed Docker installation instructions.")
            return 1
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"✗ Error: Output directory does not exist: {output_dir}")
        return 1
    
    # Find report.json
    report_file = output_dir / "report.json"
    if not report_file.exists():
        print(f"✗ Error: report.json not found in {output_dir}")
        return 1
    
    print(f"Loading report from: {report_file}")
    report_data = load_report(report_file)
    
    # Extract language pair from path
    parts = output_dir.parts
    try:
        outputs_idx = parts.index('outputs')
        lang_pair = parts[outputs_idx + 2]  # e.g., "en-es"
        task_lang = get_target_lang_code(lang_pair)
    except (ValueError, IndexError):
        print("⚠ Warning: Could not determine language pair, defaulting to 'en'")
        task_lang = "en"
    
    print(f"Language pair: {lang_pair}, Target language for segmentation: {task_lang}")
    print("Using COMET-DA (reference-based) for evaluation")
    
    # Create test directory
    test_dir = output_dir / "segale_test"
    test_dir.mkdir(exist_ok=True)
    
    # Create JSONL files
    system_file = test_dir / "system.jsonl"
    ref_file = test_dir / "reference.jsonl"
    
    print("\n" + "="*80)
    print("Step 1: Creating SEGALE-compatible JSONL files")
    print("="*80)
    create_segale_jsonl(report_data, system_file, output_dir, file_type="system")
    create_segale_jsonl(report_data, ref_file, output_dir, file_type="reference")
    
    # Limit samples if requested
    if args.max_samples:
        print(f"\n⚠ Limiting to first {args.max_samples} samples for testing")
        # Recreate files with limited samples
        limited_data = report_data.copy()
        limited_data['samples'] = report_data['samples'][:args.max_samples]
        create_segale_jsonl(limited_data, system_file, output_dir, file_type="system")
        create_segale_jsonl(limited_data, ref_file, output_dir, file_type="reference")
    
    # Run segale-align
    print("\n" + "="*80)
    print("Step 2: Running segale-align")
    print("="*80)
    aligned_file = run_segale_align(
        system_file,
        ref_file,
        test_dir,
        segmenter=args.segmenter,
        task_lang=task_lang,
        proc_device=args.proc_device,
        use_docker=args.use_docker,
        embedding_model=args.embedding_model
    )
    
    if not aligned_file:
        print("✗ Alignment failed, stopping.")
        return 1
    
    # Run segale-eval
    print("\n" + "="*80)
    print("Step 3: Running segale-eval (COMET-DA reference-based)")
    print("="*80)
    eval_files = run_segale_eval(aligned_file, use_docker=args.use_docker)
    
    if not eval_files:
        print("✗ Evaluation failed, stopping.")
        return 1
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"✓ System JSONL: {system_file}")
    print(f"✓ Reference JSONL: {ref_file}")
    print(f"✓ Aligned file: {aligned_file}")
    for key, path in eval_files.items():
        print(f"✓ {key.capitalize()} file: {path}")
    
    print("\n✓ SEGALE test completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Review the aligned file: {aligned_file}")
    print(f"2. Review evaluation results: {eval_files.get('result', 'N/A')}")
    print(f"3. Check COMET-DA scores in the result file")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


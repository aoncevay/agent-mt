"""
Utility functions for metrics evaluation.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


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


def load_report(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load report.json from output directory.
    
    Args:
        output_dir: Output directory containing report.json
    
    Returns:
        Report data dictionary, or None if not found
    """
    report_file = output_dir / "report.json"
    if not report_file.exists():
        return None
    
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠ Warning: Could not read report.json: {e}")
        return None


def parse_system_name_from_path(output_dir: Path) -> Optional[str]:
    """
    Parse system name (workflow+model) from output directory path.
    
    Args:
        output_dir: Output directory path (e.g., outputs/wmt25/en-zht/IRB.term/gpt-4-1)
    
    Returns:
        System name (e.g., "IRB+gpt-4-1") or None
    """
    try:
        parts = output_dir.parts
        # Find the model name (last part)
        if len(parts) >= 1:
            model_name = parts[-1]
            # Find workflow acronym (second to last, may have .term suffix)
            if len(parts) >= 2:
                workflow_dir = parts[-2]
                workflow_acronym = workflow_dir.replace('.term', '')
                # Combine workflow + model (e.g., "IRB+gpt-4-1")
                return f"{workflow_acronym}+{model_name}"
    except (IndexError, AttributeError):
        pass
    return None


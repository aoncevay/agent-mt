"""
Mapping of workflow names to their acronyms for output directory naming.
"""

from pathlib import Path
from typing import Optional

# Workflow name to acronym mapping
WORKFLOW_ACRONYMS = {
    "zero_shot": "ZS",
    "zero_shot_term": "ZS",
    "MaMT_translate_postedit": "MaMT",
    "MaMT_translate_postedit_proofread": "MaMT",
    "SbS_step_by_step": "SbS",
    "SbS_chat_step_by_step": "SbS_chat",
    "MAATS_multi_agents": "MAATS_multi",
    "MAATS_single_agent": "MAATS_single",
    "IRB_refine": "IRB",
    "DeLTA_multi_agents": "DeLTA",
    "ADT_multi_agents": "ADT"
}


def get_workflow_acronym(workflow_name: str) -> str:
    """Get acronym for a workflow name."""
    return WORKFLOW_ACRONYMS.get(workflow_name, workflow_name)


def build_output_dir(
    dataset: str,
    lang_pair: str,
    workflow_name: str,
    model_name: str,
    use_terminology: bool = False,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Build output directory path following the structure:
    outputs/{dataset}/{lang_pair}/{workflow_acronym}{.term}/{model}
    
    Args:
        dataset: Dataset name (e.g., "wmt25", "dolfin")
        lang_pair: Language pair (e.g., "en-zh", "en-es")
        workflow_name: Workflow name (e.g., "ADT_multi_agents")
        model_name: Model name (e.g., "qwen3-235b")
        use_terminology: Whether terminology is enabled
        base_dir: Base output directory (defaults to outputs/ relative to project root)
    
    Returns:
        Path to the output directory: outputs/{dataset}/{lang_pair}/{acronym}{.term}/{model}
    """
    if base_dir is None:
        # Compute base dir relative to this file to avoid circular imports
        # This file is in src/, so go up one level to project root, then to outputs/
        base_dir = Path(__file__).parent.parent / "outputs"
        base_dir = base_dir.resolve()
    
    # Get workflow acronym
    acronym = get_workflow_acronym(workflow_name)
    
    # Build workflow directory name: acronym or acronym.term
    if use_terminology:
        workflow_dir = f"{acronym}.term"
    else:
        workflow_dir = acronym
    
    # Full path: outputs/{dataset}/{lang_pair}/{acronym}{.term}/{model}
    return base_dir / dataset / lang_pair / workflow_dir / model_name


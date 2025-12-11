"""
Workflow modules for different agentic translation configurations.
"""

from typing import Dict, Any, Optional, List

# Workflow name to module mapping
WORKFLOW_REGISTRY = {
    "zero_shot": "zero_shot",
    "zero_shot_term": "zero_shot_term",
    "MaMT_translate_postedit": "MaMT_translate_postedit",
    "MaMT_translate_postedit_proofread": "MaMT_translate_postedit_proofread",
    "MaMT_translate_postedit_proofread_term": "MaMT_translate_postedit_proofread_term",
    "SbS_step_by_step": "SbS_step_by_step",
    "SbS_chat_step_by_step": "SbS_chat_step_by_step",
    "SbS_chat_step_by_step_term": "SbS_chat_step_by_step_term",
    "MAATS_multi_agents": "MAATS_multi_agents",
    "MAATS_multi_agents_term": "MAATS_multi_agents_term",
    "MAATS_single_agent": "MAATS_single_agent",
    "MAATS_single_agent_term": "MAATS_single_agent_term",
    "IRB_refine": "IRB_refine",
    "IRB_refine_term": "IRB_refine_term",
    "DeLTA_multi_agents": "DeLTA_multi_agents"
}


def get_workflow(workflow_name: str):
    """Get workflow module by name."""
    if workflow_name not in WORKFLOW_REGISTRY:
        raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(WORKFLOW_REGISTRY.keys())}")
    
    module_name = WORKFLOW_REGISTRY[workflow_name]
    return __import__(f"workflows.{module_name}", fromlist=[module_name])


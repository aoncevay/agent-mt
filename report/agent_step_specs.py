"""
Workflow step specifications for agent contribution analysis.

Each step spec tells the analyzer how to derive the translation state after a step.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StepSpec:
    """One workflow step in the observable output sequence."""

    step_name: str
    agent_type: str
    output_index: int
    translation_mode: str  # one of: direct, carry, none, append


def _clip_specs_to_outputs(specs: List[StepSpec], num_outputs: int) -> List[StepSpec]:
    """Drop specs that reference output indices that do not exist."""
    return [s for s in specs if 0 <= s.output_index < num_outputs]


def _generic_specs(num_outputs: int) -> List[StepSpec]:
    return [
        StepSpec(
            step_name=f"agent_{i}",
            agent_type="generic_agent",
            output_index=i,
            translation_mode="direct",
        )
        for i in range(num_outputs)
    ]


def _adt_specs(num_outputs: int) -> List[StepSpec]:
    # ADT returns [discourse_1, discourse_2, ..., discourse_n, final_concatenated].
    if num_outputs <= 0:
        return []
    if num_outputs == 1:
        return [
            StepSpec(
                step_name="final_translation",
                agent_type="aggregation",
                output_index=0,
                translation_mode="direct",
            )
        ]

    specs: List[StepSpec] = []
    for i in range(num_outputs - 1):
        specs.append(
            StepSpec(
                step_name=f"discourse_{i + 1}_translation",
                agent_type="translation_agent_discourse",
                output_index=i,
                translation_mode="append",
            )
        )
    specs.append(
        StepSpec(
            step_name="final_concatenation",
            agent_type="aggregation",
            output_index=num_outputs - 1,
            translation_mode="direct",
        )
    )
    return specs


def resolve_workflow_step_specs(workflow_name: str, num_outputs: int) -> List[StepSpec]:
    """
    Resolve workflow-specific step specs using the workflow name from report.json.
    """
    if num_outputs <= 0:
        return []

    if workflow_name in {"zero_shot", "zero_shot_term"}:
        specs = [
            StepSpec("translate", "translate", 0, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "MaMT_translate_postedit":
        specs = [
            StepSpec("translate", "translate", 0, "direct"),
            StepSpec("postedit", "postedit", 1, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "MaMT_translate_postedit_proofread":
        specs = [
            StepSpec("translate", "translate", 0, "direct"),
            StepSpec("postedit", "postedit", 1, "direct"),
            StepSpec("proofread", "proofread", 2, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "IRB_refine":
        specs = [
            StepSpec("initial_translation", "translate", 0, "direct"),
            StepSpec("refinement", "refine", 1, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "MAATS_multi_agents":
        specs = [
            StepSpec("zero_shot_translation", "translate", 0, "direct"),
            StepSpec("terminology_eval", "mqm_evaluator", 1, "carry"),
            StepSpec("accuracy_eval", "mqm_evaluator", 2, "carry"),
            StepSpec("linguistic_conventions_eval", "mqm_evaluator", 3, "carry"),
            StepSpec("locale_conventions_eval", "mqm_evaluator", 4, "carry"),
            StepSpec("design_and_markup_eval", "mqm_evaluator", 5, "carry"),
            StepSpec("style_eval", "mqm_evaluator", 6, "carry"),
            StepSpec("audience_appropriateness_eval", "mqm_evaluator", 7, "carry"),
            StepSpec("refinement", "refine", 8, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "MAATS_single_agent":
        specs = [
            StepSpec("zero_shot_translation", "translate", 0, "direct"),
            StepSpec("mqm_evaluation", "mqm_evaluator", 1, "carry"),
            StepSpec("refinement", "refine", 2, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "SbS_step_by_step":
        specs = [
            StepSpec("research", "research", 0, "none"),
            StepSpec("draft", "draft", 1, "direct"),
            StepSpec("refinement", "refine", 2, "direct"),
            StepSpec("proofread", "proofread", 3, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "SbS_chat_step_by_step":
        specs = [
            StepSpec("research", "research", 0, "none"),
            StepSpec("draft", "draft", 1, "direct"),
            StepSpec("refinement", "refine", 2, "direct"),
            StepSpec("proofread", "proofread", 3, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "DeLTA_multi_agents":
        specs = [
            StepSpec("final_translation", "translate", 0, "direct"),
        ]
        return _clip_specs_to_outputs(specs, num_outputs)

    if workflow_name == "ADT_multi_agents":
        return _adt_specs(num_outputs)

    # Fallback for unknown/new workflows.
    return _generic_specs(num_outputs)

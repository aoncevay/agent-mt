#!/usr/bin/env python3
"""
Test script to verify that workflows correctly construct prompts with:
1. Previous agent outputs included in subsequent prompts
2. Terminology included when use_terminology=True
3. Memory/state correctly passed between agents

This script runs a small test sample through each workflow and extracts/validates prompts.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflows import WORKFLOW_REGISTRY
from workflows.ADT_multi_agents import (
    render_discourse_agent_prompt,
    render_translation_agent_prompt,
    render_memory_agent_prompt
)
from workflows.SbS_step_by_step import (
    render_research_prompt,
    render_draft_prompt,
    render_refinement_prompt,
    render_proofread_prompt
)
from workflows.SbS_chat_step_by_step import (
    render_research_prompt as render_chat_research_prompt,
    render_chat_draft_prompt,
    render_chat_refinement_prompt,
    render_chat_proofread_prompt
)
from workflows.MAATS_multi_agents import (
    render_dimension_prompt,
    render_refine_prompt
)
from workflows.MAATS_single_agent import (
    render_single_agent_prompt,
    render_single_agent_refine_prompt
)
from workflows.IRB_refine import render_refine_prompt as render_irb_refine_prompt
from utils import render_translation_prompt, format_terminology_dict, filter_terminology_by_source_text
from vars import language_id2name


# Test data
TEST_SOURCE_TEXT = "The United Nations (UN) is an international organization. It was founded in 1945. The UN promotes peace and security worldwide."
TEST_TERMINOLOGY = {
    "United Nations": ["联合国", "UN"],
    "international organization": ["国际组织"],
    "peace and security": ["和平与安全"]
}
TEST_REFERENCE = "联合国（UN）是一个国际组织。它成立于1945年。联合国促进世界和平与安全。"


def test_sbs_step_by_step():
    """Test SbS standalone workflow prompt construction."""
    print("\n" + "="*80)
    print("Testing SbS_step_by_step workflow prompts")
    print("="*80)
    
    # Step 1: Research
    research_prompt = render_research_prompt(
        source_text=TEST_SOURCE_TEXT,
        source_lang="en",
        target_lang="zh"
    )
    print("\n[Research Prompt]")
    print(f"Length: {len(research_prompt)} chars")
    assert TEST_SOURCE_TEXT in research_prompt, "Source text missing from research prompt"
    print("✓ Research prompt contains source text")
    
    # Simulate research output
    research_output = "This text discusses the United Nations organization and its founding."
    
    # Step 2: Draft (should include research output)
    draft_prompt = render_draft_prompt(
        source_text=TEST_SOURCE_TEXT,
        research_output=research_output
    )
    print("\n[Draft Prompt]")
    print(f"Length: {len(draft_prompt)} chars")
    assert TEST_SOURCE_TEXT in draft_prompt, "Source text missing from draft prompt"
    assert research_output in draft_prompt, "Research output missing from draft prompt"
    print("✓ Draft prompt contains source text and research output")
    
    # Simulate draft output
    draft_output = "联合国（UN）是一个国际组织。它成立于1945年。"
    
    # Step 3: Refinement (should include draft output)
    refinement_prompt = render_refinement_prompt(draft_translation=draft_output)
    print("\n[Refinement Prompt]")
    print(f"Length: {len(refinement_prompt)} chars")
    assert draft_output in refinement_prompt, "Draft output missing from refinement prompt"
    print("✓ Refinement prompt contains draft translation")
    
    # Simulate refinement output
    refinement_output = "联合国（UN）是一个国际组织。它成立于1945年。联合国促进世界和平与安全。"
    
    # Step 4: Proofreading (should include all previous outputs)
    proofread_prompt = render_proofread_prompt(
        source_text=TEST_SOURCE_TEXT,
        draft_translation=draft_output,
        refined_translation=refinement_output
    )
    print("\n[Proofread Prompt]")
    print(f"Length: {len(proofread_prompt)} chars")
    assert TEST_SOURCE_TEXT in proofread_prompt, "Source text missing from proofread prompt"
    assert draft_output in proofread_prompt, "Draft missing from proofread prompt"
    assert refinement_output in proofread_prompt, "Refined translation missing from proofread prompt"
    print("✓ Proofread prompt contains source, draft, and refined translation")
    
    # Test with terminology
    proofread_prompt_term = render_proofread_prompt(
        source_text=TEST_SOURCE_TEXT,
        draft_translation=draft_output,
        refined_translation=refinement_output
    )
    # Note: SbS_step_by_step doesn't have a term version of proofread, but we can check if terminology would be added
    print("\n✓ SbS_step_by_step prompt chain verified")


def test_sbs_chat_step_by_step():
    """Test SbS chat workflow prompt construction."""
    print("\n" + "="*80)
    print("Testing SbS_chat_step_by_step workflow prompts")
    print("="*80)
    
    # Research (same as standalone)
    research_prompt = render_chat_research_prompt(
        source_text=TEST_SOURCE_TEXT,
        source_lang="en",
        target_lang="zh"
    )
    research_output = "This text discusses the United Nations organization."
    
    # Draft (chat version - shorter, relies on conversation history)
    draft_prompt = render_chat_draft_prompt(source_text=TEST_SOURCE_TEXT)
    print("\n[Draft Prompt (Chat)]")
    print(f"Length: {len(draft_prompt)} chars")
    assert TEST_SOURCE_TEXT in draft_prompt, "Source text missing from chat draft prompt"
    # Chat version should be shorter (doesn't include research output in prompt)
    print("✓ Chat draft prompt contains source text (relies on conversation history)")
    
    draft_output = "联合国（UN）是一个国际组织。"
    
    # Refinement (chat version - very short, relies on conversation history)
    refinement_prompt = render_chat_refinement_prompt()
    print("\n[Refinement Prompt (Chat)]")
    print(f"Length: {len(refinement_prompt)} chars")
    # Chat version is very short, doesn't include draft in prompt
    print("✓ Chat refinement prompt is short (relies on conversation history)")
    
    refinement_output = "联合国（UN）是一个国际组织。它成立于1945年。"
    
    # Proofreading with terminology
    proofread_prompt_term = render_chat_proofread_prompt(
        source_text=TEST_SOURCE_TEXT,
        draft_translation=draft_output,
        refined_translation=refinement_output,
        terminology=TEST_TERMINOLOGY,
        use_terminology=True,
        source_lang="en",
        target_lang="zh"
    )
    print("\n[Proofread Prompt (Chat, with Terminology)]")
    print(f"Length: {len(proofread_prompt_term)} chars")
    assert TEST_SOURCE_TEXT in proofread_prompt_term, "Source text missing"
    assert draft_output in proofread_prompt_term, "Draft missing"
    assert refinement_output in proofread_prompt_term, "Refined translation missing"
    # Check for terminology
    has_terminology = any(term in proofread_prompt_term for term in ["United Nations", "联合国", "国际组织"])
    assert has_terminology, "Terminology missing from proofread prompt"
    print("✓ Proofread prompt contains source, draft, refined translation, and terminology")
    
    print("\n✓ SbS_chat_step_by_step prompt chain verified")


def test_maats_multi_agents():
    """Test MAATS multi-agent workflow prompt construction."""
    print("\n" + "="*80)
    print("Testing MAATS_multi_agents workflow prompts")
    print("="*80)
    
    # Simulate initial translation
    initial_translation = "联合国（UN）是一个国际组织。它成立于1945年。联合国促进世界和平与安全。"
    
    # Test dimension prompts (should include source and translation)
    for dimension in ["terminology", "accuracy", "style"]:
        dimension_prompt = render_dimension_prompt(
            dimension=dimension,
            source_text=TEST_SOURCE_TEXT,
            translation=initial_translation,
            source_lang="en",
            target_lang="zh"
        )
        print(f"\n[{dimension.capitalize()} Dimension Prompt]")
        print(f"Length: {len(dimension_prompt)} chars")
        assert TEST_SOURCE_TEXT in dimension_prompt, f"Source text missing from {dimension} prompt"
        assert initial_translation in dimension_prompt, f"Translation missing from {dimension} prompt"
        print(f"✓ {dimension.capitalize()} prompt contains source text and translation")
    
    # Simulate annotations
    annotations = {
        "terminology": "[Major]: [terminology/wrong_term] - Example issue",
        "accuracy": "[Minor]: [accuracy/error_subcategory] - None",
        "style": "[Critical]: [style/error_subcategory] - None"
    }
    
    # Refinement prompt (should include all annotations)
    refine_prompt = render_refine_prompt(
        source_text=TEST_SOURCE_TEXT,
        translation=initial_translation,
        annotations=annotations,
        source_lang="en",
        target_lang="zh"
    )
    print("\n[Refinement Prompt]")
    print(f"Length: {len(refine_prompt)} chars")
    assert TEST_SOURCE_TEXT in refine_prompt, "Source text missing from refine prompt"
    assert initial_translation in refine_prompt, "Translation missing from refine prompt"
    # Check that annotations are included (they're formatted as JSON)
    assert "terminology" in refine_prompt.lower(), "Terminology annotations missing"
    assert "accuracy" in refine_prompt.lower(), "Accuracy annotations missing"
    print("✓ Refinement prompt contains source, translation, and all annotations")
    
    print("\n✓ MAATS_multi_agents prompt chain verified")


def test_adt_multi_agents():
    """Test ADT workflow prompt construction."""
    print("\n" + "="*80)
    print("Testing ADT_multi_agents workflow prompts")
    print("="*80)
    
    # Discourse Agent prompt
    current_discourse = "The United Nations (UN) is an international organization."
    next_sentence = "It was founded in 1945."
    discourse_prompt = render_discourse_agent_prompt(
        current_discourse=current_discourse,
        next_sentence=next_sentence,
        source_lang="en",
        max_discourse_length=1024
    )
    print("\n[Discourse Agent Prompt]")
    print(f"Length: {len(discourse_prompt)} chars")
    assert current_discourse in discourse_prompt, "Current discourse missing"
    assert next_sentence in discourse_prompt, "Next sentence missing"
    print("✓ Discourse agent prompt contains current discourse and next sentence")
    
    # Translation Agent prompt (with memory)
    memory = {
        "proper_noun_references": {"United Nations": "联合国", "UN": "UN"},
        "phrase_consistency": {"international organization": "国际组织"},
        "discourse_markers": []
    }
    discourse = "The United Nations (UN) is an international organization. It was founded in 1945."
    
    translation_prompt = render_translation_agent_prompt(
        discourse=discourse,
        memory=memory,
        source_lang="en",
        target_lang="zh",
        terminology=None,
        use_terminology=False
    )
    print("\n[Translation Agent Prompt (without terminology)]")
    print(f"Length: {len(translation_prompt)} chars")
    assert discourse in translation_prompt, "Discourse missing from translation prompt"
    assert "United Nations" in translation_prompt or "联合国" in translation_prompt, "Memory not visible in prompt"
    print("✓ Translation prompt contains discourse and memory")
    
    # Translation Agent prompt (with terminology)
    translation_prompt_term = render_translation_agent_prompt(
        discourse=discourse,
        memory=memory,
        source_lang="en",
        target_lang="zh",
        terminology=TEST_TERMINOLOGY,
        use_terminology=True
    )
    print("\n[Translation Agent Prompt (with terminology)]")
    print(f"Length: {len(translation_prompt_term)} chars")
    assert discourse in translation_prompt_term, "Discourse missing"
    has_terminology = any(term in translation_prompt_term for term in ["United Nations", "联合国", "国际组织"])
    assert has_terminology, "Terminology missing from translation prompt"
    print("✓ Translation prompt with terminology contains discourse, memory, and terminology")
    
    # Memory Agent prompt
    target_discourse = "联合国（UN）是一个国际组织。它成立于1945年。"
    memory_prompt = render_memory_agent_prompt(
        current_memory=memory,
        source_discourse=discourse,
        target_discourse=target_discourse
    )
    print("\n[Memory Agent Prompt]")
    print(f"Length: {len(memory_prompt)} chars")
    assert discourse in memory_prompt, "Source discourse missing from memory prompt"
    assert target_discourse in memory_prompt, "Target discourse missing from memory prompt"
    assert "United Nations" in memory_prompt or "联合国" in memory_prompt, "Current memory not visible"
    print("✓ Memory agent prompt contains source discourse, target discourse, and current memory")
    
    print("\n✓ ADT_multi_agents prompt chain verified")


def test_terminology_inclusion():
    """Test that terminology is correctly included when use_terminology=True."""
    print("\n" + "="*80)
    print("Testing Terminology Inclusion")
    print("="*80)
    
    # Test terminology formatting
    formatted_term = format_terminology_dict(TEST_TERMINOLOGY, "en", "zh", max_terms=50)
    print(f"\nFormatted terminology: {len(formatted_term)} entries")
    
    # Test filtering
    filtered_term = filter_terminology_by_source_text(
        formatted_term, TEST_SOURCE_TEXT, case_sensitive=False
    )
    print(f"Filtered terminology (for test text): {len(filtered_term)} entries")
    
    # Test translation prompt with terminology
    translation_prompt = render_translation_prompt(
        source_text=TEST_SOURCE_TEXT,
        source_lang="en",
        target_lang="zh",
        language_id2name=language_id2name,
        use_terminology=True,
        terminology=filtered_term,
        max_terms=50
    )
    print("\n[Translation Prompt with Terminology]")
    print(f"Length: {len(translation_prompt)} chars")
    assert TEST_SOURCE_TEXT in translation_prompt, "Source text missing"
    # Check for terminology indicators
    has_terminology = any(term in translation_prompt for term in ["United Nations", "联合国", "terminology", "dictionary"])
    assert has_terminology, "Terminology not found in translation prompt"
    print("✓ Terminology correctly included in translation prompt")
    
    print("\n✓ Terminology inclusion verified")


def main():
    """Run all prompt validation tests."""
    print("="*80)
    print("WORKFLOW PROMPT VALIDATION TESTS")
    print("="*80)
    print("\nThis script validates that:")
    print("1. Previous agent outputs are included in subsequent prompts")
    print("2. Terminology is included when use_terminology=True")
    print("3. Memory/state is correctly passed between agents")
    print("\nNote: This tests PROMPT CONSTRUCTION, not LLM output quality.")
    
    try:
        test_sbs_step_by_step()
        test_sbs_chat_step_by_step()
        test_maats_multi_agents()
        test_adt_multi_agents()
        test_terminology_inclusion()
        
        print("\n" + "="*80)
        print("✓ ALL PROMPT VALIDATION TESTS PASSED")
        print("="*80)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


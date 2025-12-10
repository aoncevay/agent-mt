"""
SbS Step-by-Step translation workflow: Research -> Draft -> Refinement -> Proofreading.
Based on "Translating Step-by-Step: Decomposing the Translation Process for Improved Translation Quality of Long-Form Texts"
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage

try:
    from ..translation import create_bedrock_llm
    from ..utils import load_template, get_language_name
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm
    from utils import load_template, get_language_name
    from vars import language_id2name


def render_research_prompt(source_text: str, source_lang: str, target_lang: str) -> str:
    """Render the pre-translation research prompt."""
    template = load_template("SbS_research.jinja")
    return template.render(
        source_text=source_text,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name)
    )


def render_draft_prompt(source_text: str, research_output: str) -> str:
    """Render the drafting prompt."""
    template = load_template("SbS_draft.jinja")
    return template.render(
        source_text=source_text,
        research_output=research_output
    )


def render_refinement_prompt(draft_translation: str) -> str:
    """Render the refinement prompt."""
    template = load_template("SbS_refinement.jinja")
    return template.render(draft_translation=draft_translation)


def render_proofread_prompt(source_text: str, draft_translation: str, refined_translation: str) -> str:
    """Render the proofreading prompt."""
    template = load_template("SbS_proofread.jinja")
    return template.render(
        source_text=source_text,
        draft_translation=draft_translation,
        refined_translation=refined_translation
    )


def run_workflow(
    source_text: str,
    source_lang: str,
    target_lang: str,
    model_id: str,
    terminology: Optional[Dict[str, list]] = None,
    region: Optional[str] = None,
    max_retries: int = 3,
    initial_backoff: float = 2.0,
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run four-agent step-by-step translation workflow: Research -> Draft -> Refinement -> Proofreading.
    
    Args:
        source_text: Source text to translate
        source_lang: Source language code
        target_lang: Target language code
        model_id: Bedrock model ID
        terminology: Optional terminology dictionary (not used in this workflow)
        region: AWS region
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff delay
        reference: Optional reference translation (not used in this workflow)
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs [research_output, draft_output, refinement_output, proofread_output]
        - 'tokens_input': Total input tokens
        - 'tokens_output': Total output tokens
        - 'latency': Total workflow time in seconds
    """
    import time
    
    # Create LLM
    llm = create_bedrock_llm(model_id, region)
    
    total_tokens_input = 0
    total_tokens_output = 0
    start_time = time.time()
    
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    # Step 1: Research Agent
    print("    [Agent 1/4] Research...")
    research_prompt = render_research_prompt(source_text, source_lang, target_lang)
    
    research_output = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=research_prompt)
            response = llm.invoke([message])
            
            research_output = response.content.strip()
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(research_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(research_output) // 4
            
            total_tokens_input += tokens_input
            total_tokens_output += tokens_output
            
            break
        
        except (ReadTimeoutError, ClientError) as e:
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str or 
                "read timeout" in error_str or
                isinstance(e, ReadTimeoutError)
            )
            
            if not is_timeout:
                raise
            
            if attempt < max_retries:
                backoff_time = (2 ** attempt) * initial_backoff
                print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
            else:
                raise RuntimeError(f"Research failed after {max_retries + 1} attempts due to timeout") from e
    
    if research_output is None:
        raise RuntimeError("Research step failed")
    
    # Step 2: Draft Agent (standalone - includes research output in prompt)
    # This version uses standalone prompts (no conversation history) to minimize token usage
    print("    [Agent 2/4] Draft...")
    draft_prompt = render_draft_prompt(source_text, research_output)
    
    draft_output = None
    for attempt in range(max_retries + 1):
        try:
            # Standalone call: draft_prompt already includes research_output
            message = HumanMessage(content=draft_prompt)
            response = llm.invoke([message])
            
            draft_output = response.content.strip()
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                # Estimate: draft_prompt already includes research_output, so just use draft_prompt length
                tokens_input = len(draft_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(draft_output) // 4
            
            total_tokens_input += tokens_input
            total_tokens_output += tokens_output
            
            break
        
        except (ReadTimeoutError, ClientError) as e:
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str or 
                "read timeout" in error_str or
                isinstance(e, ReadTimeoutError)
            )
            
            if not is_timeout:
                raise
            
            if attempt < max_retries:
                backoff_time = (2 ** attempt) * initial_backoff
                print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
            else:
                raise RuntimeError(f"Draft failed after {max_retries + 1} attempts due to timeout") from e
    
    if draft_output is None:
        raise RuntimeError("Draft step failed")
    
    # Step 3: Refinement Agent (standalone - includes draft output in prompt)
    # This version uses standalone prompts (no conversation history) to minimize token usage
    print("    [Agent 3/4] Refinement...")
    refinement_prompt = render_refinement_prompt(draft_output)
    
    refinement_output = None
    for attempt in range(max_retries + 1):
        try:
            # Standalone call: refinement_prompt already includes draft_output
            message = HumanMessage(content=refinement_prompt)
            response = llm.invoke([message])
            
            refinement_output = response.content.strip()
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                # Estimate: refinement_prompt already includes draft_output, so just use refinement_prompt length
                tokens_input = len(refinement_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(refinement_output) // 4
            
            total_tokens_input += tokens_input
            total_tokens_output += tokens_output
            
            break
        
        except (ReadTimeoutError, ClientError) as e:
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str or 
                "read timeout" in error_str or
                isinstance(e, ReadTimeoutError)
            )
            
            if not is_timeout:
                raise
            
            if attempt < max_retries:
                backoff_time = (2 ** attempt) * initial_backoff
                print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
            else:
                raise RuntimeError(f"Refinement failed after {max_retries + 1} attempts due to timeout") from e
    
    if refinement_output is None:
        raise RuntimeError("Refinement step failed")
    
    # Step 4: Proofreading Agent (standalone - includes all previous outputs)
    # Paper says: "proofreading requires a new perspective after a break from revising"
    print("    [Agent 4/4] Proofreading...")
    proofread_prompt = render_proofread_prompt(source_text, draft_output, refinement_output)
    
    proofread_output = None
    for attempt in range(max_retries + 1):
        try:
            # Standalone call: proofread_prompt is self-contained with all necessary context
            message = HumanMessage(content=proofread_prompt)
            response = llm.invoke([message])
            
            proofread_output = response.content.strip()
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(proofread_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(proofread_output) // 4
            
            total_tokens_input += tokens_input
            total_tokens_output += tokens_output
            
            break
        
        except (ReadTimeoutError, ClientError) as e:
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str or 
                "read timeout" in error_str or
                isinstance(e, ReadTimeoutError)
            )
            
            if not is_timeout:
                raise
            
            if attempt < max_retries:
                backoff_time = (2 ** attempt) * initial_backoff
                print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
            else:
                raise RuntimeError(f"Proofreading failed after {max_retries + 1} attempts due to timeout") from e
    
    if proofread_output is None:
        raise RuntimeError("Proofreading step failed")
    
    latency = time.time() - start_time
    
    return {
        "outputs": [research_output, draft_output, refinement_output, proofread_output],
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


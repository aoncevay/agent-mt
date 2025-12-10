"""
SbS Step-by-Step translation workflow (CHAT VERSION): Research -> Draft -> Refinement -> Proofreading.
Uses conversation history for Research->Draft->Refinement (as paper suggests improves performance).
Based on "Translating Step-by-Step: Decomposing the Translation Process for Improved Translation Quality of Long-Form Texts"

This version uses conversation history, which may increase token usage but potentially improves quality.
Compare with SbS_step_by_step (standalone version) to measure performance and cost differences.
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


def render_chat_draft_prompt(source_text: str) -> str:
    """Render the drafting prompt (chat version - doesn't include research output, uses conversation history)."""
    template = load_template("SbS_chat_draft.jinja")
    return template.render(source_text=source_text)


def render_chat_refinement_prompt() -> str:
    """Render the refinement prompt (chat version - doesn't include draft output, uses conversation history)."""
    template = load_template("SbS_chat_refinement.jinja")
    return template.render()


def render_chat_proofread_prompt(source_text: str, draft_translation: str, refined_translation: str) -> str:
    """Render the proofreading prompt (same as standalone - proofreading is always standalone per paper)."""
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
    Run four-agent step-by-step translation workflow with conversation history: Research -> Draft -> Refinement -> Proofreading.
    
    This version uses conversation history for Research->Draft->Refinement (as the paper suggests improves performance).
    Proofreading uses a standalone call (as per paper: "requires a new perspective after a break from revising").
    
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
        - 'tokens_input': Total input tokens (may be higher due to conversation history)
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
    
    # Step 1: Research Agent (standalone)
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
    
    # Step 2: Draft Agent (continues conversation with research)
    # Paper says: "With the exception of the proofreading step, we found that continuing 
    # the conversation improved performance."
    print("    [Agent 2/4] Draft...")
    draft_prompt = render_chat_draft_prompt(source_text)
    
    draft_output = None
    for attempt in range(max_retries + 1):
        try:
            # Continue conversation: include research response for context
            # The draft_prompt does NOT include research_output (relies on conversation history)
            messages = [
                HumanMessage(content=research_prompt),
                AIMessage(content=research_output),
                HumanMessage(content=draft_prompt)
            ]
            response = llm.invoke(messages)
            
            draft_output = response.content.strip()
            
            # Get token counts
            # Note: tokens_input includes all previous messages in conversation history
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                # Estimate: research prompt + research output + draft prompt
                tokens_input = (len(research_prompt) + len(research_output) + len(draft_prompt)) // 4
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
    
    # Step 3: Refinement Agent (continues conversation)
    # Paper says continuing conversation improves performance
    print("    [Agent 3/4] Refinement...")
    refinement_prompt = render_chat_refinement_prompt()
    
    refinement_output = None
    for attempt in range(max_retries + 1):
        try:
            # Continue conversation: include all previous steps for context
            # The refinement_prompt does NOT include draft_output (relies on conversation history)
            messages = [
                HumanMessage(content=research_prompt),
                AIMessage(content=research_output),
                HumanMessage(content=draft_prompt),
                AIMessage(content=draft_output),
                HumanMessage(content=refinement_prompt)
            ]
            response = llm.invoke(messages)
            
            refinement_output = response.content.strip()
            
            # Get token counts
            # Note: tokens_input includes ALL previous messages in conversation history
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                # Estimate: all previous content + refinement prompt
                total_prev = len(research_prompt) + len(research_output) + len(draft_prompt) + len(draft_output) + len(refinement_prompt)
                tokens_input = total_prev // 4
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
    
    # Step 4: Proofreading Agent (NEW conversation, not continuing)
    # Paper says: "proofreading requires a new perspective after a break from revising"
    # and "we found that continuing the conversation improved performance" EXCEPT for proofreading
    print("    [Agent 4/4] Proofreading...")
    proofread_prompt = render_chat_proofread_prompt(source_text, draft_output, refinement_output)
    
    proofread_output = None
    for attempt in range(max_retries + 1):
        try:
            # Start NEW conversation for proofreading (standalone, as per paper)
            # The proofread_prompt is self-contained with all necessary context
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


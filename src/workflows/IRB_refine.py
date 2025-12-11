"""
IRB-MT workflow: Two-stage self-refine translation.
Based on "IRB-MT at WMT25 Translation Task: A Simple Agentic System Using an Off-the-Shelf LLM"

Workflow:
1. Initial translation using base translation prompt (with or without terminology)
2. Refinement agent that reviews and improves the translation

The refinement prompt includes the original translation prompt, source text, and initial translation.
The model is instructed to reason about improvements and output the solution in <solution></solution> tags.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
import re

try:
    from ..translation import create_bedrock_llm
    from ..utils import load_template, render_translation_prompt, format_terminology_dict, filter_terminology_by_source_text
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm
    from utils import load_template, render_translation_prompt, format_terminology_dict, filter_terminology_by_source_text
    from vars import language_id2name


def render_refine_prompt(
    translation_prompt: str,
    original_text: str,
    translation: str,
    reasoning_words: int = 500
) -> str:
    """Render the IRB refinement prompt."""
    template = load_template("IRB_refine.jinja")
    return template.render(
        translation_prompt=translation_prompt,
        original_text=original_text,
        translation=translation,
        reasoning_words=reasoning_words
    )


def extract_solution(text: str) -> str:
    """
    Extract the solution from <solution></solution> tags.
    If tags are not found, return the entire text (fallback).
    """
    # Try to find content between <solution> and </solution> tags
    pattern = r'<solution>(.*?)</solution>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no tags found, return the entire text (fallback)
    return text.strip()


def run_workflow(
    source_text: str,
    source_lang: str,
    target_lang: str,
    model_id: str,
    terminology: Optional[Dict[str, list]] = None,
    use_terminology: bool = False,
    region: Optional[str] = None,
    max_retries: int = 3,
    initial_backoff: float = 2.0,
    reference: Optional[str] = None,
    reasoning_words: int = 500
) -> Dict[str, Any]:
    """
    Run IRB two-stage self-refine translation workflow.
    
    Args:
        source_text: Source text to translate
        source_lang: Source language code
        target_lang: Target language code
        model_id: Bedrock model ID
        terminology: Optional terminology dictionary (used in initial translation if use_terminology=True)
        use_terminology: If True, use terminology dictionary in initial translation step
        region: AWS region
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff delay
        reference: Optional reference translation (not used in this workflow)
        reasoning_words: Target number of words for reasoning (default: 300, as in paper)
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs [initial_translation, refined_translation]
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
    outputs = []
    
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    # Step 1: Initial translation (with terminology if available)
    print("    [Agent 1/2] Initial translation...")
    
    # Filter terminology to only include terms that appear in source text
    filtered_terminology = None
    if use_terminology and terminology:
        filtered_terminology = format_terminology_dict(terminology, source_lang, target_lang, max_terms=50)
        if filtered_terminology:
            filtered_terminology = filter_terminology_by_source_text(
                filtered_terminology, source_text, case_sensitive=False
            )
            if filtered_terminology:
                print(f"    Using {len(filtered_terminology)} relevant terminology entries "
                      f"(out of {len(terminology)} total)")
            else:
                print(f"    No terminology entries found in source text (out of {len(terminology)} total)")
    
    translation_prompt = render_translation_prompt(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        language_id2name=language_id2name,
        use_terminology=filtered_terminology is not None,
        terminology=filtered_terminology,
        max_terms=None if filtered_terminology else 50
    )
    
    initial_translation = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=translation_prompt)
            response = llm.invoke([message])
            
            initial_translation = response.content.strip()
            
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(translation_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(initial_translation) // 4
            
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
                raise RuntimeError(f"Initial translation failed after {max_retries + 1} attempts due to timeout") from e
    
    if initial_translation is None:
        raise RuntimeError("Initial translation step failed")
    
    outputs.append(initial_translation)
    
    # Step 2: Refinement agent
    print("    [Agent 2/2] Refinement...")
    refine_prompt = render_refine_prompt(
        translation_prompt=translation_prompt,
        original_text=source_text,
        translation=initial_translation,
        reasoning_words=reasoning_words
    )
    
    refined_translation = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=refine_prompt)
            response = llm.invoke([message])
            
            # Extract solution from <solution></solution> tags
            full_response = response.content.strip()
            refined_translation = extract_solution(full_response)
            
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(refine_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(full_response) // 4  # Count all tokens, including reasoning
            
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
    
    if refined_translation is None:
        raise RuntimeError("Refinement step failed")
    
    outputs.append(refined_translation)
    
    latency = time.time() - start_time
    
    return {
        "outputs": outputs,  # [initial_translation, refined_translation]
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


"""
MaMT workflow: Translate + Postedit.
Based on the Multi-agent MT paper (WMT 2025).
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage

try:
    from ..translation import create_bedrock_llm
    from ..utils import (
        render_translation_prompt,
        render_postedit_prompt,
        parse_postedit_response,
        apply_postedit_corrections
    )
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm
    from utils import (
        render_translation_prompt,
        render_postedit_prompt,
        parse_postedit_response,
        apply_postedit_corrections
    )
    from vars import language_id2name


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
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run two-agent translation workflow: Translate -> Postedit.
    
    Args:
        source_text: Source text to translate
        source_lang: Source language code
        target_lang: Target language code
        model_id: Bedrock model ID
        terminology: Optional terminology dictionary (not used in this workflow)
        region: AWS region
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff delay
        reference: Optional reference translation for postedit agent
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs [translate_output, postedit_output]
        - 'tokens_input': Total input tokens
        - 'tokens_output': Total output tokens
        - 'latency': Total workflow time in seconds
    """
    import time
    
    # Create LLMs with different temperatures per paper:
    # - Translation: temperature=0 (reproducibility)
    # - Postedit: temperature=1 (exploration, encourages broader error detection)
    llm_translate = create_bedrock_llm(model_id, region, temperature=0.0)
    llm_postedit = create_bedrock_llm(model_id, region, temperature=1.0)
    
    total_tokens_input = 0
    total_tokens_output = 0
    start_time = time.time()
    
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    # Step 1: Translate Agent (temperature=0 for reproducibility)
    print("    [Agent 1/2] Translate...")
    translate_prompt = render_translation_prompt(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        language_id2name=language_id2name,
        use_terminology=False,
        terminology=None,
        max_terms=50
    )
    
    translation = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=translate_prompt)
            response = llm_translate.invoke([message])
            
            translation = response.content.strip()
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(translate_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(translation) // 4
            
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
                raise RuntimeError(f"Translation failed after {max_retries + 1} attempts due to timeout") from e
    
    if translation is None:
        raise RuntimeError("Translation step failed")
    
    # Step 2: Postedit Agent (temperature=1 for exploration, as per paper)
    print("    [Agent 2/2] Postedit...")
    postedit_prompt = render_postedit_prompt(
        source_text=source_text,
        translation=translation,
        source_lang=source_lang,
        target_lang=target_lang,
        language_id2name=language_id2name,
        reference=reference
    )
    
    postedit_output = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=postedit_prompt)
            response = llm_postedit.invoke([message])
            
            postedit_response_text = response.content.strip()
            
            # Parse corrections and apply them
            corrections = parse_postedit_response(postedit_response_text)
            postedit_output = apply_postedit_corrections(translation, corrections)
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(postedit_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(postedit_response_text) // 4
            
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
                raise RuntimeError(f"Postedit failed after {max_retries + 1} attempts due to timeout") from e
    
    if postedit_output is None:
        raise RuntimeError("Postedit step failed")
    
    latency = time.time() - start_time
    
    return {
        "outputs": [translation, postedit_output],
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


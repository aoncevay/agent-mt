"""
MaMT workflow WITH TERMINOLOGY: Translate + Postedit + Proofread.
Includes terminology dictionary in the proofreading step.
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
        apply_postedit_corrections,
        load_template,
        get_language_name,
        format_terminology_dict,
        filter_terminology_by_source_text
    )
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm
    from utils import (
        render_translation_prompt,
        render_postedit_prompt,
        parse_postedit_response,
        apply_postedit_corrections,
        load_template,
        get_language_name,
        format_terminology_dict,
        filter_terminology_by_source_text
    )
    from vars import language_id2name


def render_proofread_prompt_term(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    language_id2name: Dict[str, str],
    domain: str = "general",
    terminology: Optional[Dict[str, list]] = None
) -> str:
    """Render a proofread prompt with terminology (if available) using Jinja template."""
    template = load_template("MaMT_proofread_term.jinja")
    
    # Format and filter terminology if available
    formatted_terminology = None
    if terminology:
        formatted_terminology = format_terminology_dict(terminology, source_lang, target_lang, max_terms=50)
        if formatted_terminology:
            formatted_terminology = filter_terminology_by_source_text(
                formatted_terminology, source_text, case_sensitive=False
            )
            if formatted_terminology:
                print(f"    Using {len(formatted_terminology)} relevant terminology entries in proofreading")
    
    return template.render(
        source_text=source_text,
        translation=translation,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name),
        domain=domain,
        terminology=formatted_terminology
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
    reference: Optional[str] = None,
    domain: str = "general"
) -> Dict[str, Any]:
    """
    Run three-agent translation workflow with terminology support: Translate -> Postedit -> Proofread.
    
    Terminology is included in the proofreading step (last agent).
    
    Args:
        source_text: Source text to translate
        source_lang: Source language code
        target_lang: Target language code
        model_id: Bedrock model ID
        terminology: Optional terminology dictionary (used in proofreading step)
        region: AWS region
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff delay
        reference: Optional reference translation for postedit agent
        domain: Domain for proofread agent (default: "general")
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs [translate_output, postedit_output, proofread_output]
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
    
    # Step 1: Translate Agent
    print("    [Agent 1/3] Translate...")
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
            response = llm.invoke([message])
            
            translation = response.content.strip()
            
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
    
    # Step 2: Postedit Agent
    print("    [Agent 2/3] Postedit...")
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
            response = llm.invoke([message])
            
            postedit_response_text = response.content.strip()
            corrections = parse_postedit_response(postedit_response_text)
            postedit_output = apply_postedit_corrections(translation, corrections)
            
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
    
    # Step 3: Proofread Agent (with terminology if available)
    print("    [Agent 3/3] Proofread...")
    proofread_prompt = render_proofread_prompt_term(
        source_text=source_text,
        translation=postedit_output,
        source_lang=source_lang,
        target_lang=target_lang,
        language_id2name=language_id2name,
        domain=domain,
        terminology=terminology
    )
    
    proofread_output = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=proofread_prompt)
            response = llm.invoke([message])
            
            proofread_output = response.content.strip()
            
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
                raise RuntimeError(f"Proofread failed after {max_retries + 1} attempts due to timeout") from e
    
    if proofread_output is None:
        raise RuntimeError("Proofread step failed")
    
    latency = time.time() - start_time
    
    return {
        "outputs": [translation, postedit_output, proofread_output],
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


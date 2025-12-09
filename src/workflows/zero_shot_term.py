"""
Zero-shot workflow with terminology - direct translation with terminology.
"""

from typing import Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

try:
    from ..translation import create_bedrock_llm
    from ..utils import render_translation_prompt, filter_terminology_by_source_text
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm
    from utils import render_translation_prompt, filter_terminology_by_source_text
    from vars import language_id2name


def run_workflow(
    source_text: str,
    source_lang: str,
    target_lang: str,
    model_id: str,
    terminology: Optional[Dict[str, list]] = None,
    region: Optional[str] = None,
    max_retries: int = 3,
    initial_backoff: float = 2.0
) -> Dict[str, Any]:
    """
    Run single agent translation workflow with terminology.
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs (single item for this workflow)
        - 'tokens_input': Total input tokens
        - 'tokens_output': Total output tokens
        - 'latency': Translation time in seconds
    """
    import time
    
    # Create LLM
    llm = create_bedrock_llm(model_id, region)
    
    # Filter terminology to only include terms that appear in source text
    filtered_terminology = None
    if terminology:
        filtered_terminology = filter_terminology_by_source_text(
            terminology, source_text, case_sensitive=False
        )
        if filtered_terminology:
            print(f"    Using {len(filtered_terminology)} relevant terminology entries "
                  f"(out of {len(terminology)} total)")
        else:
            print(f"    No terminology entries found in source text (out of {len(terminology)} total)")
    
    # Create prompt (with filtered terminology)
    prompt = render_translation_prompt(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        language_id2name=language_id2name,
        use_terminology=True,
        terminology=filtered_terminology,
        max_terms=None  # No need to limit since we've already filtered
    )
    
    # Translate with retry logic
    start_time = time.time()
    last_exception = None
    
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=prompt)
            response = llm.invoke([message])
            
            translation = response.content.strip()
            latency = time.time() - start_time
            
            # Get token counts from response if available
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            # If not available, estimate (rough approximation: ~4 chars per token)
            if tokens_input == 0:
                tokens_input = len(prompt) // 4
            if tokens_output == 0:
                tokens_output = len(translation) // 4
            
            return {
                "outputs": [translation],
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "latency": latency
            }
        
        except (ReadTimeoutError, ClientError) as e:
            last_exception = e
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
    
    if last_exception:
        raise RuntimeError("Translation failed after all retries") from last_exception
    raise RuntimeError("Translation failed for unknown reason")


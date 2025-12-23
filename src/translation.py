"""
Translation functions using LangChain/LangGraph with Amazon Bedrock.
"""

import os
import time
from typing import Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Import for timeout error handling
try:
    from botocore.exceptions import ReadTimeoutError, ClientError
except ImportError:
    # Fallback if botocore is not available
    ReadTimeoutError = Exception
    ClientError = Exception

try:
    from .utils import render_translation_prompt
    from .vars import language_id2name
except ImportError:
    # For direct script execution
    from utils import render_translation_prompt
    from vars import language_id2name


class TranslationState(TypedDict):
    """State for the translation workflow."""
    source_text: str
    source_lang: str
    target_lang: str
    terminology: Optional[Dict[str, list]]
    translation: str
    use_terminology: bool


def create_bedrock_llm(
    model_id: str, 
    region: Optional[str] = None, 
    temperature: float = 0.0,
    model_provider: Optional[str] = None
) -> ChatBedrock:
    """
    Create a Bedrock LLM instance with configurable temperature.
    
    Args:
        model_id: Bedrock model ID (e.g., "qwen.qwen3-32b-v1:0") or ARN
        region: AWS region (defaults to AWS_REGION env var or us-east-2 for model IDs, us-east-1 for ARNs)
        temperature: Sampling temperature (default: 0.0 for reproducibility)
                    - 0.0: Deterministic, reproducible (recommended for experiments)
                    - 0.1: Near-deterministic fallback if 0.0 causes issues
                    - Higher values: More creative but less reproducible
        model_provider: Provider name (e.g., "anthropic") - required when using ARNs
    
    Note:
        Papers use temperature=0 for reproducibility (IRB-WMT25, MaMT final translation).
        Some papers use temperature=1 for exploration (MaMT postedit), but we default to 0.0
        for consistency and reproducibility across all workflows.
        
        When using ARNs (application-inference-profile), model_provider must be specified.
        Default region for ARNs is us-east-1, for model IDs it's us-east-2.
    """
    # Determine if model_id is an ARN
    is_arn = model_id.startswith("arn:aws:bedrock:")
    
    # Set default region: us-east-1 for ARNs, us-east-2 for model IDs
    if region is None:
        if is_arn:
            aws_region = os.getenv("AWS_REGION", "us-east-1")
        else:
            aws_region = os.getenv("AWS_REGION", "us-east-2")
    else:
        aws_region = region
    
    # Build ChatBedrock kwargs
    bedrock_kwargs = {
        "model_id": model_id,
        "region_name": aws_region,
        "credentials_profile_name": None,
        "model_kwargs": {
            "temperature": temperature,
            "max_tokens": 4096,
        },
    }
    
    # Add model_provider if provided (required for ARNs)
    if model_provider:
        bedrock_kwargs["model_provider"] = model_provider
    
    bedrock_llm = ChatBedrock(**bedrock_kwargs)
    
    return bedrock_llm


def create_translate_node(llm: ChatBedrock, use_terminology: bool):
    """Create a translation node with the LLM bound."""
    def translate_node(state: TranslationState) -> TranslationState:
        """Node that performs the translation."""
        source_text = state["source_text"]
        source_lang = state["source_lang"]
        target_lang = state["target_lang"]
        terminology = state.get("terminology", {}) if use_terminology else None
        
        # Create prompt
        prompt = render_translation_prompt(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            language_id2name=language_id2name,
            use_terminology=use_terminology,
            terminology=terminology,
            max_terms=50
        )
        
        # Call the LLM
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        # Extract translation
        translation = response.content.strip()
        
        return {"translation": translation}
    
    return translate_node


def create_translation_graph(llm: ChatBedrock, use_terminology: bool) -> StateGraph:
    """Create a LangGraph workflow for translation."""
    workflow = StateGraph(TranslationState)
    
    # Add the translation node (with LLM bound)
    translate_node_func = create_translate_node(llm, use_terminology)
    workflow.add_node("translate", translate_node_func)
    
    # Set entry point
    workflow.set_entry_point("translate")
    
    # Always end after translation
    workflow.add_edge("translate", END)
    
    return workflow.compile()


def translate_text(
    source_text: str,
    source_lang: str,
    target_lang: str,
    model_id: str,
    use_terminology: bool = False,
    terminology: Optional[Dict[str, list]] = None,
    region: Optional[str] = None,
    max_retries: int = 3,
    initial_backoff: float = 2.0
) -> str:
    """
    Translate text using Bedrock model with retry logic for timeout errors.
    
    Args:
        source_text: Text to translate
        source_lang: Source language code (e.g., "en", "zht")
        target_lang: Target language code (e.g., "zht", "en")
        model_id: Bedrock model ID
        use_terminology: Whether to use terminology in the prompt
        terminology: Terminology dictionary (source_term -> [target_terms])
        region: AWS region (defaults to AWS_REGION env var or us-east-2)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff delay in seconds (default: 2.0)
    
    Returns:
        Translated text
    
    Raises:
        Exception: If all retry attempts fail
    """
    # Create LLM
    llm = create_bedrock_llm(model_id, region)
    
    # Create workflow
    workflow = create_translation_graph(llm, use_terminology)
    
    # Prepare initial state
    initial_state = {
        "source_text": source_text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "terminology": terminology or {},
        "translation": "",
        "use_terminology": use_terminology,
    }
    
    # Retry logic with exponential backoff
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            # Run the workflow
            result = workflow.invoke(initial_state)
            return result["translation"]
        
        except (ReadTimeoutError, ClientError) as e:
            last_exception = e
            # Check if it's a timeout error
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str or 
                "read timeout" in error_str or
                isinstance(e, ReadTimeoutError)
            )
            
            if not is_timeout:
                # Not a timeout error, don't retry
                raise
            
            if attempt < max_retries:
                # Calculate exponential backoff: 2^attempt * initial_backoff
                backoff_time = (2 ** attempt) * initial_backoff
                print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), "
                      f"retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
            else:
                # All retries exhausted
                print(f"    ✗ All {max_retries + 1} attempts failed")
                raise RuntimeError(f"Translation failed after {max_retries + 1} attempts due to timeout") from e
        
        except Exception as e:
            # For other exceptions, don't retry
            raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise RuntimeError("Translation failed after all retries") from last_exception
    raise RuntimeError("Translation failed for unknown reason")


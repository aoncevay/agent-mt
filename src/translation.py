"""
Translation functions using LangChain/LangGraph with Amazon Bedrock and OpenAI (via cdao).
"""

import os
import time
from typing import Dict, Any, Optional, List, Union
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Import for timeout error handling
try:
    from botocore.exceptions import ReadTimeoutError, ClientError
except ImportError:
    # Fallback if botocore is not available
    ReadTimeoutError = Exception
    ClientError = Exception

# Try to import cdao (optional, only needed for OpenAI models in internal environments)
try:
    import cdao
    CDAO_AVAILABLE = True
except ImportError:
    CDAO_AVAILABLE = False
    cdao = None

try:
    from .utils import render_translation_prompt
    from .vars import language_id2name, model_name2openai_id
except ImportError:
    # For direct script execution
    from utils import render_translation_prompt
    from vars import language_id2name, model_name2openai_id


class TranslationState(TypedDict):
    """State for the translation workflow."""
    source_text: str
    source_lang: str
    target_lang: str
    terminology: Optional[Dict[str, list]]
    translation: str
    use_terminology: bool


class ChatCDAO:
    """
    Wrapper class for cdao Azure OpenAI client that mimics ChatBedrock interface.
    This allows workflows to use OpenAI models (via cdao) without code changes.
    """
    
    def __init__(self, model_id: str, temperature: float = 0.0):
        """
        Initialize cdao client.
        
        Args:
            model_id: OpenAI model ID (e.g., "gpt-4.1-mini-2025-04-14")
            temperature: Sampling temperature (default: 0.0)
        """
        if not CDAO_AVAILABLE:
            raise ImportError(
                "cdao library is not available. "
                "OpenAI models require cdao to be installed in internal environments."
            )
        
        self.model_id = model_id
        self.temperature = temperature
        # Lazy initialization: create client on first use (may help with environment timing)
        self._client = None
        # Store response metadata for token counting
        self._last_response_metadata = {}
    
    @property
    def client(self):
        """Lazy initialization of cdao client."""
        if self._client is None:
            self._client = cdao.azure_openai_client(api_version='2024-12-01-preview')
        return self._client
    
    def invoke(self, messages: List[BaseMessage]) -> 'CDAOResponse':
        """
        Invoke the model with messages.
        
        Args:
            messages: List of LangChain messages (HumanMessage, AIMessage, etc.)
        
        Returns:
            CDAOResponse object with .content and .response_metadata
        """
        # Convert LangChain messages to cdao format
        cdao_messages = []
        for msg in messages:
            # Get content, handling None or empty cases
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            
            # Skip empty messages
            if not content or not str(content).strip():
                continue
            
            # Determine role based on message type
            if isinstance(msg, HumanMessage):
                cdao_messages.append({"role": "user", "content": str(content)})
            elif isinstance(msg, AIMessage):
                cdao_messages.append({"role": "assistant", "content": str(content)})
            else:
                # Fallback: default to user role
                cdao_messages.append({"role": "user", "content": str(content)})
        
        # Ensure we have at least one message
        if not cdao_messages:
            raise ValueError("No messages provided to ChatCDAO.invoke()")
        
        # Call cdao (exactly matching the working example)
        # Note: The error "Provider gpt-4 model does not support chat" suggests
        # the model ID might be parsed incorrectly by Azure OpenAI SDK
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=cdao_messages,
                temperature=self.temperature
            )
        except Exception as e:
            # Re-raise with context
            error_msg = str(e)
            raise RuntimeError(
                f"cdao chat completion failed for model {self.model_id}: {error_msg}"
            ) from e
        
        # Extract content
        content = response.choices[0].message.content
        
        # Approximate token counts (cdao doesn't provide usage info)
        # Estimate: ~4 characters per token (rough approximation)
        total_chars = sum(len(msg.get("content", "")) for msg in cdao_messages)
        prompt_tokens = total_chars // 4
        completion_tokens = len(content) // 4
        
        # Store metadata
        self._last_response_metadata = {
            'token_usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
        }
        
        return CDAOResponse(content, self._last_response_metadata)
    
    @property
    def response_metadata(self):
        """Get response metadata (for compatibility with ChatBedrock interface)."""
        return self._last_response_metadata


class CDAOResponse:
    """
    Response object that mimics ChatBedrock response interface.
    """
    
    def __init__(self, content: str, response_metadata: Dict[str, Any]):
        self.content = content
        self.response_metadata = response_metadata


def create_llm(
    model_id: str, 
    region: Optional[str] = None, 
    temperature: float = 0.0,
    model_provider: Optional[str] = None,
    model_type: Optional[str] = None  # "bedrock" or "openai", auto-detect if None
) -> Union[ChatBedrock, ChatCDAO]:
    """
    Create an LLM instance (Bedrock or OpenAI via cdao) with configurable temperature.
    Automatically detects model type if not specified.
    
    Args:
        model_id: Model ID (Bedrock model ID/ARN or OpenAI model ID)
        region: AWS region (only used for Bedrock, defaults based on model type)
        temperature: Sampling temperature (default: 0.0 for reproducibility)
        model_provider: Provider name (e.g., "anthropic") - required when using Bedrock ARNs
        model_type: "bedrock" or "openai" - auto-detected if None
    
    Returns:
        ChatBedrock or ChatCDAO instance
    """
    # Auto-detect model type if not specified
    if model_type is None:
        # Check if it's an OpenAI model ID
        if model_id in model_name2openai_id.values():
            model_type = "openai"
        elif model_id.startswith("arn:aws:bedrock:"):
            model_type = "bedrock"
        else:
            # Default to bedrock for backward compatibility
            model_type = "bedrock"
    
    if model_type == "openai":
        return create_openai_llm(model_id, temperature)
    else:
        return create_bedrock_llm(model_id, region, temperature, model_provider)


def create_openai_llm(
    model_id: str,
    temperature: float = 0.0
) -> ChatCDAO:
    """
    Create an OpenAI LLM instance via cdao with configurable temperature.
    
    Args:
        model_id: OpenAI model ID (e.g., "gpt-4.1-mini-2025-04-14")
        temperature: Sampling temperature (default: 0.0 for reproducibility)
    
    Returns:
        ChatCDAO instance
    """
    return ChatCDAO(model_id, temperature)


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
    
    # If using ARN and model_provider not provided, raise error
    if is_arn and not model_provider:
        raise ValueError(
            f"model_provider is required when using ARN as model_id. "
            f"Received ARN: {model_id}, model_provider: {model_provider}"
        )
    
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
    
    # Add provider for ARNs
    # Based on LangChain ChatBedrock docs, when using ARNs, provider should be top-level
    # The warning "model_provider was transferred to model_kwargs" is misleading - 
    # it's actually expected as a top-level parameter called "provider" or "model_provider"
    if model_provider:
        # Try "provider" first (common name in AWS SDKs)
        bedrock_kwargs["provider"] = model_provider
        # Also set "model_provider" in case that's what it expects
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


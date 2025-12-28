"""
MAATS Multi-Agent workflow: Zero-shot -> 7 specialized MQM evaluators -> Refinement.
Based on "MAATS: A Multi-Agent Automated Translation System Based on MQM Evaluation"

Workflow:
1. Zero-shot translation
2. 7 specialized agents (each evaluates one dimension independently):
   - Terminology
   - Accuracy
   - Linguistic Conventions
   - Locale Conventions
   - Design and Markup
   - Style
   - Audience Appropriateness
3. Refinement agent (receives ALL annotations from all 7 agents)
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage
import json

try:
    from ..translation import create_bedrock_llm, create_llm
    from ..utils import load_template, get_language_name, render_translation_prompt, format_terminology_dict, filter_terminology_by_source_text
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm, create_llm
    from utils import load_template, get_language_name, render_translation_prompt, format_terminology_dict, filter_terminology_by_source_text
    from vars import language_id2name


# Dimension agents in order
DIMENSION_AGENTS = [
    "terminology",
    "accuracy",
    "linguistic_conventions",
    "locale_conventions",
    "design_and_markup",
    "style",
    "audience_appropriateness"
]


def render_dimension_prompt(
    dimension: str,
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str
) -> str:
    """Render prompt for a specific dimension agent."""
    template = load_template(f"MAATS/{dimension}.jinja")
    return template.render(
        source_text=source_text,
        translation=translation,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name)
    )


def render_refine_prompt(
    source_text: str,
    translation: str,
    annotations: Dict[str, str],
    source_lang: str,
    target_lang: str,
    terminology: Optional[Dict[str, list]] = None,
    use_terminology: bool = False
) -> str:
    """Render refinement prompt with all annotations (with optional terminology)."""
    # Use terminology template if use_terminology is True and terminology is available
    if use_terminology and terminology:
        template = load_template("MAATS/refine_term.jinja")
        # Format and filter terminology if available
        formatted_terminology = format_terminology_dict(terminology, source_lang, target_lang, max_terms=50)
        if formatted_terminology:
            formatted_terminology = filter_terminology_by_source_text(
                formatted_terminology, source_text, case_sensitive=False
            )
    else:
        template = load_template("MAATS/refine.jinja")
        formatted_terminology = None
    
    # Format annotations as JSON for clarity
    annotations_text = json.dumps(annotations, indent=2, ensure_ascii=False)
    return template.render(
        source_text=source_text,
        translation=translation,
        annotations=annotations_text,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name),
        terminology=formatted_terminology
    )


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
    model_provider: Optional[str] = None,
    model_type: Optional[str] = None,
    base_translation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run MAATS multi-agent translation workflow.
    
    Args:
        source_text: Source text to translate
        source_lang: Source language code
        target_lang: Target language code
        model_id: Bedrock model ID
        terminology: Optional terminology dictionary (used in zero-shot translation if use_terminology=True)
        use_terminology: If True, use terminology dictionary in zero-shot translation step
        region: AWS region
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff delay
        reference: Optional reference translation (not used in this workflow)
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs [zero_shot, terminology_annotation, accuracy_annotation, ..., refined_translation]
        - 'tokens_input': Total input tokens
        - 'tokens_output': Total output tokens
        - 'latency': Total workflow time in seconds
    """
    import time
    
    # Create LLM (supports both Bedrock and OpenAI via cdao)
    if model_type:
        llm = create_llm(model_id, region, model_provider=model_provider, model_type=model_type)
    else:
        llm = create_bedrock_llm(model_id, region, model_provider=model_provider)
    
    total_tokens_input = 0
    total_tokens_output = 0
    start_time = time.time()
    outputs = []
    
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    # Step 1: Zero-shot translation (with terminology if use_terminology=True)
    # Use base_translation if provided, otherwise generate translation
    if base_translation is not None:
        print("    [Agent 0/9] Using base translation (skipping zero-shot translation step)...")
        translation = base_translation
        # Filter terminology for later use in refinement
        filtered_terminology = None
        if use_terminology and terminology:
            filtered_terminology = format_terminology_dict(terminology, source_lang, target_lang, max_terms=50)
            if filtered_terminology:
                filtered_terminology = filter_terminology_by_source_text(
                    filtered_terminology, source_text, case_sensitive=False
                )
        # No tokens consumed for base translation (already generated)
        outputs.append(translation)
    else:
        print("    [Agent 0/9] Zero-shot translation...")
        
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
        
        zero_shot_prompt = render_translation_prompt(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            language_id2name=language_id2name,
            use_terminology=filtered_terminology is not None,
            terminology=filtered_terminology,
            max_terms=None if filtered_terminology else 50
        )
        
        translation = None
        for attempt in range(max_retries + 1):
            try:
                message = HumanMessage(content=zero_shot_prompt)
                response = llm.invoke([message])
                
                translation = response.content.strip()
                
                # Get token counts
                tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                
                if tokens_input == 0:
                    tokens_input = len(zero_shot_prompt) // 4
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
                    raise RuntimeError(f"Zero-shot translation failed after {max_retries + 1} attempts due to timeout") from e
        
        if translation is None:
            raise RuntimeError("Zero-shot translation step failed")
        
        outputs.append(translation)
    
    # Steps 2-8: Dimension evaluator agents (each evaluates independently)
    annotations = {}
    
    for i, dimension in enumerate(DIMENSION_AGENTS, start=1):
        print(f"    [Agent {i}/9] {dimension.replace('_', ' ').title()} evaluation...")
        dimension_prompt = render_dimension_prompt(
            dimension=dimension,
            source_text=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        annotation = None
        for attempt in range(max_retries + 1):
            try:
                message = HumanMessage(content=dimension_prompt)
                response = llm.invoke([message])
                
                annotation = response.content.strip()
                
                # Get token counts
                tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                
                if tokens_input == 0:
                    tokens_input = len(dimension_prompt) // 4
                if tokens_output == 0:
                    tokens_output = len(annotation) // 4
                
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
                    raise RuntimeError(f"{dimension} evaluation failed after {max_retries + 1} attempts due to timeout") from e
        
        if annotation is None:
            annotation = f"No annotation available for {dimension}."
        
        annotations[dimension] = annotation
        outputs.append(annotation)
    
    # Step 9: Refinement agent (receives ALL annotations)
    print("    [Agent 9/9] Refinement...")
    refine_prompt = render_refine_prompt(
        source_text=source_text,
        translation=translation,
        annotations=annotations,
        source_lang=source_lang,
        target_lang=target_lang,
        terminology=filtered_terminology,
        use_terminology=use_terminology
    )
    
    refined_translation = None
    for attempt in range(max_retries + 1):
        try:
            message = HumanMessage(content=refine_prompt)
            response = llm.invoke([message])
            
            refined_translation = response.content.strip()
            
            # Get token counts
            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
            
            if tokens_input == 0:
                tokens_input = len(refine_prompt) // 4
            if tokens_output == 0:
                tokens_output = len(refined_translation) // 4
            
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
        "outputs": outputs,  # [zero_shot, term_annotation, accuracy_annotation, ..., refined]
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


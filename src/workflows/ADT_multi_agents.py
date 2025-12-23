"""
ADT (AgentDiscoTrans) workflow: Discourse-level document translation.
Based on "AgentDiscoTrans: Agentic LLMs for Discourse-level Machine Translation"

Implements Algorithm 1 from the paper:
- Discourse Agent: Segments document into coherent discourse units
- Translation Agent: Translates each discourse using Memory
- Memory Agent: Updates Memory with proper nouns, phrases, discourse markers

Workflow processes documents discourse-by-discourse (not sentence-by-sentence).
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage
import json
import re

try:
    from ..translation import create_bedrock_llm, create_llm
    from ..utils import load_template, get_language_name, format_terminology_dict, filter_terminology_by_source_text
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm, create_llm
    from utils import load_template, get_language_name, format_terminology_dict, filter_terminology_by_source_text
    from vars import language_id2name


# Default parameters (from paper)
DEFAULT_MAX_DISCOURSE_LENGTH = 1024  # Maximum token length per discourse (L_max)


def split_sentences(text: str, lang: str) -> List[str]:
    """
    Split text into sentences based on language.
    Uses pysbd for most languages, jieba for Chinese.
    """
    if lang.startswith("zh"):
        # Use jieba for Chinese segmentation
        try:
            import jieba
            sentences = list(jieba.cut(text, cut_all=False))
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except ImportError:
            print("    ⚠ Warning: jieba not installed. Install with: pip install jieba")
            # Fallback: split by Chinese punctuation
            sentences = re.split(r'[。！？\n]', text)
            return [s.strip() for s in sentences if s.strip()]
    else:
        try:
            import pysbd
            segmenter = pysbd.Segmenter(language=lang[:2], clean=True)
            sentences = segmenter.segment(text)
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except ImportError:
            print("    ⚠ Warning: pysbd not installed. Install with: pip install pysbd")
            # Fallback: split by periods and newlines
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            if not sentences:
                parts = re.split(r'\.\s+', text)
                sentences = [s.strip() + '.' for s in parts if s.strip()]
            return sentences
        except Exception as e:
            print(f"    ⚠ Warning: Sentence segmentation failed: {e}. Using fallback.")
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            if not sentences:
                parts = re.split(r'\.\s+', text)
                sentences = [s.strip() + '.' for s in parts if s.strip()]
            return sentences


def estimate_token_count(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
    return len(text) // 4


def render_discourse_agent_prompt(
    current_discourse: str,
    next_sentence: str,
    source_lang: str,
    max_discourse_length: int = DEFAULT_MAX_DISCOURSE_LENGTH
) -> str:
    """Render prompt for Discourse Agent (decides if next sentence should be included)."""
    template = load_template("ADT/discourse_agent.jinja")
    return template.render(
        current_discourse=current_discourse,
        next_sentence=next_sentence,
        source_lang_name=get_language_name(source_lang, language_id2name),
        max_discourse_length=max_discourse_length
    )


def render_translation_agent_prompt(
    discourse: str,
    memory: Dict[str, Any],
    source_lang: str,
    target_lang: str,
    terminology: Optional[Dict[str, list]] = None,
    use_terminology: bool = False
) -> str:
    """Render prompt for Translation Agent (translates discourse using Memory)."""
    if use_terminology and terminology:
        template = load_template("ADT/translation_agent_term.jinja")
        # Format and filter terminology if available
        formatted_terminology = format_terminology_dict(terminology, source_lang, target_lang, max_terms=50)
        if formatted_terminology:
            formatted_terminology = filter_terminology_by_source_text(
                formatted_terminology, discourse, case_sensitive=False
            )
    else:
        template = load_template("ADT/translation_agent.jinja")
        formatted_terminology = None
    
    # Format memory as JSON string for the prompt
    memory_json = json.dumps(memory, indent=2, ensure_ascii=False)
    
    return template.render(
        discourse=discourse,
        memory_json=memory_json,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name),
        terminology=formatted_terminology
    )


def render_memory_agent_prompt(
    current_memory: Dict[str, Any],
    source_discourse: str,
    target_discourse: str
) -> str:
    """Render prompt for Memory Agent (updates Memory based on discourse translation)."""
    template = load_template("ADT/memory_agent.jinja")
    # Format current memory as JSON
    memory_json = json.dumps(current_memory, indent=2, ensure_ascii=False)
    return template.render(
        current_memory_json=memory_json,
        source_discourse=source_discourse,
        target_discourse=target_discourse
    )


def parse_memory_update(response: str) -> Dict[str, Any]:
    """
    Parse Memory Agent response to extract updated memory.
    Expected format: JSON object with proper_noun_references, phrase_consistency, discourse_markers
    """
    try:
        # Try to extract JSON from response
        # Look for JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            memory_dict = json.loads(json_match.group(0))
            # Ensure all required keys exist
            result = {
                "proper_noun_references": memory_dict.get("proper_noun_references", {}),
                "phrase_consistency": memory_dict.get("phrase_consistency", {}),
                "discourse_markers": memory_dict.get("discourse_markers", [])
            }
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback: return empty memory structure
    return {
        "proper_noun_references": {},
        "phrase_consistency": {},
        "discourse_markers": []
    }


def discourse_segmentation(
    document: str,
    source_lang: str,
    llm,
    max_discourse_length: int = DEFAULT_MAX_DISCOURSE_LENGTH,
    max_retries: int = 3,
    initial_backoff: float = 2.0
) -> List[str]:
    """
    Algorithm 2: Discourse Agent - Segment document into discourses.
    
    Input: Document D (string), maximum token length L_max
    Output: List of discourses DS = [d_1, d_2, ..., d_n]
    """
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    # Step 1: Break document into sentences
    sentences = split_sentences(document, source_lang)
    print(f"    Split document into {len(sentences)} sentences")
    
    if not sentences:
        return [document]  # Fallback: return entire document as single discourse
    
    discourses = []
    st = 0  # Start index (0-based)
    
    while st < len(sentences):
        # Initialize current discourse with first sentence
        current_discourse = sentences[st]
        en = st + 1
        
        # Try to include more sentences
        while en < len(sentences):
            next_sentence = sentences[en]
            
            # Check token length constraint
            candidate_discourse = current_discourse + " " + next_sentence
            if estimate_token_count(candidate_discourse) > max_discourse_length:
                break  # Would exceed max length
            
            # Ask Discourse Agent if next sentence should be included
            discourse_prompt = render_discourse_agent_prompt(
                current_discourse=current_discourse,
                next_sentence=next_sentence,
                source_lang=source_lang,
                max_discourse_length=max_discourse_length
            )
            
            should_include = False
            for attempt in range(max_retries + 1):
                try:
                    message = HumanMessage(content=discourse_prompt)
                    response = llm.invoke([message])
                    response_text = response.content.strip().lower()
                    
                    # Parse response: look for "yes" or "no"
                    if "yes" in response_text and "no" not in response_text:
                        should_include = True
                    elif "no" in response_text:
                        should_include = False
                    else:
                        # Default: include if response is unclear
                        should_include = True
                    
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
                        import time
                        backoff_time = (2 ** attempt) * initial_backoff
                        print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                        time.sleep(backoff_time)
                    else:
                        # On final failure, default to including the sentence
                        should_include = True
                        print(f"    ⚠ Discourse Agent failed after {max_retries + 1} attempts, defaulting to include sentence")
            
            if should_include:
                current_discourse = candidate_discourse
                en += 1
            else:
                break  # Start new discourse
        
        # Add completed discourse to list
        discourses.append(current_discourse)
        st = en
    
    print(f"    Segmented into {len(discourses)} discourse(s)")
    return discourses


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
    max_discourse_length: int = DEFAULT_MAX_DISCOURSE_LENGTH,
    model_provider: Optional[str] = None,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Algorithm 1: Agentic Document-Level Translation Workflow.
    
    Input: Document D = ⟨s_1, s_2, ..., s_k⟩, source language, target language
    Output: Translation T = ⟨t_1, t_2, ..., t_n⟩
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
    
    # Step 1: Initialize memory K ← ∅
    memory = {
        "proper_noun_references": {},
        "phrase_consistency": {},
        "discourse_markers": []
    }
    
    # Step 2: DS ← DiscourseAgent(D)
    print("    [Step 1/3] Discourse Segmentation...")
    discourses = discourse_segmentation(
        document=source_text,
        source_lang=source_lang,
        llm=llm,
        max_discourse_length=max_discourse_length,
        max_retries=max_retries,
        initial_backoff=initial_backoff
    )
    
    # Step 3: For each discourse d_i ∈ DS
    print(f"    [Step 2/3] Translating {len(discourses)} discourse(s)...")
    translations = []
    
    for i, discourse in enumerate(discourses):
        print(f"      Discourse {i+1}/{len(discourses)} ({len(discourse)} chars)...")
        
        # Step 3a: t_i ← TranslationAgent(d_i, K)
        translation_prompt = render_translation_agent_prompt(
            discourse=discourse,
            memory=memory,
            source_lang=source_lang,
            target_lang=target_lang,
            terminology=terminology,
            use_terminology=use_terminology
        )
        
        translation = None
        for attempt in range(max_retries + 1):
            try:
                message = HumanMessage(content=translation_prompt)
                response = llm.invoke([message])
                
                translation = response.content.strip()
                
                tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                
                if tokens_input == 0:
                    tokens_input = len(translation_prompt) // 4
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
                    print(f"        ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
                else:
                    raise RuntimeError(f"Translation failed after {max_retries + 1} attempts due to timeout") from e
        
        if translation is None:
            raise RuntimeError(f"Translation failed for discourse {i+1}")
        
        translations.append(translation)
        outputs.append(translation)
        
        # Step 3b: K ← MemoryAgent(K, d_i, t_i)
        print(f"      Updating memory...")
        memory_prompt = render_memory_agent_prompt(
            current_memory=memory,
            source_discourse=discourse,
            target_discourse=translation
        )
        
        memory_updated = False
        for attempt in range(max_retries + 1):
            try:
                message = HumanMessage(content=memory_prompt)
                response = llm.invoke([message])
                
                updated_memory = parse_memory_update(response.content.strip())
                
                # Merge with existing memory (update, don't replace)
                memory["proper_noun_references"].update(updated_memory.get("proper_noun_references", {}))
                memory["phrase_consistency"].update(updated_memory.get("phrase_consistency", {}))
                # Append new discourse markers (avoid duplicates)
                new_markers = updated_memory.get("discourse_markers", [])
                for marker in new_markers:
                    if marker not in memory["discourse_markers"]:
                        memory["discourse_markers"].append(marker)
                
                tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                
                if tokens_input == 0:
                    tokens_input = len(memory_prompt) // 4
                if tokens_output == 0:
                    tokens_output = len(response.content) // 4
                
                total_tokens_input += tokens_input
                total_tokens_output += tokens_output
                
                memory_updated = True
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
                    print(f"        ⚠ Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
                else:
                    print(f"        ⚠ Memory update failed after {max_retries + 1} attempts, continuing with existing memory")
                    break
        
        if not memory_updated:
            print(f"        ⚠ Warning: Memory update failed, continuing with existing memory")
    
    # Step 4: T ← concatenate(t_1, t_2, ..., t_n)
    print("    [Step 3/3] Concatenating translations...")
    final_translation = "\n\n".join(translations)  # Join with double newline for readability
    outputs.append(final_translation)  # Add final concatenated translation as last output
    
    latency = time.time() - start_time
    
    return {
        "outputs": outputs,  # Individual discourse translations + final concatenated translation
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


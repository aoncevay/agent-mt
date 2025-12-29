"""
DeLTA (Document-levEL Translation Agent) workflow.
Based on "DELTA: AN ONLINE DOCUMENT-LEVEL TRANSLATION AGENT BASED ON MULTI-LEVEL MEMORY"

Implements Algorithm 1 from the paper:
- Proper Noun Records (R): Maintains consistency of proper noun translations
- Bilingual Summary (As, At): Source and target summaries for coherence
- Long-Term Memory (N): Broader context retrieval
- Short-Term Memory (M): Immediate context (last k sentences)

Workflow processes documents sentence-by-sentence (online approach).
"""

from typing import Dict, Any, Optional, List, Tuple
from langchain_core.messages import HumanMessage
import re
import random

try:
    from ..translation import create_bedrock_llm, create_llm
    from ..utils import load_template, get_language_name, format_terminology_dict, filter_terminology_by_source_text
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm, create_llm
    from utils import load_template, get_language_name, format_terminology_dict, filter_terminology_by_source_text
    from vars import language_id2name


# Default parameters (can be adjusted)
DEFAULT_SUMMARY_STEP = 10  # Update summary every m sentences
DEFAULT_LONG_WINDOW = 10    # Long-term memory size (l)
DEFAULT_SHORT_WINDOW = 3    # Short-term memory size (k)
DEFAULT_TOP_K = 2           # Number of relevant sentences to retrieve from long-term memory


def parse_proper_nouns(response: str) -> List[Tuple[str, str]]:
    """
    Parse proper noun pairs from the extractor response.
    Format: "NASA" - "美国国家航空航天局", "Kepler" - "开普勒"
    Returns list of (source_noun, target_noun) tuples.
    """
    if not response or response.strip() in ['N/A', 'None', '', '无']:
        return []
    
    pairs = []
    # Split by comma, but handle quoted strings
    parts = re.split(r',\s*(?=")', response)
    for part in parts:
        part = part.strip()
        if ' - ' in part:
            try:
                src_ent, tgt_ent = part.split(' - ', 1)
                src_ent = src_ent.strip().strip('"').strip("'")
                tgt_ent = tgt_ent.strip().strip('"').strip("'")
                if src_ent and tgt_ent and tgt_ent != 'N/A':
                    pairs.append((src_ent, tgt_ent))
            except ValueError:
                continue
    return pairs


def parse_retrieved_sentences(response: str, max_num: int) -> List[int]:
    """
    Parse sentence indices from memory retriever response.
    Expected format: [1, 3, 5] or "1, 3, 5"
    """
    if not response:
        return []
    
    try:
        # Try to evaluate as Python list
        indices = eval(response)
        if isinstance(indices, list):
            return [i for i in indices if isinstance(i, int) and i >= 1]
    except:
        # Try to parse as comma-separated numbers
        numbers = re.findall(r'\d+', response)
        indices = [int(n) for n in numbers if int(n) >= 1]
        return indices[:max_num]
    
    return []


def render_proper_noun_extractor_prompt(
    source_sentence: str,
    target_sentence: str,
    source_lang: str,
    target_lang: str
) -> str:
    """Render prompt for proper noun extraction."""
    template = load_template("DeLTA/proper_noun_extractor.jinja")
    return template.render(
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name)
    )


def render_source_summary_writer_prompt(source_paragraph: str) -> str:
    """Render prompt for source summary generation."""
    template = load_template("DeLTA/source_summary_writer.jinja")
    return template.render(source_paragraph=source_paragraph)


def render_target_summary_writer_prompt(target_paragraph: str) -> str:
    """Render prompt for target summary generation."""
    template = load_template("DeLTA/target_summary_writer.jinja")
    return template.render(target_paragraph=target_paragraph)


def render_source_summary_merger_prompt(summary_1: str, summary_2: str) -> str:
    """Render prompt for merging source summaries."""
    template = load_template("DeLTA/source_summary_merger.jinja")
    return template.render(summary_1=summary_1, summary_2=summary_2)


def render_target_summary_merger_prompt(summary_1: str, summary_2: str) -> str:
    """Render prompt for merging target summaries."""
    template = load_template("DeLTA/target_summary_merger.jinja")
    return template.render(summary_1=summary_1, summary_2=summary_2)


def render_memory_retriever_prompt(
    sentence_list: List[str],
    query: str,
    top_num: int
) -> str:
    """Render prompt for memory retrieval."""
    template = load_template("DeLTA/memory_retriever.jinja")
    
    # Format sentence list
    sent_list = '\n'.join([f'<Sentence {idx + 1}> {sent}' for idx, sent in enumerate(sentence_list)])
    
    # Generate example (for demonstration)
    random.seed(0)
    example_indices = random.sample(list(range(max(10, top_num))), min(top_num, max(10, top_num)))
    example_indices.sort()
    example_number = ', '.join([str(i) for i in example_indices[:-1]]) + f' and {example_indices[-1]}' if top_num > 1 else str(example_indices[0])
    example_list = str(example_indices)
    
    return template.render(
        sentence_list=sent_list,
        query=query,
        top_num=top_num,
        example_number=example_number,
        example_list=example_list
    )


def render_document_translator_prompt(
    source_sentence: str,
    source_lang: str,
    target_lang: str,
    source_summary: str,
    target_summary: str,
    proper_noun_records: str,
    source_context: str,
    target_context: str,
    relevant_instances: str,
    terminology: Optional[Dict[str, list]] = None,
    use_terminology: bool = False
) -> str:
    """Render prompt for document translation (with optional terminology)."""
    if use_terminology and terminology:
        template = load_template("DeLTA/document_translator_term.jinja")
        # Format and filter terminology if available
        formatted_terminology = format_terminology_dict(terminology, source_lang, target_lang, max_terms=50)
        if formatted_terminology:
            formatted_terminology = filter_terminology_by_source_text(
                formatted_terminology, source_sentence, case_sensitive=False
            )
        return template.render(
            source_sentence=source_sentence,
            source_lang_name=get_language_name(source_lang, language_id2name),
            target_lang_name=get_language_name(target_lang, language_id2name),
            source_summary=source_summary or "N/A",
            target_summary=target_summary or "N/A",
            proper_noun_records=proper_noun_records or "N/A",
            source_context=source_context or "N/A",
            target_context=target_context or "N/A",
            relevant_instances=relevant_instances or "N/A",
            terminology=formatted_terminology
        )
    else:
        template = load_template("DeLTA/document_translator.jinja")
        return template.render(
            source_sentence=source_sentence,
            source_lang_name=get_language_name(source_lang, language_id2name),
            target_lang_name=get_language_name(target_lang, language_id2name),
            source_summary=source_summary or "N/A",
            target_summary=target_summary or "N/A",
            proper_noun_records=proper_noun_records or "N/A",
            source_context=source_context or "N/A",
            target_context=target_context or "N/A",
            relevant_instances=relevant_instances or "N/A"
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
    summary_step: int = DEFAULT_SUMMARY_STEP,
    long_window: int = DEFAULT_LONG_WINDOW,
    model_provider: Optional[str] = None,
    short_window: int = DEFAULT_SHORT_WINDOW,
    top_k: int = DEFAULT_TOP_K,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run DeLTA document-level translation workflow.
    
    Args:
        source_text: Source text to translate (can be multi-sentence, will be split)
        source_lang: Source language code
        target_lang: Target language code
        model_id: Bedrock model ID
        terminology: Optional terminology dictionary (used in document translator if use_terminology=True)
        use_terminology: If True, use terminology dictionary in document translator step
        region: AWS region
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff delay
        reference: Optional reference translation (not used in this workflow)
        summary_step: Update summary every m sentences (default: 10)
        long_window: Long-term memory size l (default: 10)
        short_window: Short-term memory size k (default: 3)
        top_k: Number of relevant sentences to retrieve (default: 2)
    
    Returns:
        Dictionary with:
        - 'outputs': List of translated sentences
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
    
    try:
        from botocore.exceptions import ReadTimeoutError, ClientError
    except ImportError:
        ReadTimeoutError = Exception
        ClientError = Exception
    
    # Split source text into sentences
    # For Chinese: split by punctuation marks (。！？), for other languages: use pysbd
    def split_sentences_delta(text: str, lang: str) -> List[str]:
        """
        Split sentences based on the language:
        - Chinese (lang starts with 'zh'): Split by punctuation marks (。！？).
        - Other languages: Use pysbd's Segmenter.
        """
        if lang.startswith("zh"):
            # Split by Chinese punctuation marks (correct sentence segmentation)
            # Note: jieba.cut() is for word segmentation, not sentence segmentation
            sentences = re.split(r'[。！？\n]', text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            try:
                import pysbd
                # Use pySBD for sentence segmentation
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
                # Fallback: split by periods and newlines
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
                if not sentences:
                    parts = re.split(r'\.\s+', text)
                    sentences = [s.strip() + '.' for s in parts if s.strip()]
                return sentences
    
    source_sentences = split_sentences_delta(source_text, source_lang)
    
    if not source_sentences:
        raise ValueError("No sentences found in source text after segmentation")
    
    # Initialize memory components (Algorithm 1)
    proper_noun_records: Dict[str, str] = {}  # R: {proper_noun: translation}
    source_summary: Optional[str] = None      # As
    target_summary: Optional[str] = None      # At
    long_term_memory: List[Tuple[str, str]] = []  # N: [(src, tgt), ...]
    short_term_memory: List[Tuple[str, str]] = []  # M: [(src, tgt), ...]
    
    translated_sentences = []
    segment_buffer = []  # Buffer for summary generation
    
    print(f"    Processing {len(source_sentences)} sentences with DeLTA workflow...")
    
    for i, source_sentence in enumerate(source_sentences):
        # Step 1: Retrieve memory (Algorithm 1, lines 337-338)
        # R^: Proper nouns in current sentence that are in records
        relevant_proper_nouns = {
            p: proper_noun_records[p] 
            for p in proper_noun_records 
            if p in source_sentence
        }
        proper_noun_prompt = ', '.join([f'"{p}" - "{t}"' for p, t in relevant_proper_nouns.items()]) if relevant_proper_nouns else "N/A"
        
        # N^: Retrieve relevant sentences from Long-Term Memory
        relevant_srcs, relevant_tgts = [], []
        if long_term_memory and len(long_term_memory) > top_k:
            retriever_prompt = render_memory_retriever_prompt(
                sentence_list=[src for src, _ in long_term_memory],
                query=source_sentence,
                top_num=top_k
            )
            
            # Call retriever
            retrieved_indices = []
            for attempt in range(max_retries + 1):
                try:
                    message = HumanMessage(content=retriever_prompt)
                    response = llm.invoke([message])
                    response_text = response.content.strip()
                    retrieved_indices = parse_retrieved_sentences(response_text, top_k)
                    
                    tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                    tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                    
                    if tokens_input == 0:
                        tokens_input = len(retriever_prompt) // 4
                    if tokens_output == 0:
                        tokens_output = len(response_text) // 4
                    
                    total_tokens_input += tokens_input
                    total_tokens_output += tokens_output
                    
                    break
                except (ReadTimeoutError, ClientError) as e:
                    if attempt < max_retries:
                        backoff_time = (2 ** attempt) * initial_backoff
                        time.sleep(backoff_time)
                    else:
                        # Fallback: use last top_k sentences
                        retrieved_indices = list(range(max(1, len(long_term_memory) - top_k + 1), len(long_term_memory) + 1))
            
            # Get retrieved sentences (indices are 1-based)
            for idx in retrieved_indices:
                if 1 <= idx <= len(long_term_memory):
                    src, tgt = long_term_memory[idx - 1]
                    relevant_srcs.append(src)
                    relevant_tgts.append(tgt)
        elif long_term_memory:
            # If memory is smaller than top_k, use all
            relevant_srcs, relevant_tgts = zip(*long_term_memory) if long_term_memory else ([], [])
        
        relevant_instances = '\n'.join([
            f'<{get_language_name(source_lang, language_id2name)}> {src}\n<{get_language_name(target_lang, language_id2name)}> {tgt}'
            for src, tgt in zip(relevant_srcs, relevant_tgts)
        ]) if relevant_srcs else "N/A"
        
        # Short-term memory context
        src_context = '\n'.join([src for src, _ in short_term_memory]) if short_term_memory else "N/A"
        tgt_context = '\n'.join([tgt for _, tgt in short_term_memory]) if short_term_memory else "N/A"
        
        # Step 2: Translate with hybrid memory information (Algorithm 1, line 340)
        translate_prompt = render_document_translator_prompt(
            source_sentence=source_sentence,
            source_lang=source_lang,
            target_lang=target_lang,
            source_summary=source_summary or "",
            target_summary=target_summary or "",
            proper_noun_records=proper_noun_prompt,
            source_context=src_context,
            target_context=tgt_context,
            relevant_instances=relevant_instances
        )
        
        target_sentence = None
        for attempt in range(max_retries + 1):
            try:
                message = HumanMessage(content=translate_prompt)
                response = llm.invoke([message])
                
                target_sentence = response.content.strip()
                # Take first line if multiple lines
                if '\n' in target_sentence:
                    target_sentence = target_sentence.split('\n')[0].strip()
                
                tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                
                if tokens_input == 0:
                    tokens_input = len(translate_prompt) // 4
                if tokens_output == 0 and target_sentence:
                    tokens_output = len(target_sentence) // 4
                
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
                    time.sleep(backoff_time)
                else:
                    raise RuntimeError(f"Translation failed for sentence {i+1} after {max_retries + 1} attempts") from e
        
        if target_sentence is None:
            raise RuntimeError(f"Translation failed for sentence {i+1}")
        
        translated_sentences.append(target_sentence)
        
        # Step 3: Update memory (Algorithm 1, lines 343-345)
        # Extract new proper nouns
        extract_prompt = render_proper_noun_extractor_prompt(
            source_sentence=source_sentence,
            target_sentence=target_sentence,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        new_proper_nouns = []
        for attempt in range(max_retries + 1):
            try:
                message = HumanMessage(content=extract_prompt)
                response = llm.invoke([message])
                response_text = response.content.strip()
                new_proper_nouns = parse_proper_nouns(response_text)
                
                tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                
                if tokens_input == 0:
                    tokens_input = len(extract_prompt) // 4
                if tokens_output == 0:
                    tokens_output = len(response_text) // 4
                
                total_tokens_input += tokens_input
                total_tokens_output += tokens_output
                
                break
            except (ReadTimeoutError, ClientError) as e:
                if attempt < max_retries:
                    backoff_time = (2 ** attempt) * initial_backoff
                    time.sleep(backoff_time)
                else:
                    # Continue without proper nouns if extraction fails
                    break
        
        # Add new proper nouns to records (only if not already present)
        for src_noun, tgt_noun in new_proper_nouns:
            if src_noun not in proper_noun_records:
                proper_noun_records[src_noun] = tgt_noun
        
        # Update Long-Term Memory (last l sentences)
        long_term_memory.append((source_sentence, target_sentence))
        if len(long_term_memory) > long_window:
            long_term_memory = long_term_memory[-long_window:]
        
        # Update Short-Term Memory (last k sentences)
        short_term_memory.append((source_sentence, target_sentence))
        if len(short_term_memory) > short_window:
            short_term_memory = short_term_memory[-short_window:]
        
        # Step 4: Update summaries every m sentences (Algorithm 1, lines 346-351)
        segment_buffer.append((source_sentence, target_sentence))
        
        if (i + 1) % summary_step == 0 and segment_buffer:
            # Generate segment summaries
            # Filter out any empty sentences from buffer
            src_segment = '\n'.join([src for src, _ in segment_buffer if src and src.strip()])
            tgt_segment = '\n'.join([tgt for _, tgt in segment_buffer if tgt and tgt.strip()])
            
            # Skip summary generation if segments are empty
            if not src_segment.strip() or not tgt_segment.strip():
                segment_buffer = []  # Clear buffer and continue
                continue
            
            # Source summary
            src_sum_prompt = render_source_summary_writer_prompt(src_segment)
            src_seg_summary = None
            for attempt in range(max_retries + 1):
                try:
                    message = HumanMessage(content=src_sum_prompt)
                    response = llm.invoke([message])
                    src_seg_summary = response.content.strip()
                    
                    tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                    tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                    
                    if tokens_input == 0:
                        tokens_input = len(src_sum_prompt) // 4
                    if tokens_output == 0 and src_seg_summary:
                        tokens_output = len(src_seg_summary) // 4
                    
                    total_tokens_input += tokens_input
                    total_tokens_output += tokens_output
                    
                    break
                except (ReadTimeoutError, ClientError) as e:
                    if attempt < max_retries:
                        backoff_time = (2 ** attempt) * initial_backoff
                        time.sleep(backoff_time)
                    else:
                        break
            
            # Target summary
            tgt_sum_prompt = render_target_summary_writer_prompt(tgt_segment)
            tgt_seg_summary = None
            for attempt in range(max_retries + 1):
                try:
                    message = HumanMessage(content=tgt_sum_prompt)
                    response = llm.invoke([message])
                    tgt_seg_summary = response.content.strip()
                    
                    tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                    tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                    
                    if tokens_input == 0:
                        tokens_input = len(tgt_sum_prompt) // 4
                    if tokens_output == 0 and tgt_seg_summary:
                        tokens_output = len(tgt_seg_summary) // 4
                    
                    total_tokens_input += tokens_input
                    total_tokens_output += tokens_output
                    
                    break
                except (ReadTimeoutError, ClientError) as e:
                    if attempt < max_retries:
                        backoff_time = (2 ** attempt) * initial_backoff
                        time.sleep(backoff_time)
                    else:
                        break
            
            # Merge with overall summaries
            if src_seg_summary and tgt_seg_summary:
                if source_summary:
                    # Merge source summary
                    merge_src_prompt = render_source_summary_merger_prompt(source_summary, src_seg_summary)
                    for attempt in range(max_retries + 1):
                        try:
                            message = HumanMessage(content=merge_src_prompt)
                            response = llm.invoke([message])
                            source_summary = response.content.strip()
                            
                            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                            
                            if tokens_input == 0:
                                tokens_input = len(merge_src_prompt) // 4
                            if tokens_output == 0 and source_summary:
                                tokens_output = len(source_summary) // 4
                            
                            total_tokens_input += tokens_input
                            total_tokens_output += tokens_output
                            
                            break
                        except (ReadTimeoutError, ClientError) as e:
                            if attempt < max_retries:
                                backoff_time = (2 ** attempt) * initial_backoff
                                time.sleep(backoff_time)
                            else:
                                source_summary = src_seg_summary  # Fallback
                                break
                else:
                    source_summary = src_seg_summary
                
                if target_summary:
                    # Merge target summary
                    merge_tgt_prompt = render_target_summary_merger_prompt(target_summary, tgt_seg_summary)
                    for attempt in range(max_retries + 1):
                        try:
                            message = HumanMessage(content=merge_tgt_prompt)
                            response = llm.invoke([message])
                            target_summary = response.content.strip()
                            
                            tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
                            tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
                            
                            if tokens_input == 0:
                                tokens_input = len(merge_tgt_prompt) // 4
                            if tokens_output == 0 and target_summary:
                                tokens_output = len(target_summary) // 4
                            
                            total_tokens_input += tokens_input
                            total_tokens_output += tokens_output
                            
                            break
                        except (ReadTimeoutError, ClientError) as e:
                            if attempt < max_retries:
                                backoff_time = (2 ** attempt) * initial_backoff
                                time.sleep(backoff_time)
                            else:
                                target_summary = tgt_seg_summary  # Fallback
                                break
                else:
                    target_summary = tgt_seg_summary
                
                segment_buffer = []  # Clear buffer after summary update
    
    # Combine all translated sentences
    final_translation = '\n'.join(translated_sentences)
    
    latency = time.time() - start_time
    
    return {
        "outputs": [final_translation],  # Single output: full document translation
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }


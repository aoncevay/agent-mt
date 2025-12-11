"""
Utility functions for translation workflows.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, Template


def get_language_name(lang_code: str, language_id2name: Dict[str, str]) -> str:
    """Get full language name from code."""
    return language_id2name.get(lang_code, lang_code)


def filter_terminology_by_source_text(
    terminology: Dict[str, List[str]],
    source_text: str,
    case_sensitive: bool = False
) -> Dict[str, List[str]]:
    """
    Filter terminology dictionary to only include terms that appear in the source text.
    
    Args:
        terminology: Dictionary with source terms as keys, target terms as values
        source_text: Source text to search in
        case_sensitive: Whether to do case-sensitive matching (default: False)
    
    Returns:
        Filtered terminology dictionary containing only terms found in source_text
    """
    if not terminology or not source_text:
        return {}
    
    filtered = {}
    source_lower = source_text if case_sensitive else source_text.lower()
    
    for source_term, target_terms in terminology.items():
        # Check if the source term appears in the text
        search_term = source_term if case_sensitive else source_term.lower()
        
        # Simple substring match (can be improved with word boundary matching if needed)
        if search_term in source_lower:
            filtered[source_term] = target_terms
    
    return filtered


def format_terminology_dict(
    proper_dict: Dict[str, List[str]], 
    source_lang: str, 
    target_lang: str,
    max_terms: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Format terminology dictionary for the translation direction.
    The proper dict keys are in source language, values are in target language.
    """
    if max_terms:
        items = list(proper_dict.items())[:max_terms]
        return dict(items)
    return proper_dict


def load_template(template_name: str, template_dir: Optional[Path] = None) -> Template:
    """Load a Jinja2 template."""
    if template_dir is None:
        template_dir = Path(__file__).parent / "templates"
    
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True
    )
    return env.get_template(template_name)


def render_translation_prompt(
    source_text: str,
    source_lang: str,
    target_lang: str,
    language_id2name: Dict[str, str],
    use_terminology: bool = False,
    terminology: Optional[Dict[str, List[str]]] = None,
    max_terms: int = 50,
    domain_description: str = "part of a financial document.",
    domain_guidance: str = (
        "Ensure the translation is formal, precise, and consistent with "
        "standard financial and corporate communication. Use correct financial "
        "terminology and maintain a neutral, objective tone."
    )
) -> str:
    """Render a translation prompt using Jinja template."""
    if use_terminology:
        template = load_template("translation_with_terminology.jinja")
        formatted_terminology = format_terminology_dict(
            terminology or {}, source_lang, target_lang, max_terms
        )
    else:
        template = load_template("translation_without_terminology.jinja")
        formatted_terminology = None
    
    return template.render(
        source_text=source_text,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name),
        terminology=formatted_terminology,
        domain_description=domain_description,
        domain_guidance=domain_guidance
    )


def render_postedit_prompt(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    language_id2name: Dict[str, str],
    reference: Optional[str] = None
) -> str:
    """Render a postedit prompt using Jinja template."""
    template = load_template("MaMT_postedit.jinja")
    return template.render(
        source_text=source_text,
        translation=translation,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name),
        reference=reference
    )


def render_proofread_prompt(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    language_id2name: Dict[str, str],
    domain: str = "general"
) -> str:
    """Render a proofread prompt using Jinja template."""
    template = load_template("MaMT_proofread.jinja")
    return template.render(
        source_text=source_text,
        translation=translation,
        source_lang_name=get_language_name(source_lang, language_id2name),
        target_lang_name=get_language_name(target_lang, language_id2name),
        domain=domain
    )


def sanitize_response(response_text: str) -> str:
    """
    Sanitize the response text to prepare for JSON parsing.
    Removes markdown code blocks and extra whitespace.
    """
    # Remove markdown code blocks if present
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    # Remove leading/trailing whitespace
    response_text = response_text.strip()
    return response_text


def clean_suggestion(suggestion: str) -> str:
    """
    Clean the suggestion text.
    Removes extra whitespace and normalizes the string.
    """
    if not suggestion:
        return ""
    return suggestion.strip()


def preserve_paragraph(original_text: str, corrected_text: str) -> str:
    """
    Preserve paragraph structure from original text.
    If corrected text loses paragraph breaks, restore them from original.
    """
    # Split by paragraphs (double newlines)
    original_paragraphs = original_text.split('\n\n')
    corrected_paragraphs = corrected_text.split('\n\n')
    
    # If paragraph count matches, preserve structure
    if len(original_paragraphs) == len(corrected_paragraphs):
        return corrected_text
    
    # Otherwise, try to preserve at least the paragraph breaks
    # This is a simple implementation - can be enhanced if needed
    return corrected_text


def parse_postedit_response(response_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the postedit agent's JSON response following Algorithm 1.
    
    Returns:
        Dictionary mapping error spans to their corrections:
        {
            "<error span>": {
                "category": "<category>",
                "severity": <1-4>,
                "suggestion": "<fix>"
            },
            ...
        }
    """
    try:
        # Sanitize response
        safe_response = sanitize_response(response_text)
        
        # Try to extract JSON from the response
        # Look for JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', safe_response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            # If no JSON found, try parsing the whole response
            parsed = json.loads(safe_response)
        
        # Ensure parsed is a dictionary
        if not isinstance(parsed, dict):
            return {}
        
        return parsed
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # If parsing fails, return empty dict
        print(f"    ⚠ Warning: Failed to parse postedit response: {e}")
        return {}


def apply_postedit_corrections(
    translation: str,
    corrections: Dict[str, Dict[str, Any]],
    min_safe_span_len: int = 2
) -> str:
    """
    Apply postedit corrections to translation using Algorithm 1 from the paper.
    
    Implements two substitution strategies:
    1. Space-sensitive substitution: Replace spans only when surrounded by spaces
    2. Fallback substitution: If no replacement occurs, substitute the span wherever it appears
    
    Args:
        translation: Original translation text (tgt_text in Algorithm 1)
        corrections: Dictionary of corrections from parse_postedit_response (parsed in Algorithm 1)
        min_safe_span_len: Minimum length for safe span replacement (MIN_SAFE_SPAN_LEN in Algorithm 1)
    
    Returns:
        Corrected translation
    """
    if not corrections:
        return translation
    
    corrected = translation
    MIN_SAFE_SPAN_LEN = min_safe_span_len
    
    try:
        # Iterate through each span in the parsed corrections
        for span, info in corrections.items():
            # Skip if info is not a dictionary
            if not isinstance(info, dict):
                continue
            
            # Extract and clean suggestion
            suggestion = clean_suggestion(info.get("suggestion", ""))
            
            # Skip conditions from Algorithm 1
            if span.lower() == "no-error":
                continue
            if not suggestion or suggestion == "":
                continue
            if suggestion == span:
                continue
            if len(span) < MIN_SAFE_SPAN_LEN:
                continue
            
            # Strategy 1: Space-sensitive substitution
            # Replace spans only when surrounded by spaces to avoid partial-word errors
            space = " "
            pattern_space = space + re.escape(span) + space
            corrected, count = re.subn(pattern_space, space + suggestion + space, corrected)
            
            if count == 0:
                # Strategy 2: Fallback substitution
                # If no replacement occurs, substitute the span wherever it appears
                pattern_general = re.escape(span)
                corrected, _ = re.subn(pattern_general, suggestion, corrected)
    
    except Exception as e:
        # If any error occurs, reset to original and preserve paragraph structure
        print(f"    ⚠ Warning: Error applying corrections: {e}")
        corrected = translation
        corrected = preserve_paragraph(translation, corrected)
    
    return corrected




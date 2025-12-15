"""
Python script to build a parallel corpus from the IRS dataset.
The folder article contains the articles for different languages: p{article_number}{language_code}.md
The folder index contains the article index that we extracted from the markdown files.
We want to build a parallel corpus of the articles and the indexes, as follows:

1. We will process the articles in language pairs English-X (where X is the target language).
2. We should skip the introductory part of the article, so we will skip the first main chunk of the article.
3. We will start aligning from the second main chunk, and adding subsections only if:
    3.1 In the target language index, there is the same number of subsections as the same hierarchy level in the English index. If there is no subsection in the target language index, then we will not add any subsections.
    3.2 If the English side surpasses 3k tokens, then we will stop adding subsections.
    3.3 Each chunk will have a sequential index starting from 0 (doc_idx)
4. We will save the parallel corpus (in the "parallel_corpus" subfolder) in a jsonl file per language pair, with the following structure:
    {"en": ..., target_lang: ..., "article_id": ..., "doc_idx": ... }

Target languages: 
- Korean (ko)
- Spanish (es, replace the "sp" in the files)
- Russian (ru)
- Vietnamese (vi)
- Chinese simplified (zh, instead of "zhs")
- Chinese traditional (zht)
Skip "ht" language (Haitian Creole).
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken
import shutil

# Initialize tokenizer for English (using cl100k_base encoding, same as GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Language mappings
LANG_MAPPING = {
    "ko": "ko",
    "es": "sp",  # Spanish files use "sp"
    "ru": "ru",
    "vi": "vie",  # Vietnamese files use "vie"
    "zh": "zhs",  # Chinese simplified files use "zhs"
    "zht": "zht"
}

TARGET_LANGUAGES = ["ko", "es", "ru", "vi", "zh", "zht"]

MAX_TOKENS = 2000
MIN_TOKENS = 1000


@dataclass
class IndexNode:
    """Represents a node in the index hierarchy."""
    title: str
    level: int  # Indentation level (0 = top level)
    anchor: Optional[str] = None  # URL anchor if available
    children: List['IndexNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


def parse_index_file(index_path: Path) -> List[IndexNode]:
    """
    Parse an index file (markdown-like format with indentation).
    Returns a list of top-level IndexNode objects.
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    root_nodes = []
    stack = []  # Stack to track parent nodes at each level
    
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        
        # Count leading spaces to determine level (2 spaces = 1 level)
        leading_spaces = len(line) - len(line.lstrip(' '))
        level = leading_spaces // 2
        
        # Extract title and anchor from markdown link format: * [Title](url#anchor)
        match = re.match(r'\s*\*\s+\[([^\]]+)\]\([^)]+#([^)]+)\)', line)
        if not match:
            # Try without anchor
            match = re.match(r'\s*\*\s+\[([^\]]+)\]\([^)]+\)', line)
            if match:
                title = match.group(1)
                anchor = None
            else:
                # Plain text line
                title = line.strip().lstrip('*').strip()
                anchor = None
        else:
            title = match.group(1)
            anchor = match.group(2)
        
        node = IndexNode(title=title, level=level, anchor=anchor)
        
        # Update stack to only include nodes at levels less than current
        while stack and stack[-1].level >= level:
            stack.pop()
        
        if stack:
            stack[-1].children.append(node)
        else:
            root_nodes.append(node)
        
        stack.append(node)
    
    return root_nodes


def get_section_count(node: IndexNode) -> Dict[int, int]:
    """
    Get the count of subsections at each hierarchy level for a node.
    Returns a dict mapping level to count.
    """
    counts = {}
    
    def traverse(n: IndexNode, depth: int):
        if depth not in counts:
            counts[depth] = 0
        counts[depth] += len(n.children)
        for child in n.children:
            traverse(child, depth + 1)
    
    traverse(node, 0)
    return counts


def validate_hierarchy_match(en_node: IndexNode, target_node: IndexNode) -> bool:
    """
    Validate that two nodes have the same hierarchy structure.
    Returns True if they match at all levels.
    This is a prevalidation step to ensure sections can be aligned.
    """
    # First check: same number of direct children
    if len(en_node.children) != len(target_node.children):
        return False
    
    # Recursively check all children
    for en_child, target_child in zip(en_node.children, target_node.children):
        if not validate_hierarchy_match(en_child, target_child):
            return False
    
    return True


def validate_hierarchy_at_level(en_node: IndexNode, target_node: IndexNode, max_depth: int = 1) -> bool:
    """
    Validate hierarchy up to a certain depth level.
    Useful for incremental validation when building chunks.
    """
    if max_depth == 0:
        return True
    
    if len(en_node.children) != len(target_node.children):
        return False
    
    if max_depth > 1:
        for en_child, target_child in zip(en_node.children, target_node.children):
            if not validate_hierarchy_at_level(en_child, target_child, max_depth - 1):
                return False
    
    return True


def clean_markdown_urls(text: str) -> str:
    """
    Remove markdown URL patterns [text](url) or [text](url "title") from text.
    This helps with token counting by removing URLs that shouldn't count.
    Keeps only the link text, removes the URL and optional title.
    """
    if not text:
        return text
    # Remove markdown links: [text](url) or [text](url "title") -> text
    # This regex handles:
    # - [text](url)
    # - [text](url "title")
    # - [text](url 'title')
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text


def contains_table_or_image(text: str) -> bool:
    """
    Check if text contains tables (2+ pipes | ... |) or images (!This is an Image:).
    Returns True if the section should be skipped.
    """
    if not text:
        return False
    
    # Check for images
    if "!This is an Image:" in text:
        return True
    
    # Check for tables: look for lines with 2 or more pipes
    lines = text.split('\n')
    for line in lines:
        # Count pipes in the line
        pipe_count = line.count('|')
        if pipe_count >= 2:
            return True
    
    return False


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken, after cleaning markdown URLs."""
    if not text or not isinstance(text, str):
        return 0
    # Clean markdown URLs before counting tokens
    cleaned_text = clean_markdown_urls(text)
    tokens = tokenizer.encode(cleaned_text)
    return len(tokens)


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation, extra spaces)."""
    text = text.lower()
    # Remove markdown links and formatting
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove markdown links
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def extract_section_from_markdown(markdown_text: str, section_title: str, anchor: Optional[str] = None) -> Optional[str]:
    """
    Extract a section from markdown text based on section title or anchor.
    Returns the section content including the header.
    """
    lines = markdown_text.split('\n')
    section_start = None
    section_end = None
    
    normalized_title = normalize_text(section_title)
    
    # Find section start by matching header
    for i, line in enumerate(lines):
        # Check for markdown headers (##, ###, ####, etc.)
        header_match = re.match(r'^(#+)\s+(.+)', line)
        if header_match:
            header_text = header_match.group(2).strip()
            normalized_header = normalize_text(header_text)
            
            # Try to match title (fuzzy match)
            # Check if normalized title matches normalized header (or vice versa)
            if (normalized_title in normalized_header or 
                normalized_header in normalized_title or
                normalized_title == normalized_header):
                section_start = i
                break
            
            # Also check for anchor in the line or nearby
            if anchor and anchor in line:
                section_start = i
                break
    
    if section_start is None:
        return None
    
    # Find section end (next header at same or higher level)
    start_level = len(re.match(r'^(#+)', lines[section_start]).group(1)) if re.match(r'^#+', lines[section_start]) else 0
    
    for i in range(section_start + 1, len(lines)):
        line = lines[i]
        header_match = re.match(r'^(#+)\s+', line)
        if header_match:
            current_level = len(header_match.group(1))
            if current_level <= start_level:
                section_end = i
                break
    
    if section_end is None:
        section_end = len(lines)
    
    return '\n'.join(lines[section_start:section_end])


def extract_sections_recursive(markdown_text: str, node: IndexNode) -> str:
    """
    Recursively extract sections from markdown based on index node hierarchy.
    Returns concatenated section text.
    """
    sections = []
    
    # Try to extract current section
    section_text = extract_section_from_markdown(markdown_text, node.title, node.anchor)
    if section_text:
        sections.append(section_text)
    
    # Recursively extract children
    for child in node.children:
        child_text = extract_sections_recursive(markdown_text, child)
        if child_text:
            sections.append(child_text)
    
    return '\n\n'.join(sections) if sections else ""


def find_main_sections(index_nodes: List[IndexNode]) -> List[IndexNode]:
    """
    Find main sections (top-level nodes that are not introductory).
    The first main section is considered introductory and should be skipped.
    Since index_nodes are already the root nodes (top-level), we can return them directly.
    """
    # The index_nodes list already contains the top-level nodes (root nodes)
    # These are the main sections
    return index_nodes


def build_chunk_from_sections(
    en_markdown: str,
    target_markdown: str,
    en_node: IndexNode,
    target_node: IndexNode,
    max_tokens: int = MAX_TOKENS
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Build a chunk from sections, validating hierarchy incrementally and respecting token limit.
    Returns (en_text, target_text, is_complete) where is_complete indicates if we reached max_tokens.
    
    Strategy:
    1. Start with main section
    2. Add subsections incrementally, validating hierarchy at each step
    3. Stop when token limit is reached or hierarchy doesn't match
    """
    en_text = ""
    target_text = ""
    
    # Start with the main section
    en_section = extract_section_from_markdown(en_markdown, en_node.title, en_node.anchor)
    target_section = extract_section_from_markdown(target_markdown, target_node.title, target_node.anchor)
    
    if not en_section or not target_section:
        return None, None, False
    
    en_text = en_section
    target_text = target_section
    
    # Check if main section exceeds token limit
    # We allow larger sections through - post-processing will split them by super headers
    # Only skip if it's extremely large (more than 20k tokens) to avoid memory issues
    en_token_count = count_tokens(en_text)
    if en_token_count > 20000:
        # Main section is extremely large - skip this chunk to avoid memory issues
        return None, None, False
    
    # Try to add subsections incrementally, validating at each step
    # We can add up to the point where hierarchy is consistent
    for en_child, target_child in zip(en_node.children, target_node.children):
        # Prevalidate: check if this subsection has matching hierarchy
        if not validate_hierarchy_match(en_child, target_child):
            # Hierarchy doesn't match - stop adding subsections
            break
        
        # Extract this subsection and its children
        en_child_text = extract_sections_recursive(en_markdown, en_child)
        target_child_text = extract_sections_recursive(target_markdown, target_child)
        
        if not en_child_text or not target_child_text:
            # Couldn't extract - skip this subsection
            continue
        
        # Skip subsections that contain tables or images (in either language)
        if contains_table_or_image(en_child_text) or contains_table_or_image(target_child_text):
            # Skip this subsection - contains tables or images
            continue
        
        # Check if adding this would exceed token limit
        potential_en_text = en_text + '\n\n' + en_child_text
        potential_token_count = count_tokens(potential_en_text)
        
        if potential_token_count > max_tokens:
            # Would exceed limit - stop here
            return en_text, target_text, True
        
        # Safe to add - update texts
        en_text = potential_en_text
        target_text = target_text + '\n\n' + target_child_text
    
    return en_text, target_text, False


def process_article_pair(
    article_id: str,
    target_lang: str,
    base_dir: Path
) -> List[Dict[str, Any]]:
    """
    Process a single article pair (English + target language).
    Returns a list of parallel corpus entries.
    """
    # Map language codes
    file_lang_code = LANG_MAPPING.get(target_lang, target_lang)
    
    # Load index files
    index_dir = base_dir / "index" / article_id
    en_index_path = index_dir / f"{article_id}_en_index.json"
    target_index_path = index_dir / f"{article_id}_{file_lang_code}_index.json"
    
    if not en_index_path.exists() or not target_index_path.exists():
        print(f"Warning: Index files not found for {article_id} ({target_lang})")
        return []
    
    # Load article files
    article_dir = base_dir / "article"
    en_article_path = article_dir / f"{article_id}en.md"
    target_article_path = article_dir / f"{article_id}{file_lang_code}.md"
    
    if not en_article_path.exists() or not target_article_path.exists():
        print(f"Warning: Article files not found for {article_id} ({target_lang})")
        return []
    
    # Parse indexes
    en_index_nodes = parse_index_file(en_index_path)
    target_index_nodes = parse_index_file(target_index_path)
    
    # Load markdown articles
    with open(en_article_path, 'r', encoding='utf-8') as f:
        en_markdown = f.read()
    
    with open(target_article_path, 'r', encoding='utf-8') as f:
        target_markdown = f.read()
    
    # Find main sections
    en_main_sections = find_main_sections(en_index_nodes)
    target_main_sections = find_main_sections(target_index_nodes)
    
    # Skip first main section (introductory), start from second
    if len(en_main_sections) < 2 or len(target_main_sections) < 2:
        print(f"Warning: Not enough main sections for {article_id} ({target_lang})")
        return []
    
    # Align main sections (assuming they correspond by index)
    # This is a simplification - in practice, you might need better alignment
    corpus_entries = []
    doc_idx = 0
    
    # Process from second main section onwards
    print(f"    Found {len(en_main_sections)} EN main sections, {len(target_main_sections)} target main sections")
    print(f"    Processing {min(len(en_main_sections), len(target_main_sections)) - 1} main sections (skipping first)")
    
    sections_processed = 0
    sections_skipped_hierarchy = 0
    sections_skipped_too_large = 0
    sections_skipped_no_extract = 0
    
    for i in range(1, min(len(en_main_sections), len(target_main_sections))):
        en_node = en_main_sections[i]
        target_node = target_main_sections[i]
        
        # Validate that main sections match
        if not validate_hierarchy_match(en_node, target_node):
            print(f"    Warning: Hierarchy mismatch for main section {i} ({en_node.title[:50]}...) in {article_id} ({target_lang})")
            sections_skipped_hierarchy += 1
            continue
        
        # Build chunk
        en_text, target_text, is_complete = build_chunk_from_sections(
            en_markdown, target_markdown, en_node, target_node, MAX_TOKENS
        )
        
        if not en_text or not target_text:
            if not en_text and not target_text:
                # Check if it was too large
                en_section = extract_section_from_markdown(en_markdown, en_node.title, en_node.anchor)
                if en_section and count_tokens(en_section) > MAX_TOKENS:
                    sections_skipped_too_large += 1
                    print(f"    Section {i} too large ({count_tokens(en_section)} tokens), skipping")
                else:
                    sections_skipped_no_extract += 1
                    print(f"    Could not extract section {i}")
            continue
        
        # Clean markdown URLs from the text before saving
        en_text_cleaned = clean_markdown_urls(en_text)
        target_text_cleaned = clean_markdown_urls(target_text)
        
        corpus_entries.append({
            "en": en_text_cleaned,
            target_lang: target_text_cleaned,
            "article_id": article_id,
            "doc_idx": doc_idx
        })
        doc_idx += 1
        sections_processed += 1
    
    print(f"    Built {sections_processed} chunks (skipped: {sections_skipped_hierarchy} hierarchy, {sections_skipped_too_large} too large, {sections_skipped_no_extract} no extract)")
    return corpus_entries


def normalize_linebreaks(text: str) -> str:
    """
    Reduce 2+ consecutive linebreaks to exactly 1 (\n).
    This ensures consistent line spacing and helps with alignment.
    """
    if not text:
        return text
    # Replace 2+ newlines with exactly 1 newline
    text = re.sub(r'\n{2,}', '\n', text)
    return text


def get_header_type(line: str) -> Optional[Tuple[str, Optional[int]]]:
    """
    Detect header type in a line. Returns (header_type, level) or None.
    Types:
    - ('markdown', level): starts with #, ##, ###, ####, etc. (level = number of #)
    - ('bold', None): starts with ** (bold text as header, may end with ** or .)
    - ('bold_italic', None): starts with _** (bold italic text as header, may end with _** or .)
    """
    stripped = line.strip()
    
    # Markdown headers: #, ##, ###, ####, etc.
    if stripped.startswith('#'):
        # Count the number of # characters
        level = len(stripped) - len(stripped.lstrip('#'))
        return ('markdown', level)
    
    # Bold italic headers: _**text**_ or _**text**_. (with optional period)
    if stripped.startswith('_**'):
        # Check if it ends with _** or _**.
        if stripped.endswith('_**') or stripped.endswith('_**.'):
            return ('bold_italic', None)
    
    # Bold headers: **text** or **text**. (with optional period)
    if stripped.startswith('**'):
        # Check if it ends with ** or **.
        if stripped.endswith('**') or stripped.endswith('**.'):
            return ('bold', None)
    
    return None


def is_header_line(line: str) -> bool:
    """Check if a line is any type of header."""
    return get_header_type(line) is not None


def headers_compatible(en_line: str, target_line: str) -> bool:
    """
    Check if two lines have compatible headers (same type AND same level for markdown).
    Returns True if both are headers of the same type and level, or both are not headers.
    """
    en_header = get_header_type(en_line)
    target_header = get_header_type(target_line)
    
    # Both must be headers and of the same type
    if en_header is None or target_header is None:
        return en_header == target_header  # Both None (not headers) or one is header, one isn't
    
    # Both are headers - check type and level
    en_type, en_level = en_header
    target_type, target_level = target_header
    
    # Types must match
    if en_type != target_type:
        return False
    
    # For markdown headers, levels must also match
    if en_type == 'markdown':
        return en_level == target_level
    
    # For bold and bold_italic, type match is sufficient
    return True


def find_last_compatible_point(en_lines: List[str], target_lines: List[str]) -> Optional[int]:
    """
    Find the last line index where headers are compatible between English and target.
    Goes line by line checking header compatibility STRICTLY.
    Stops at the FIRST mismatch and returns the last compatible point before it.
    Returns the index of the last compatible line, or None if no compatibility found.
    """
    min_len = min(len(en_lines), len(target_lines))
    if min_len == 0:
        return None
    
    last_compatible = None
    
    # Go line by line and check header compatibility STRICTLY
    for i in range(min_len):
        en_line = en_lines[i]
        target_line = target_lines[i]
        
        en_is_header = is_header_line(en_line)
        target_is_header = is_header_line(target_line)
        
        # STRICT CHECK: If either side has a header, they MUST be compatible
        if en_is_header or target_is_header:
            # At least one side has a header - must check compatibility
            if en_is_header and target_is_header:
                # Both are headers - must be same type AND level (for markdown)
                if headers_compatible(en_line, target_line):
                    last_compatible = i
                else:
                    # Header mismatch - STOP IMMEDIATELY
                    # Return the last compatible point BEFORE this mismatch
                    return last_compatible
            else:
                # One side has header, other doesn't - this is a MISMATCH
                # Return the last compatible point BEFORE this mismatch
                return last_compatible
        else:
            # Neither is a header - both are content lines, so compatible
            last_compatible = i
    
    # If we get here, all lines are compatible
    # Also check if line counts match - if not, we need to truncate
    if len(en_lines) != len(target_lines):
        return last_compatible
    
    return last_compatible if last_compatible is not None else (min_len - 1 if min_len > 0 else None)


def truncate_before_header(en_lines: List[str], target_lines: List[str], last_compatible: int) -> Tuple[List[str], List[str]]:
    """
    Truncate at the last compatible point, but if that line is a header, truncate before it.
    This ensures chunks don't end with headers (which is unnatural) and headers can start the next chunk.
    Returns (en_lines, target_lines) truncated appropriately.
    """
    if last_compatible < 0:
        return [], []
    
    # We want to include the last_compatible line (it's compatible)
    # But if it's a header, we should truncate before it so the header can start the next chunk
    truncate_idx = last_compatible + 1  # Include the last compatible line
    
    # Check if the last compatible line is a header
    if last_compatible < len(en_lines) and last_compatible < len(target_lines):
        en_line = en_lines[last_compatible]
        target_line = target_lines[last_compatible]
        
        # If either is a header, truncate before it (don't include the header)
        if is_header_line(en_line) or is_header_line(target_line):
            if last_compatible > 0:
                truncate_idx = last_compatible  # Don't include the header
            else:
                # Can't truncate before first line if it's a header - return empty
                return [], []
    
    # Truncate at the determined point (includes last_compatible if it's not a header)
    return en_lines[:truncate_idx], target_lines[:truncate_idx]


def find_next_compatible_super_header(en_lines: List[str], target_lines: List[str], start_idx: int) -> Optional[int]:
    """
    Find the next compatible markdown super header (#, ##, ###, etc.) starting from start_idx.
    Returns the line index of the compatible header, or None if not found.
    """
    min_len = min(len(en_lines), len(target_lines))
    
    for i in range(start_idx, min_len):
        en_line = en_lines[i]
        target_line = target_lines[i]
        
        # Check if both are markdown headers (super headers only - # format)
        en_header = get_header_type(en_line)
        target_header = get_header_type(target_line)
        
        if en_header and target_header:
            en_type, _ = en_header
            target_type, _ = target_header
            if en_type == 'markdown' and target_type == 'markdown':
                # Both are markdown headers - check if they're compatible
                if headers_compatible(en_line, target_line):
                    return i
    
    return None


def split_chunk_by_super_headers(en_text: str, target_text: str) -> List[Tuple[str, str]]:
    """
    Split a chunk into multiple chunks based on compatible markdown super headers (# format).
    After splitting, each chunk is validated for alignment.
    Returns a list of (en_chunk, target_chunk) tuples.
    """
    chunks = []
    
    # Normalize line breaks and filter images
    en_text = normalize_linebreaks(en_text)
    target_text = normalize_linebreaks(target_text)
    
    # Filter image lines
    en_lines = [line for line in en_text.split('\n') if "!This is an Image:" not in line]
    target_lines = [line for line in target_text.split('\n') if "!This is an Image:" not in line]
    
    if not en_lines or not target_lines:
        return []
    
    current_start = 0
    min_len = min(len(en_lines), len(target_lines))
    
    while current_start < min_len:
        # Find the next compatible super header
        next_header_idx = find_next_compatible_super_header(en_lines, target_lines, current_start)
        
        if next_header_idx is None:
            # No more compatible super headers - take remaining content as last chunk
            if current_start < min_len:
                en_chunk = '\n'.join(en_lines[current_start:])
                target_chunk = '\n'.join(target_lines[current_start:])
                # ALWAYS validate alignment and header compatibility
                en_chunk_lines = en_chunk.split('\n')
                target_chunk_lines = target_chunk.split('\n')
                last_compat = find_last_compatible_point(en_chunk_lines, target_chunk_lines)
                if last_compat is not None and last_compat >= 0:
                    # Truncate before header if needed
                    en_aligned_lines, target_aligned_lines = truncate_before_header(en_chunk_lines, target_chunk_lines, last_compat)
                    if en_aligned_lines and target_aligned_lines:
                        en_aligned = '\n'.join(en_aligned_lines)
                        target_aligned = '\n'.join(target_aligned_lines)
                        if count_tokens(en_aligned) >= MIN_TOKENS:
                            chunks.append((en_aligned, target_aligned))
            break
        
        # If we found a header and it's not at the start, create a chunk up to (but not including) the header
        # But first check alignment of content before the header
        if next_header_idx > current_start:
            # Check if content before header is aligned
            pre_header_en = en_lines[current_start:next_header_idx]
            pre_header_target = target_lines[current_start:next_header_idx]
            
            # ALWAYS check header compatibility, even if line counts match
            last_compat = find_last_compatible_point(pre_header_en, pre_header_target)
            if last_compat is not None and last_compat >= 0:
                # Truncate before header if needed
                en_aligned_lines, target_aligned_lines = truncate_before_header(pre_header_en, pre_header_target, last_compat)
                if en_aligned_lines and target_aligned_lines:
                    en_aligned = '\n'.join(en_aligned_lines)
                    target_aligned = '\n'.join(target_aligned_lines)
                    if count_tokens(en_aligned) >= MIN_TOKENS:
                        chunks.append((en_aligned, target_aligned))
        
        # Now find where this header section ends (next compatible super header or end)
        section_start = next_header_idx
        next_section_header = find_next_compatible_super_header(en_lines, target_lines, section_start + 1)
        
        if next_section_header is None:
            # This is the last section - take everything from this header to the end
            en_chunk = '\n'.join(en_lines[section_start:])
            target_chunk = '\n'.join(target_lines[section_start:])
            # ALWAYS validate alignment and header compatibility
            en_chunk_lines = en_chunk.split('\n')
            target_chunk_lines = target_chunk.split('\n')
            last_compat = find_last_compatible_point(en_chunk_lines, target_chunk_lines)
            if last_compat is not None and last_compat >= 0:
                # Truncate before header if needed
                en_aligned_lines, target_aligned_lines = truncate_before_header(en_chunk_lines, target_chunk_lines, last_compat)
                if en_aligned_lines and target_aligned_lines:
                    en_aligned = '\n'.join(en_aligned_lines)
                    target_aligned = '\n'.join(target_aligned_lines)
                    if count_tokens(en_aligned) >= MIN_TOKENS:
                        chunks.append((en_aligned, target_aligned))
            break
        else:
            # Create chunk from this header to the next header
            en_chunk = '\n'.join(en_lines[section_start:next_section_header])
            target_chunk = '\n'.join(target_lines[section_start:next_section_header])
            # ALWAYS validate alignment and header compatibility
            en_chunk_lines = en_chunk.split('\n')
            target_chunk_lines = target_chunk.split('\n')
            last_compat = find_last_compatible_point(en_chunk_lines, target_chunk_lines)
            if last_compat is not None and last_compat >= 0:
                # Truncate before header if needed
                en_aligned_lines, target_aligned_lines = truncate_before_header(en_chunk_lines, target_chunk_lines, last_compat)
                if en_aligned_lines and target_aligned_lines:
                    en_aligned = '\n'.join(en_aligned_lines)
                    target_aligned = '\n'.join(target_aligned_lines)
                    if count_tokens(en_aligned) >= MIN_TOKENS:
                        chunks.append((en_aligned, target_aligned))
            current_start = next_section_header
    
    return chunks


def postprocess_chunk(en_text: str, target_text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Post-process a chunk to ensure better alignment:
    1. Normalize line breaks
    2. Filter out image lines (if both sides have "!This is an Image:" at same line)
    3. Always verify header compatibility line by line
    4. If compatibility breaks, truncate at last compatible point
    
    Returns (en_text, target_text, is_valid) where is_valid indicates if chunk should be kept.
    """
    # Step 1: Normalize line breaks
    en_text = normalize_linebreaks(en_text)
    target_text = normalize_linebreaks(target_text)
    
    # Step 2: Split into lines
    en_lines = en_text.split('\n')
    target_lines = target_text.split('\n')
    
    # Step 3: Filter out ANY lines containing image markers (aggressive filtering)
    # Remove lines with image markers from both sides, regardless of alignment
    filtered_en_lines = []
    filtered_target_lines = []
    
    # Filter English lines - remove any line containing image marker
    for en_line in en_lines:
        if "!This is an Image:" not in en_line:
            filtered_en_lines.append(en_line)
    
    # Filter target lines - remove any line containing image marker
    for target_line in target_lines:
        if "!This is an Image:" not in target_line:
            filtered_target_lines.append(target_line)
    
    # Reconstruct text from filtered lines
    en_text = '\n'.join(filtered_en_lines)
    target_text = '\n'.join(filtered_target_lines)
    
    # Step 4: Check header compatibility line by line
    en_lines = en_text.split('\n')
    target_lines = target_text.split('\n')
    last_compatible = find_last_compatible_point(en_lines, target_lines)
    
    if last_compatible is None:
        # No compatible point found - this means there was a mismatch from the start
        # We cannot keep this chunk - discard it
        return None, None, False
    
    # Check if we need to truncate (if compatibility broke before the end)
    min_len = min(len(en_lines), len(target_lines))
    needs_truncation = (last_compatible < min_len - 1) or (len(en_lines) != len(target_lines))
    
    if needs_truncation:
        # Truncate both sides at the last compatible point
        # But avoid ending at a header - truncate before it if needed
        en_truncated_lines, target_truncated_lines = truncate_before_header(en_lines, target_lines, last_compatible)
        
        if not en_truncated_lines or not target_truncated_lines:
            # Truncation resulted in empty - cannot keep this chunk
            # If truncation was needed, it means there was a header mismatch
            return None, None, False
        
        en_truncated = '\n'.join(en_truncated_lines)
        target_truncated = '\n'.join(target_truncated_lines)
        
        # Check if truncated chunk still meets minimum token requirement
        en_token_count = count_tokens(en_truncated)
        if en_token_count >= MIN_TOKENS:
            return en_truncated, target_truncated, True
        else:
            # Too short after truncation - cannot keep (there was a mismatch)
            return None, None, False
    else:
        # Full compatibility - verify one more time that all lines are compatible
        # This is a final check to ensure no mismatches
        for i in range(min(len(en_lines), len(target_lines))):
            en_line = en_lines[i]
            target_line = target_lines[i]
            en_is_header = is_header_line(en_line)
            target_is_header = is_header_line(target_line)
            
            if en_is_header or target_is_header:
                if not headers_compatible(en_line, target_line):
                    # Found a mismatch - should not happen if last_compatible was correct
                    # But if it does, truncate at this point
                    if i > 0:
                        en_truncated_lines, target_truncated_lines = truncate_before_header(en_lines, target_lines, i - 1)
                        if en_truncated_lines and target_truncated_lines:
                            en_truncated = '\n'.join(en_truncated_lines)
                            target_truncated = '\n'.join(target_truncated_lines)
                            if count_tokens(en_truncated) >= MIN_TOKENS:
                                return en_truncated, target_truncated, True
                    return None, None, False
        
        # All lines are compatible - keep as is
        return en_text, target_text, True


def create_inspection_output(entries: List[Dict[str, Any]], target_lang: str, base_dir: Path):
    """
    Create inspection output files for manual validation.
    Creates separate markdown files for each chunk that can be viewed side-by-side.
    """
    inspection_dir = base_dir / "to_inspect" / f"en_{target_lang}"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    # clean previous inspection files subdirs (chunk_*)
    for subdir in inspection_dir.glob("chunk_*"):
        shutil.rmtree(subdir)
    
    for entry in entries:
        chunk_id = f"chunk_{entry['doc_idx']:05d}"
        chunk_dir = inspection_dir / chunk_id
        chunk_dir.mkdir(exist_ok=True)
        
        # Write English side
        en_file = chunk_dir / "en.md"
        with open(en_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {entry['doc_idx']} - English\n\n")
            f.write(f"**Article ID:** {entry['article_id']}\n\n")
            f.write("---\n\n")
            f.write(entry["en"])
        
        # Write target language side
        target_file = chunk_dir / f"{target_lang}.md"
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {entry['doc_idx']} - {target_lang.upper()}\n\n")
            f.write(f"**Article ID:** {entry['article_id']}\n\n")
            f.write("---\n\n")
            f.write(entry[target_lang])
        
        # Also create a combined file for easy comparison
        combined_file = chunk_dir / "combined.md"
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {entry['doc_idx']} - Comparison\n\n")
            f.write(f"**Article ID:** {entry['article_id']}\n\n")
            f.write("---\n\n")
            f.write("## English\n\n")
            f.write(entry["en"])
            f.write("\n\n---\n\n")
            f.write(f"## {target_lang.upper()}\n\n")
            f.write(entry[target_lang])
    
    print(f"  Created inspection files in {inspection_dir}")


def main():
    """Main function to process all articles and build parallel corpus."""
    base_dir = Path(__file__).parent
    
    # Create output directory
    output_dir = base_dir / "parallel_corpus"
    output_dir.mkdir(exist_ok=True)
    
    # Find all article IDs
    index_dir = base_dir / "index"
    article_ids = [d.name for d in index_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    
    print(f"Found {len(article_ids)} articles")
    
    # Initialize summary statistics
    summary_stats = []
    
    # Process each language pair
    for target_lang in TARGET_LANGUAGES:
        print(f"\nProcessing language pair: en-{target_lang}")
        output_file = output_dir / f"en_{target_lang}.jsonl"
        
        all_entries = []
        
        for article_id in sorted(article_ids):
            print(f"  Processing {article_id}...")
            entries = process_article_pair(article_id, target_lang, base_dir)
            all_entries.extend(entries)
            print(f"    Added {len(entries)} chunks")
        
        # Post-process chunks for alignment and split by super headers
        print(f"    Post-processing {len(all_entries)} chunks...")
        postprocessed_entries = []
        current_doc_idx = 0
        
        chunks_split = 0
        chunks_no_split = 0
        chunks_discarded_postprocess = 0
        
        for entry in all_entries:
            en_text = entry["en"]
            target_text = entry[target_lang]
            
            # First try to split by super headers (this can create multiple chunks)
            split_chunks = split_chunk_by_super_headers(en_text, target_text)
            
            if split_chunks:
                chunks_split += 1
                # We got multiple chunks from splitting
                for en_chunk, target_chunk in split_chunks:
                    # Apply additional post-processing to each split chunk
                    en_processed, target_processed, is_valid = postprocess_chunk(en_chunk, target_chunk)
                    
                    if is_valid and en_processed and target_processed:
                        new_entry = {
                            "en": en_processed,
                            target_lang: target_processed,
                            "article_id": entry["article_id"],
                            "doc_idx": current_doc_idx
                        }
                        postprocessed_entries.append(new_entry)
                        current_doc_idx += 1
                    else:
                        chunks_discarded_postprocess += 1
            else:
                chunks_no_split += 1
                # No split possible, try regular post-processing
                en_processed, target_processed, is_valid = postprocess_chunk(en_text, target_text)
                
                if is_valid and en_processed and target_processed:
                    entry["en"] = en_processed
                    entry[target_lang] = target_processed
                    entry["doc_idx"] = current_doc_idx
                    postprocessed_entries.append(entry)
                    current_doc_idx += 1
                else:
                    chunks_discarded_postprocess += 1
                    print(f"    Post-processing: discarded chunk {entry.get('doc_idx', '?')} (alignment issue)")
        
        print(f"    After post-processing: {len(postprocessed_entries)} chunks (split: {chunks_split}, no split: {chunks_no_split}, discarded: {chunks_discarded_postprocess})")
        
        # Filter out chunks with less than MIN_TOKENS tokens (after post-processing)
        filtered_entries = []
        chunks_filtered_too_short = 0
        for entry in postprocessed_entries:
            en_text = entry["en"]
            en_token_count = count_tokens(en_text)
            if en_token_count >= MIN_TOKENS:
                filtered_entries.append(entry)
            else:
                chunks_filtered_too_short += 1
                print(f"    Filtered out chunk {entry.get('doc_idx', '?')} with {en_token_count} tokens (< {MIN_TOKENS})")
        
        print(f"    After filtering: {len(filtered_entries)} chunks (filtered out {chunks_filtered_too_short} too short)")
        
        # Write to jsonl file
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in filtered_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(filtered_entries)} total chunks (filtered from {len(all_entries)}) to {output_file}")
        
        # Calculate statistics
        en_total_tokens = 0
        target_total_tokens = 0
        en_max_tokens = 0
        en_min_tokens = float('inf')
        
        for entry in filtered_entries:
            en_tokens = count_tokens(entry["en"])
            target_tokens = count_tokens(entry[target_lang])
            en_total_tokens += en_tokens
            target_total_tokens += target_tokens
            en_max_tokens = max(en_max_tokens, en_tokens)
            en_min_tokens = min(en_min_tokens, en_tokens)
        
        num_chunks = len(filtered_entries)
        en_avg_tokens = en_total_tokens / num_chunks if num_chunks > 0 else 0
        target_avg_tokens = target_total_tokens / num_chunks if num_chunks > 0 else 0
        
        # Warn if chunks are unexpectedly large
        if en_max_tokens > MAX_TOKENS * 1.5:  # Allow 50% tolerance
            print(f"    Warning: Some chunks exceed expected size (max: {en_max_tokens} tokens, expected max: {MAX_TOKENS})")
        
        # Store statistics for final summary
        summary_stats.append({
            'lang_pair': f'en-{target_lang}',
            'num_docs': num_chunks,  # num_docs is actually number of chunks
            'en_avg_tokens': en_avg_tokens,
            'en_total_tokens': en_total_tokens,
            'target_avg_tokens': target_avg_tokens,
            'target_total_tokens': target_total_tokens
        })
        
        # Create inspection output
        create_inspection_output(filtered_entries, target_lang, base_dir)
    
    # Print final summary
    print("\n" + "="*80)
    print("PARALLEL CORPUS SUMMARY")
    print("="*80)
    print(f"{'Language Pair':<20} {'Documents':<12} {'EN Avg Tokens':<18} {'EN Total Tokens':<18} {'Target Avg Tokens':<20} {'Target Total Tokens':<20}")
    print("-"*80)
    
    for stats in summary_stats:
        print(f"{stats['lang_pair']:<20} {stats['num_docs']:<12} {stats['en_avg_tokens']:<18.1f} {stats['en_total_tokens']:<18} {stats['target_avg_tokens']:<20.1f} {stats['target_total_tokens']:<20}")
    
    # Grand totals
    total_docs = sum(s['num_docs'] for s in summary_stats)
    total_en_tokens = sum(s['en_total_tokens'] for s in summary_stats)
    total_target_tokens = sum(s['target_total_tokens'] for s in summary_stats)
    
    print("-"*80)
    print(f"{'TOTAL':<20} {total_docs:<12} {'':<18} {total_en_tokens:<18} {'':<20} {total_target_tokens:<20}")
    print("="*80)


if __name__ == "__main__":
    main()

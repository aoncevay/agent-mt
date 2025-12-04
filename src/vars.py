model_name2bedrock_id = {
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "qwen3-235b": "qwen.qwen3-235b-a22b-2507-v1:0",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "claude-sonnet-3-7": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "gemma-3-12b-it": "google.gemma-3-12b-it"
    #"gemma-3-27b": "huggingface-vlm-gemma-3-27b-instruct",
    #"claude-sonnet-4-5": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    #"claude-opus-4-5": "anthropic.claude-opus-4-5-20251101-v1:0",
}

language_id2name = {
    "en": "English",
    "zht": "Traditional Chinese",
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "ko": "Korean",
    "vi": "Vietnamese"
}

# Workflow name to module function mapping
WORKFLOW_REGISTRY = {
    "single_agent": "single_agent",
    "single_agent_term": "single_agent_term",
    "dual_agent": "dual_agent",
    "triple_agent": "triple_agent",
}
model_name2bedrock_id = {
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "qwen3-235b": "qwen.qwen3-235b-a22b-2507-v1:0",

    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
}

model_name2openai_id = {
    "gpt-4-1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4-1": "gpt-4.1-2025-04-14",
    "gpt-4-1-nano": "gpt-4.1-nano-2025-04-14",
    "o4-mini": "o4-mini-2025-04-16",
}

# Mapping of model names to Bedrock Application Inference Profile ARNs
# These ARNs require model_provider to be specified when using ChatBedrock
model_name2bedrock_arn = {
    "claude-sonnet-4": "arn:aws:bedrock:us-east-1:145023110438:application-inference-profile/69rh5jj9ixo6",
    "claude-sonnet-4-5": "arn:aws:bedrock:us-east-1:145023110438:application-inference-profile/5bczxc9bbzmo",
}

# Mapping of model names to their providers (required when using ARNs)
model_name2provider = {
    "claude-sonnet-4": "anthropic",
    "claude-sonnet-4-5": "anthropic",
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

# Prompt profiles by dataset/domain
PROMPT_PROFILES = {
    "financial": {
        "domain_description": "part of a financial document.",
        "domain_guidance": (
            "Ensure the translation is formal, precise, and consistent with "
            "standard financial and corporate communication. Use correct "
            "financial terminology and maintain a neutral, objective tone."
        ),
    },
    "tax_legal": {
        "domain_description": "discussing tax topics.",
        "domain_guidance": (
            "Ensure the translation is accurate, terminologically correct, "
            "and clearly conveys the underlying legal and fiscal concepts "
            "while maintaining a neutral, objective tone."
        ),
    },
}

DATASET_TO_PROMPT_PROFILE = {
    "wmt25": "financial",
    "dolfin": "financial",
    "irs": "tax_legal",
}
model_name2bedrock_id = {
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "qwen3-235b": "qwen.qwen3-235b-a22b-2507-v1:0",

    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",

    #"claude-sonnet-3-7": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    #"claude-sonnet-4-5": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    #"claude-opus-4-5": "anthropic.claude-opus-4-5-20251101-v1:0",
}
# CLAUDE MODELS ARE NOT WORKING ON EXTERNAL AWS MACHINES
# WE SHOULD ADD OPENAI MODELS (GPT4.1, o4-mini)

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
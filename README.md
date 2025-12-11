# Cost-Performance Trade-Offs of Multi-Agent Machine Translation

A framework for running multi-agent translation workflows using AWS Bedrock (and OpenAI)models.

## Quick Start

### 1. Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Create `config.env` file:**
```bash
# AWS Bedrock credentials
AWS_REGION=us-east-2
AWS_BEARER_TOKEN_BEDROCK=your_access_key

# OpenAI credentials (if using OpenAI models)
OPENAI_API_KEY=your_openai_key

# Optional: Bedrock configuration (they are set in the code)
BEDROCK_MAX_RETRIES=3
BEDROCK_INITIAL_BACKOFF=2.0
```

### 2. Run an Experiment

**Single workflow (10 samples):**
```bash
python src/run.py --dataset wmt25 --workflow zero_shot --model qwen3-32b --max_samples 10
```

**With terminology (WMT25 only):**
```bash
python src/run.py --dataset wmt25 --workflow zero_shot --model qwen3-32b --use_terminology
```

**Resume interrupted experiment:**
```bash
python src/run.py --dataset wmt25 --workflow zero_shot --model qwen3-32b --resume
```

**Run all workflows (testing with few samples):**
```bash
cd run
./run_samples_per_model.sh
```

**Run all workflows (full dataset, parallel):**
```bash
cd run
./run_all_per_model.sh
```

## Project Structure

```
agent-mt/
├── data/
│   └── raw/
│       ├── wmt25-terminology-track2/    # WMT25 dataset files
│       └── dolfin/                      # DOLFIN dataset files
├── outputs/                             # All experiment outputs
│   └── {dataset}/
│       └── {lang_pair}/
│           └── {workflow_acronym}{.term}/
│               └── {model}/
│                   ├── sample_00000_agent_0.txt
│                   ├── sample_00000_agent_1.txt
│                   └── report.json
├── src/
│   ├── run.py                           # Main experiment runner
│   ├── workflows/                       # Workflow implementations
│   ├── templates/                       # Jinja2 prompt templates
│   ├── data_loaders.py                  # Dataset loaders
│   ├── translation.py                   # Base translation functions with Bedrock
│   ├── evaluation.py                    # BLEU, chrF++, term accuracy
│   ├── vars.py                          # Model and language mappings
│   ├── utils.py                         # Additional functions
│   └── workflow_acronyms.py             # Output directory naming
└── run/                                 # Batch execution scripts
    ├── run_samples_per_model.sh         # Test with few samples
    └── run_all_per_model.sh             # Full dataset, parallel
```

## Output Structure

Outputs are organized as:
```
outputs/{dataset}/{lang_pair}/{workflow_acronym}{.term}/{model}/
```

**Example:**
```
outputs/wmt25/en-zht/ADT.term/qwen3-32b/
├── sample_00000_agent_0.txt
├── sample_00000_agent_1.txt
└── report.json
```

- **Dataset**: `wmt25` or `dolfin` (`irs` in progress)
- **Language pair**: `wmt25`: `en-zht`, `en-zht`. `dolfin`: `en-es`, `en-it`, `en-fr`, `en-de`
- **Workflow acronym**: `ADT`, `MAATS_multi`, `SbS_chat`, etc.
- **Terminology suffix**: `.term` if `--use_terminology` was used
- **Model**: `qwen3-32b`, `qwen3-235b`, etc.

## Report Structure

Each experiment generates a `report.json` with:

```json
{
  "dataset": "wmt25",
  "workflow": "zero_shot",
  "model": "qwen3-32b",
  "total_samples": 100,
  "successful_samples": 98,
  "failed_samples": 2,
  "samples": [
    {
      "sample_idx": 0,
      "sample_id": "2015_001",
      "source_lang": "en",
      "target_lang": "zht",
      "lang_pair": "en-zht",
      "chrf_scores": [45.23],
      "bleu_scores": [32.15],
      "term_success_rates": [0.85],
      "tokens_input": 1234,
      "tokens_output": 567,
      "latency": 2.34
    }
  ],
  "summary": {
    "total_tokens_input": 123400,
    "total_tokens_output": 56700,
    "total_latency_seconds": 234.0,
    "avg_latency_seconds": 2.34,
    "avg_chrf_score": 45.23,
    "avg_bleu_score": 32.15,
    "avg_term_success_rate": 0.85
  }
}
```

## Adding New Models

Edit `src/vars.py` and add your model to `model_name2bedrock_id`:

```python
model_name2bedrock_id = {
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "qwen3-235b": "qwen.qwen3-235b-a22b-2507-v1:0",
    "your-model-name": "bedrock.model.id:version",  # Add here
}
```

**Finding Bedrock model IDs:**
- AWS Console → Bedrock → Foundation models
- Format: `provider.model-name:version` (e.g., `qwen.qwen3-32b-v1:0`)

**For OpenAI models:**
- A new module or code might need to be implemented for this
- Ensure `OPENAI_API_KEY` and other keys are set in `config.env`

## Adding New Workflows

### 1. Create Workflow Module

Create `src/workflows/your_workflow.py`:

```python
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage

try:
    from ..translation import create_bedrock_llm
    from ..utils import load_template, render_translation_prompt
    from ..vars import language_id2name
except ImportError:
    from translation import create_bedrock_llm
    from utils import load_template, render_translation_prompt
    from vars import language_id2name

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
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run your workflow.
    
    Returns:
        Dictionary with:
        - 'outputs': List of agent outputs [output1, output2, ...]
        - 'tokens_input': Total input tokens
        - 'tokens_output': Total output tokens
        - 'latency': Total workflow time in seconds
    """
    import time
    
    llm = create_bedrock_llm(model_id, region)
    total_tokens_input = 0
    total_tokens_output = 0
    start_time = time.time()
    outputs = []
    
    # Your workflow logic here
    # Example:
    prompt = render_translation_prompt(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        language_id2name=language_id2name,
        use_terminology=use_terminology,
        terminology=terminology
    )
    
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    
    output = response.content.strip()
    outputs.append(output)
    
    # Count tokens
    tokens_input = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('prompt_tokens', 0)
    tokens_output = getattr(response, 'response_metadata', {}).get('token_usage', {}).get('completion_tokens', 0)
    
    if tokens_input == 0:
        tokens_input = len(prompt) // 4  # Fallback estimation
    if tokens_output == 0:
        tokens_output = len(output) // 4
    
    total_tokens_input += tokens_input
    total_tokens_output += tokens_output
    
    latency = time.time() - start_time
    
    return {
        "outputs": outputs,
        "tokens_input": total_tokens_input,
        "tokens_output": total_tokens_output,
        "latency": latency
    }
```

### 2. Register Workflow

Add to `src/workflows/__init__.py`:

```python
WORKFLOW_REGISTRY = {
    # ... existing workflows ...
    "your_workflow": "your_workflow",  # Add this line
}
```

### 3. Add Workflow Acronym (Optional)

Acronym is used to save the output files. Add to `src/workflow_acronyms.py`:

```python
WORKFLOW_ACRONYMS = {
    # ... existing acronyms ...
    "your_workflow": "YW",  # Add this line
}
```

### 4. Create Templates (if needed)

Create Jinja2 templates in `src/templates/`:

```jinja
{# src/templates/your_template.jinja #}
You are a translator from {{ source_lang_name }} to {{ target_lang_name }}.

Translate the following text:
{{ source_text }}
```

Load in your workflow:
```python
template = load_template("your_template.jinja")
prompt = template.render(
    source_text=source_text,
    source_lang_name=get_language_name(source_lang, language_id2name),
    target_lang_name=get_language_name(target_lang, language_id2name)
)
```

## Available Workflows

- `zero_shot` - Zero-shot translation
- `MaMT_translate_postedit_proofread` - Translate + Postedit + Proofread ([WMT'25 (EMNLP)](https://aclanthology.org/2025.wmt-1.53/))
- `IRB_refine` - Initial translation + Refinement ([WMT'25 (EMNLP)](https://aclanthology.org/2025.wmt-1.51/))
- `MAATS_multi_agents` - Multi-agent MQM evaluation ([arxiv](https://arxiv.org/pdf/2505.14848))
- `SbS_chat_step_by_step` - Translating step-by-step, chat-based ([WMT'24 (EMNLP)](https://aclanthology.org/2024.wmt-1.123/))
- `DeLTA_multi_agents` - Document-level translation with memory ([ICLR'25](https://openreview.net/forum?id=hoYFLRNbhc))
- `ADT_multi_agents` - Discourse-level translation ([ARR Feb'25 under-review](https://openreview.net/forum?id=JguwmASD3n))

Complementary workflows (for ablation if needed):

- `MaMT_translate_postedit` - Translate + Postedit (without Proofread)
- `SbS_step_by_step` - Step-by-step (standalone prompts, not based on the paper)
- `MAATS_single_agent` - Single-agent MQM evaluation (baseline for MAATS)

## Available Models

- `qwen3-32b` - Qwen 3 32B
- `qwen3-235b` - Qwen 3 235B
- `gpt-oss-20b` - GPT-OSS 20B
- `gpt-oss-120b` - GPT-OSS 120B

See `src/vars.py` for the complete list and to add new models.

## Datasets

### WMT25 Terminology Track2
- Location: `data/raw/wmt25-terminology-track2/`
- Files: `full_data_2015.jsonl` through `full_data_2024.jsonl`
- Supports terminology dictionaries
- Language pairs: `en-zht` (odd years), `zht-en` (even years)

### DOLFIN
- Location: `data/raw/dolfin/`
- Files: `dolfin_test_en_es.jsonl`, `dolfin_test_en_de.jsonl`, etc.
- Language pairs: `en-es`, `en-de`, `en-fr`, `en-it`
- No terminology support

## Command-Line Arguments

```bash
python src/run.py \
  --dataset {wmt25|dolfin} \
  --workflow {workflow_name} \
  --model {model_name} \
  [--target_languages {lang1 lang2 ...}] \
  [--max_samples {N}] \
  [--use_terminology] \
  [--resume] \
  [--output_dir {custom_path}]
```

- `--dataset`: Dataset name (`wmt25` or `dolfin`)
- `--workflow`: Workflow name (see Available Workflows)
- `--model`: Model name (see Available Models)
- `--target_languages`: Filter specific language pairs (optional)
- `--max_samples`: Limit number of samples per language pair (optional)
- `--use_terminology`: Enable terminology (WMT25 only)
- `--resume`: Resume interrupted experiment
- `--output_dir`: Custom output directory (optional)

## Evaluation Metrics

Each experiment computes:
- **BLEU-4**: Standard BLEU score (uses `zh` tokenizer for Chinese, `ko-mecab` for Korean)
- **chrF++**: Character and word n-gram F-score (char_order=6, word_order=2)
- **Term Success Rate**: Terminology accuracy (WMT25 with `--use_terminology` only)

Metrics are computed per sample and aggregated in the report summary.

## Configuration

### Temperature Setting

Temperature settings are workflow-specific, following the original papers:

**Per-Workflow Settings:**
- **MaMT** (`MaMT_translate_postedit`, `MaMT_translate_postedit_proofread`):
  - Translation: `temperature=0.0` (reproducibility)
  - Postedit: `temperature=1.0` (exploration, encourages broader error detection)
  - Proofread: `temperature=0.0` (reproducibility)
- **IRB**: `temperature=0.0` (no sampling, reproducibility)
- **MAATS**: `temperature=0.0` (paper uses 0-0.3, we use 0.0 for consistency)
- **SbS, DeLTA, ADT**: `temperature=0.0` (no paper specification, default for reproducibility)

**Rationale:**
- **IRB-WMT25**: "temperature was set to 0 (no sampling)" for reproducibility
- **MaMT**: Uses `temperature=1` for postedit (exploration), `temperature=0` for translation/proofread (reproducibility)
- **MAATS**: "temperature low (between 0 and 0.3)" for deterministic improvements
- **SbS, DeLTA, ADT**: No temperature specification in papers, default to 0.0 for reproducibility

**To change temperature:**
Edit the workflow file (e.g., `src/workflows/MaMT_translate_postedit.py`) or `src/translation.py` → `create_bedrock_llm()` function:
```python
llm = create_bedrock_llm(model_id, region, temperature=0.0)  # Explicit temperature
```

**Note:** Some Bedrock models may not accept exactly 0.0. If you encounter errors, try 0.1 as a fallback (near-deterministic).

## Troubleshooting

**"Module not found" errors:**
- Ensure you're running from the project root
- Check that `src/` is in Python path

**AWS/Bedrock errors:**
- Verify `config.env` has correct credentials
- Check AWS region matches your Bedrock endpoint
- Ensure model is available in your AWS region
- If temperature=0.0 causes errors, try temperature=0.1

**Out of memory:**
- Reduce `MAX_PARALLEL` in `run_all_per_model.sh`
- Process fewer samples at a time

**Resume not working:**
- Ensure you're using the same arguments (dataset, workflow, model)
- Check that `report.json` exists in the output directory

## Contributing

When adding new workflows or features:
1. Follow existing code patterns
2. Add error handling and token counting
3. Update this README
4. Test with `run_samples_per_model.sh` first or create a specific script


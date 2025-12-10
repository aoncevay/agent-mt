# Translation Experiment Runner

This document describes the refactored translation experiment system.

## Structure

```
src/
├── data_loaders.py          # Dataset loaders (WMT25, DOLFIN, etc.)
├── workflows/               # Agentic workflow modules
│   ├── __init__.py
│   ├── single_agent.py      # Single agent without terminology
│   ├── single_agent_term.py # Single agent with terminology
│   ├── dual_agent.py        # (Future) Two-agent workflow
│   └── triple_agent.py      # (Future) Three-agent workflow
├── run.py                   # Main runner script
├── evaluation.py            # chrF++ evaluation
├── translation.py           # Core translation utilities
├── utils.py                 # Utility functions
└── vars.py                  # Configuration (models, workflows, languages)

outputs/                     # Output directory
└── {dataset}.{workflow}.{model}/
    ├── sample_00000_agent_0.txt
    ├── sample_00001_agent_0.txt
    └── report.json
```

## Usage

### Basic Usage

```bash
# WMT25 dataset with single agent (no terminology) - both directions
python src/run.py --dataset wmt25 --workflow single_agent --model qwen3-235b

# WMT25 dataset - only en->zht direction
python src/run.py --dataset wmt25 --workflow single_agent_term --model qwen3-235b --target_languages zht

# WMT25 dataset - only zht->en direction
python src/run.py --dataset wmt25 --workflow zero_shot --model qwen3-235b --target_languages en

# DOLFIN dataset - all language pairs
python src/run.py --dataset dolfin --workflow zero_shot --model qwen3-235b

# DOLFIN dataset - specific language pairs
python src/run.py --dataset dolfin --workflow zero_shot --model qwen3-235b --target_languages es de

# Limit number of samples per language pair
python src/run.py --dataset wmt25 --workflow zero_shot --model qwen3-235b --max_samples 5
```

### Arguments

- `--dataset`: Dataset name
  - `wmt25`: WMT25 terminology track2 (has terminology, interleaving directions)
  - `dolfin`: DOLFIN dataset (processes all available language pairs or filtered by --target_languages)

- `--target_languages`: Target languages to process (optional)
  - For `wmt25`: `zht` (en->zht) or `en` (zht->en). If not specified, processes both directions.
  - For `dolfin`: `es`, `de`, `fr`, `it` (e.g., `--target_languages es de`). If not specified, processes all available pairs.

- `--workflow`: Workflow name
  - `single_agent`: Single agent without terminology
  - `single_agent_term`: Single agent with terminology
  - `dual_agent`: (Future) Two-agent workflow
  - `triple_agent`: (Future) Three-agent workflow

- `--model`: Model name (from vars.py)
  - `qwen3-235b`
  - `gpt-oss-120b`
  - etc.

- `--max_samples`: Maximum number of samples to process (optional)

- `--output_dir`: Custom output directory (optional, defaults to `outputs/{dataset}.{workflow}.{model}`)

## Output Structure

Each experiment creates:
- Individual translation files: `sample_{idx:05d}_agent_{agent_id}.txt`
- Summary report: `report.json` with:
  - chrF++ scores per sample
  - Token counts (input/output)
  - Latency measurements
  - Summary statistics

Both WMT25 and DOLFIN use language pair subdirectories:
```
outputs/
├── wmt25.single_agent.qwen3-235b/
│   ├── en-zht/
│   │   ├── sample_00000_agent_0.txt
│   │   └── report.json
│   └── zht-en/
│       ├── sample_00000_agent_0.txt
│       └── report.json
└── dolfin.single_agent.qwen3-235b/
    ├── en_es/
    │   ├── sample_00000_agent_0.txt
    │   └── report.json
    ├── en_de/
    │   ├── sample_00000_agent_0.txt
    │   └── report.json
    └── ...
```

**Note**: 
- Output directories use dot (`.`) separators for easy parsing: `{dataset}.{workflow}.{model}`
- Language pairs are stored in subdirectories: `{source_lang}-{target_lang}` for WMT25, `{source_lang}_{target_lang}` for DOLFIN
- Each language pair has its own report.json

## Adding New Datasets

1. Create a new loader class in `data_loaders.py` extending `BaseDataLoader`
2. Add it to the `get_data_loader()` factory function
3. Implement required methods: `load_samples()`, `get_translation_direction()`, `extract_texts()`

## Adding New Workflows

1. Create a new file in `workflows/` directory
2. Implement `run_workflow()` function with signature:
   ```python
   def run_workflow(
       source_text: str,
       source_lang: str,
       target_lang: str,
       model_id: str,
       terminology: Optional[Dict[str, list]] = None,
       region: Optional[str] = None,
       max_retries: int = 3,
       initial_backoff: float = 2.0
   ) -> Dict[str, Any]:
       # Returns: {"outputs": [...], "tokens_input": int, "tokens_output": int, "latency": float}
   ```
3. Add workflow name to `WORKFLOW_REGISTRY` in `vars.py`

## Token Counting

Token counts are estimated if not available from the API response:
- Input tokens: `len(prompt) // 4` (rough approximation)
- Output tokens: `len(translation) // 4` (rough approximation)

For accurate counts, the API response should include token usage metadata.


# Metrics Evaluation System

This folder contains the implementation for document-level evaluation metrics, based on the WMT25-Term term-consistency approach.

## Overview

The metrics system computes:
- **TermBasedMetric (TBM)**: Terminology consistency metrics for WMT25-Term dataset
- **COMET**: Reference-based quality metric for all datasets (DOLFIN and WMT25-Term)

## Approach

We adopt the term-consistency approach from WMT25-Term track2:
1. **Split**: Documents are split into aligned segments (using LaBSE embeddings)
2. **Align**: Terms are aligned between source and target segments
3. **Evaluate**: Compute TBM metrics (first/frequent/predefined) and COMET scores per segment

## Key Differences from Original

1. **CDAO instead of OpenAI API**: Uses our CDAO library for LLM calls (no API key needed)
2. **Report.json format**: Reads from our experiment output structure
3. **Incremental saving**: Results are saved incrementally, not after all processing
4. **Output format**: JSON per dataset/lang_pair with `{"workflow+model": {...}}` structure

## Files

- `utils.py`: Utility functions for reading outputs and reports
- `docpreprocessor.py`: Document splitting and alignment (adapted from WMT25-Term)
- `termbasedmetric.py`: Terminology-based metrics (adapted from WMT25-Term, uses CDAO)
- `comet_evaluator.py`: COMET score calculation per segment
- `evaluate_experiments.py`: Main evaluation script
- `fewshot/`: Few-shot examples for term alignment (en-fr, en-it added for DOLFIN)

## Usage

```bash
# Evaluate all experiments for a dataset/lang_pair
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht

# Evaluate specific experiments
python metrics/evaluate_experiments.py --outputs-dir outputs/wmt25/en-zht

# Evaluate only COMET (DOLFIN)
python metrics/evaluate_experiments.py --dataset dolfin --lang-pair en-fr --comet-only
```

## Output Format

Results are saved as `{dataset}_{lang_pair}_metrics.json`:

```json
{
  "IRB+gpt-4-1": {
    "first": {"micro": 0.75, "macro": 0.82},
    "frequent": {"micro": 0.80, "macro": 0.85},
    "predefined": {"micro": 0.65, "macro": 0.70},
    "comet": {
      "avg_comet": 0.82,
      "min_comet": 0.65,
      "max_comet": 0.95
    }
  }
}
```

For DOLFIN (COMET only):
```json
{
  "IRB+gpt-4-1": {
    "comet": {
      "avg_comet": 0.82,
      "min_comet": 0.65,
      "max_comet": 0.95
    }
  }
}
```


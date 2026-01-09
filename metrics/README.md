# Metrics Evaluation System

This folder contains the implementation for document-level evaluation metrics, based on the WMT25-Term term-consistency approach.

## Quick Setup

**All models must be available locally** (no internet connections allowed in restricted environments).

### 1. Configure Model Paths

Copy the example configuration and adjust for your environment:

```bash
cp metrics/env.example metrics/.env
# Edit metrics/.env with your paths
```

**Minimal configuration** (if models are in `~/HF_models`):
```bash
# metrics/.env
HF_MODELS_DIR=~/HF_models
```

**SageMaker configuration**:
```bash
# metrics/.env
HF_MODELS_DIR=~/user-default-efs/HF_models
STANZA_RESOURCES_DIR=~/user-default-efs/stanza_resources
```

### 2. Required Models

Download these models locally before running evaluation:

| Model | Purpose | Default Location |
|-------|---------|------------------|
| LaBSE | Paragraph alignment | `{HF_MODELS_DIR}/LaBSE` |
| MetricX-24 | Translation quality | `{HF_MODELS_DIR}/metricx-24-hybrid-large-v2p6-bfloat16` |
| MT5 tokenizer | MetricX tokenization | `{HF_MODELS_DIR}/mt5-base` |
| Awesome-align | Word alignment (TBM) | `{HF_MODELS_DIR}/awesome-align-with-co` |
| Stanza | Lemmatization (TBM) | `{STANZA_RESOURCES_DIR}/` |

### 3. Verify Configuration

```bash
python metrics/config.py
```

This prints all resolved paths and highlights missing models.

---

## Overview

The metrics system computes:
- **TermBasedMetric (TBM)**: Terminology consistency metrics for WMT25-Term dataset
- **MetricX**: Reference-based quality metric using segment-level alignment

## Approach

We adopt the term-consistency approach from WMT25-Term track2:
1. **Split**: Documents are split into aligned segments (using LaBSE embeddings)
2. **Align**: Segments aligned using VecAlign (default) or simple DP
3. **Evaluate**: Compute TBM metrics (first/frequent/predefined) and MetricX scores per segment

## Key Differences from Original

1. **CDAO instead of OpenAI API**: Uses our CDAO library for LLM calls (no API key needed)
2. **Report.json format**: Reads from our experiment output structure
3. **Incremental saving**: Results are saved incrementally, not after all processing
4. **Output format**: JSON per dataset/lang_pair with `{"workflow+model": {...}}` structure
5. **Offline-first**: All models loaded locally, no internet connections
6. **Centralized config**: Model paths configured via `.env` file

## Files

- `config.py`: Centralized configuration for model/tool paths
- `env.example`: Example configuration file (copy to `.env`)
- `utils.py`: Utility functions for reading outputs and reports
- `docpreprocessor.py`: Document splitting and alignment (using LaBSE + VecAlign/DP)
- `dp_alignment.py`: Alignment algorithms (VecAlign wrapper + simple DP fallback)
- `termbasedmetric.py`: Terminology-based metrics (adapted from WMT25-Term, uses CDAO)
- `metricx_evaluator.py`: MetricX score calculation per segment
- `evaluate_experiments.py`: Main evaluation script
- `fewshot/`: Few-shot examples for term alignment (en-fr, en-it added for DOLFIN)

## Usage

```bash
# First, verify your configuration
python metrics/config.py

# Evaluate all experiments for a dataset/lang_pair
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht

# Evaluate specific experiments
python metrics/evaluate_experiments.py --outputs-dir outputs/wmt25/en-zht

# Split GPU memory usage (useful for limited GPU memory)
# Step 1: Run alignment only (uses LaBSE, saves to tmp files)
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht --align-only

# Step 2: Run MetricX only (loads from tmp files)
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht --metricx-only

# Step 3: Run TBM only
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht --tbm-only

# Use different alignment methods
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht --aligner vecalign  # default
python metrics/evaluate_experiments.py --dataset wmt25 --lang-pair en-zht --aligner dp  # fallback
```

## Output Format

Results are saved in `metrics/results/{dataset}/{lang_pair}/{workflow}/{model}/`:

```json
// metricx.json - Translation quality scores
{
  "avg_metricx": 0.82,
  "min_metricx": 0.65,
  "max_metricx": 0.95,
  "scores": [0.82, 0.78, ...],
  "under_translated_count": 0,
  "over_translated_count": 0,
  "skipped_count": 0
}

// tbm.json - Terminology consistency (WMT25-Term only)
{
  "first": {"micro": 0.75, "macro": 0.82},
  "frequent": {"micro": 0.80, "macro": 0.85},
  "predefined": {"micro": 0.65, "macro": 0.70}
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_MODELS_DIR` | Base directory for HuggingFace models | `~/HF_models` |
| `STANZA_RESOURCES_DIR` | Stanza resources directory | `~/stanza_resources` |
| `LABSE_MODEL_PATH` | LaBSE model path (override) | `{HF_MODELS_DIR}/LaBSE` |
| `METRICX_MODEL_PATH` | MetricX model path (override) | `{HF_MODELS_DIR}/metricx-24-...` |
| `VECALIGN_PATH` | VecAlign repository path | `other_repos/vecalign` |

See `env.example` for full documentation.


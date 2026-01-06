# spaCy Setup Guide

The codebase now uses **spaCy** instead of Stanza for lemmatization (word normalization).

## Installation

### 1. Install spaCy

```bash
pip install spacy>=3.4.0
```

### 2. Download Language Models

On a machine with internet access, download the required models:

```bash
# English
python -m spacy download en_core_web_sm

# German
python -m spacy download de_core_news_sm

# Spanish
python -m spacy download es_core_news_sm

# French
python -m spacy download fr_core_news_sm

# Italian
python -m spacy download it_core_news_sm

# Chinese
python -m spacy download zh_core_web_sm
```

Or download all at once:
```bash
python -m spacy download en_core_web_sm de_core_news_sm es_core_news_sm fr_core_news_sm it_core_news_sm zh_core_web_sm
```

## Offline Use (SageMaker)

If you're in an environment without internet (like SageMaker):

1. **Download models on a machine with internet:**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download de_core_news_sm
   # ... etc
   ```

2. **Find where spaCy installed the models:**
   ```python
   import spacy
   print(spacy.util.find_model('en_core_web_sm'))
   ```
   Typically: `~/.local/lib/python3.X/site-packages/` or in your conda/env directory.

3. **Transfer the model directories to SageMaker:**
   ```bash
   # Copy the entire model directory (e.g., en_core_web_sm-3.7.1/)
   scp -r ~/.local/lib/python3.X/site-packages/en_core_web_sm* user@sagemaker:~/spacy_models/
   ```

4. **Use models from custom location:**
   ```python
   import spacy
   nlp = spacy.load('/path/to/en_core_web_sm')
   ```

   Or set `SPACY_DATA` environment variable:
   ```bash
   export SPACY_DATA=/path/to/spacy_models
   ```

## Fallback Behavior

If spaCy models are not available, the code will:
- Print a warning message
- Fall back to simple lowercase normalization
- Continue processing (no errors)

This means the code will still work, but term matching accuracy may be slightly reduced.

## Model Sizes

- Small models (`*_sm`): ~10-50 MB each
- Medium models (`*_md`): ~50-200 MB each (not needed for lemmatization)
- Large models (`*_lg`): ~500+ MB each (not needed for lemmatization)

For lemmatization, the small models (`*_sm`) are sufficient and recommended.


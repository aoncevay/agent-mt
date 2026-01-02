# Manual Setup Guide (No Docker)

This guide explains how to set up SEGALE manually without Docker, including LASER and COMET-DA models.

## Prerequisites

- Python 3.8+
- pip
- git
- Access to clone repositories (on a machine with internet)

## Step 1: Download COMET-DA Model (on local machine with internet)

```bash
# Create models directory
mkdir -p ~/Documents/Code/segale_models

# Download COMET-DA model
huggingface-cli download Unbabel/wmt22-comet-da \
    --local-dir ~/Documents/Code/segale_models/wmt22-comet-da
```

## Step 2: Clone and Setup LASER (on local machine with internet)

```bash
# Clone LASER repository
git clone https://github.com/facebookresearch/LASER ~/Documents/Code/LASER

# Install LASER dependencies first (workaround for sacremoses version conflict)
cd ~/Documents/Code/LASER
pip install sacremoses==0.1.1  # Install available version first

# Install LASER
# Note: This may fail on laser-encoders dependency if it's not in your pip index.
# That's OK - we can use LASER directly without the laser_encoders package.
pip install -e . || echo "Note: Installation may have warnings, but LASER repo is usable"

# Download LASER models for your languages
# Format: {language_code}_{script_code}
# Common codes:
#   - zho_Hant: Traditional Chinese (zht)
#   - zho_Hans: Simplified Chinese (zh)
#   - eng_Latn: English (en)
#   - spa_Latn: Spanish (es)
#   - deu_Latn: German (de)
#   - fra_Latn: French (fr)
#   - kor_Hang: Korean (ko)
#   - vie_Latn: Vietnamese (vi)
#   - rus_Cyrl: Russian (ru)
#
# For Traditional Chinese (zht), use:
bash ./nllb/download_models.sh zho_Hant

# You can download multiple languages:
# bash ./nllb/download_models.sh zho_Hant eng_Latn spa_Latn

# Install external tools
bash ./install_external_tools.sh
```

## Step 3: Transfer to Work Environment

```bash
# Transfer both COMET and LASER
rsync -avz ~/Documents/Code/segale_models/ user@work:~/Documents/Code/
rsync -avz ~/Documents/Code/LASER/ user@work:~/Documents/Code/LASER/

# Or using scp
scp -r ~/Documents/Code/segale_models/* user@work:~/Documents/Code/
scp -r ~/Documents/Code/LASER user@work:~/Documents/Code/
```

## Step 4: Install SEGALE Dependencies (on work environment)

```bash
# Navigate to SEGALE directory
cd other_repos/SEGALE

# Install SEGALE
pip install -e .

# Install additional dependencies
pip install spacy
python -m spacy download en  # Download English model (and others as needed)
```

**Note:** You don't need to install `laser_encoders` separately. If it's not available in your pip index, you can:

1. **Use LASER embed.sh directly** (recommended if you have LASER cloned):
   ```bash
   # Patch SEGALE to use LASER embed.sh directly
   python segale/patch_use_laser_directly.py --laser-dir ~/Documents/Code/LASER
   ```

2. **Use Alternative Embedding Model** (if you prefer):

If `laser_encoders` is not available in your pip index, you can use an alternative embedding model instead:

```bash
# When running test_segale.py, use --embedding-model flag
python segale/test_segale.py --output-dir outputs/... --embedding-model BAAI/bge-m3

# Or use multilingual-e5-large
python segale/test_segale.py --output-dir outputs/... --embedding-model intfloat/multilingual-e5-large
```

This will download the model from HuggingFace (if you have access) and use it instead of LASER embeddings. No `laser_encoders` installation needed!

## Step 5: Configure LASER Path

You have two options:

### Option A: Use Setup Script (Recommended)

```bash
# Run the setup script
python segale/setup_laser.py --laser-dir ~/Documents/Code/LASER
```

This will automatically patch `segale_align.py` to use your LASER path.

### Option B: Set Environment Variable

```bash
# Add to your ~/.bashrc or ~/.zshrc
export LASER=~/Documents/Code/LASER

# Or set it in the current session
export LASER=~/Documents/Code/LASER
```

Note: If using environment variable, you may need to modify `segale_align.py` to read it:
```python
# Change line 37 from:
LASER_DIR = "/opt/LASER"
# To:
LASER_DIR = os.environ.get("LASER", "/opt/LASER")
```

## Step 6: Configure COMET-DA Model

```bash
# Run the setup script
python segale/setup_local_segale.py --models-dir ~/Documents/Code
```

This will:
- Check that COMET-DA model is present
- Patch `segale_eval.py` to use local COMET-DA path
- Create environment script

## Step 7: Set Environment Variables

```bash
# Source the environment script
source segale/segale_local_env.sh

# Make sure LASER is set (if using Option A)
export LASER=~/Documents/Code/LASER
```

## Step 8: Test Installation

```bash
# Run a test
python segale/test_segale.py --output-dir outputs/wmt25/en-es/IRB/gpt-4-1 --max-samples 5
```

## Directory Structure

After setup, your `~/Documents/Code/` should look like:

```
~/Documents/Code/
├── segale_models/
│   └── wmt22-comet-da/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
└── LASER/
    ├── tasks/
    ├── nllb/
    ├── setup.py
    └── ...
```

## Troubleshooting

### "LASER not found" error

- Check that `LASER` environment variable is set: `echo $LASER`
- Verify LASER directory exists: `ls ~/Documents/Code/LASER`
- Check that `segale_align.py` uses the correct path

### "sacremoses==0.1.0 not found" error (when installing LASER)

This is a known issue when installing LASER - `laser-encoders` requires `sacremoses==0.1.0` but only `0.1.1` is available. Workaround:

```bash
# In the LASER directory
cd ~/Documents/Code/LASER

# Install available version first
pip install sacremoses==0.1.1

# Install LASER (will fail on laser-encoders, that's OK)
pip install -e . || true

# Install laser-encoders without dependency check
pip install laser_encoders==0.0.2 --no-deps

# Verify it works
python -c "from laser_encoders import LaserEncoderPipeline; print('OK')"
```

### "laser_encoders not found" error

If `laser_encoders` is not available in your pip index, you have two options:

**Option 1: Use LASER embed.sh Directly (Recommended if you have LASER cloned)**

Since you already cloned LASER, you can use it directly without the `laser_encoders` package:

```bash
# Patch SEGALE to use LASER embed.sh directly
python segale/patch_use_laser_directly.py --laser-dir ~/Documents/Code/LASER
```

This creates a wrapper that uses LASER's `embed.sh` script directly, so no `laser_encoders` package is needed.

**Option 2: Use Alternative Embedding Model**

Use a HuggingFace embedding model instead:

```bash
# Run with alternative embedding model
python segale/test_segale.py --output-dir outputs/... --embedding-model BAAI/bge-m3
```

Good alternative models:
- `BAAI/bge-m3` - Multilingual embedding model
- `intfloat/multilingual-e5-large` - Multilingual E5 model

### "spacy model not found" error

```bash
python -m spacy download en
python -m spacy download es  # For Spanish, etc.
```

### "COMET model not found" error

- Verify model path: `ls ~/Documents/Code/wmt22-comet-da`
- Run setup script again: `python segale/setup_local_segale.py --models-dir ~/Documents/Code`

### "segale-align not found" error

```bash
cd other_repos/SEGALE
pip install -e .
```

## Verification Checklist

- [ ] COMET-DA model downloaded and transferred
- [ ] LASER cloned and transferred
- [ ] SEGALE installed (`pip install -e .` in `other_repos/SEGALE`)
- [ ] `laser_encoders` installed
- [ ] spaCy models downloaded
- [ ] `LASER` environment variable set (or `segale_align.py` patched)
- [ ] Setup script run successfully
- [ ] Test script runs without errors

## Next Steps

Once everything is working:
1. Run full evaluation on your experiments
2. Review the COMET-DA scores
3. Integrate into the main codebase


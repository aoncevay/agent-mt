# Quick Start: SEGALE with Local COMET Models (Offline)

## Summary

**Model needed (reference-based evaluation only):**
- ✅ **COMET-DA** - `Unbabel/wmt22-comet-da` (~1.5GB) - Reference-based evaluation

**Total size: ~1.5GB**

## Quick Setup Steps

### 1. Download COMET Models (on local machine with internet)

```bash
# Create models directory
mkdir -p ~/Documents/Code/segale_models

# Download COMET-DA model (reference-based evaluation)
huggingface-cli download Unbabel/wmt22-comet-da \
    --local-dir ~/Documents/Code/segale_models/wmt22-comet-da
```

### 2. Transfer to Work Environment

```bash
# Using rsync (recommended)
rsync -avz ~/Documents/Code/segale_models/ user@work:~/Documents/Code/

# Or using scp
scp -r ~/Documents/Code/segale_models/* user@work:~/Documents/Code/
```

### 3. Configure SEGALE (on work environment)

```bash
# Run the setup script
python segale/setup_local_segale.py --models-dir ~/Documents/Code
```

This will:
- Check that COMET-DA model is present
- Create a backup of original `segale_eval.py`
- Patch it to use local COMET-DA model path
- Disable COMET-KIWI and MetricX evaluation (using reference-based COMET-DA only)
- Create an environment script

### 4. Run SEGALE

```bash
# Source the environment script
source segale/segale_local_env.sh

# Run the test
python segale/test_segale.py --output-dir outputs/wmt25/en-es/IRB/gpt-4-1 --max-samples 5
```

## What Gets Patched

The setup script modifies `other_repos/SEGALE/segale_eval.py` to:

1. **COMET-DA**: Replace `download_model("Unbabel/wmt22-comet-da")` with direct path
2. **COMET-KIWI**: Disabled (set to -1 scores) - not needed for reference-based evaluation
3. **MetricX**: Disabled (set to -1 scores) - using COMET-DA only

## Restore Original

If you need to restore the original SEGALE code:

```bash
python segale/setup_local_segale.py --models-dir ~/Documents/Code --restore
```

## LASER Setup (For Alignment Step)

**LASER is NOT from HuggingFace** - it's from Facebook Research GitHub.

### On Local Machine (with internet):

```bash
# Clone LASER
git clone https://github.com/facebookresearch/LASER ~/Documents/Code/LASER
cd ~/Documents/Code/LASER

# Install LASER
# Note: This may have warnings about laser-encoders, but that's OK.
# We'll use LASER embed.sh directly, so we don't need the laser_encoders package.
pip install -e .

# Download LASER models for your languages
# For Traditional Chinese (zht), use zho_Hant
bash ./nllb/download_models.sh zho_Hant
# You can download multiple: bash ./nllb/download_models.sh zho_Hant eng_Latn

# Install external tools
bash ./install_external_tools.sh
```

### Transfer to Work Environment:

```bash
# Transfer LASER
rsync -avz ~/Documents/Code/LASER/ user@work:~/Documents/Code/LASER/
# Or
scp -r ~/Documents/Code/LASER user@work:~/Documents/Code/
```

### On Work Environment:

```bash
# Install SEGALE dependencies
cd other_repos/SEGALE
pip install -e .
pip install laser_encoders==0.0.2

# Set LASER environment variable
export LASER=~/Documents/Code/LASER

# Patch SEGALE to use LASER embed.sh directly (no laser_encoders package needed)
python segale/patch_use_laser_directly.py --laser-dir ~/Documents/Code/LASER
```

**See `MANUAL_SETUP.md` for complete setup instructions.**

## Troubleshooting

**"Model not found" error:**
- Check model paths are correct
- Verify both COMET models are in the models directory
- Run `python segale/setup_local_segale.py --models-dir ~/Documents/Code` to verify

**"LASER not found" error:**
- If using Docker: Should work automatically
- If not: Set `LASER` environment variable or edit `segale_align.py`

**"Cannot connect to HuggingFace" error:**
- Make sure you ran the setup script
- Check that the patched `segale_eval.py` is being used
- Verify environment variables are set


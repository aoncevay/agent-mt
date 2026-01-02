# SEGALE Evaluation Setup

**Start here!** This README tells you which guide to follow.

## Quick Decision Tree

1. **Are you setting up for the first time?** → Go to **`MANUAL_SETUP.md`**
2. **Need a quick reference?** → Check **`QUICKSTART.md`**
3. **Having issues?** → Check troubleshooting in **`MANUAL_SETUP.md`**

## Documentation Files

### 📘 `MANUAL_SETUP.md` - **START HERE**
**Complete step-by-step setup guide (no Docker required)**

Follow this if:
- You're setting up SEGALE for the first time
- You don't have Docker access
- You need detailed instructions

**What it covers:**
- Downloading COMET-DA model
- Cloning and setting up LASER
- Installing SEGALE dependencies
- Configuring everything to work offline
- Troubleshooting common issues

### 📗 `QUICKSTART.md` - Quick Reference
**Condensed version for quick lookup**

Use this if:
- You've already set up before and need a reminder
- You want a quick checklist
- You know what you're doing

### 📙 Other Files (Reference Only)

- **`LASER_LANGUAGE_CODES.md`** - Language code mapping (e.g., `zht` → `zho_Hant`)
- **`DOCKER_SETUP.md`** - Docker setup (optional, if you have Docker access)

## Recommended Setup Flow

```bash
# 1. Follow MANUAL_SETUP.md step by step
# 2. When done, test with:
python segale/test_segale.py --output-dir outputs/... --max-samples 5
```

## Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `MANUAL_SETUP.md` | **Main guide** | First-time setup |
| `QUICKSTART.md` | Quick reference | Quick lookup |
| `LASER_LANGUAGE_CODES.md` | Language codes | When downloading LASER models |
| `DOCKER_SETUP.md` | Docker guide | If using Docker |
| `test_segale.py` | Test script | After setup |
| `setup_local_segale.py` | Configure COMET | During setup |
| `setup_laser.py` | Configure LASER path | During setup |
| `patch_use_laser_directly.py` | Use LASER directly | If laser_encoders unavailable |

## TL;DR

**Just follow `MANUAL_SETUP.md` from top to bottom. That's it!**

#!/usr/bin/env python3
"""
Patch segale_align.py to use LASER embed.sh directly instead of requiring laser_encoders.

This allows using LASER without installing the laser_encoders Python package.

Usage:
    python segale/patch_use_laser_directly.py --laser-dir ~/Documents/Code/LASER
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEGALE_DIR = PROJECT_ROOT / "other_repos" / "SEGALE"
SEGALE_ALIGN_FILE = SEGALE_DIR / "segale_align.py"


class LaserEmbedWrapper:
    """
    Wrapper to use LASER embed.sh script directly instead of laser_encoders package.
    """
    def __init__(self, laser_dir: Path):
        self.laser_dir = Path(laser_dir)
        self.embed_sh = self.laser_dir / "tasks" / "embed" / "embed.sh"
        if not self.embed_sh.exists():
            raise FileNotFoundError(f"LASER embed.sh not found: {self.embed_sh}")
    
    def encode_sentences(self, sentences):
        """
        Encode sentences using LASER embed.sh script.
        Returns embeddings as numpy arrays.
        """
        import numpy as np
        import os
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as input_file:
            input_path = input_file.name
            for sent in sentences:
                input_file.write(sent + '\n')
        
        output_path = input_path + '.emb'
        
        try:
            # Run LASER embed.sh
            cmd = f'"{self.embed_sh}" "{input_path}" "{output_path}"'
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            # Read embeddings
            with open(output_path, 'rb') as f:
                # LASER embeddings are stored as binary float32 arrays
                # Format: num_sentences * embedding_dim (1024 for laser2)
                emb_data = np.frombuffer(f.read(), dtype=np.float32)
                num_sentences = len(sentences)
                emb_dim = len(emb_data) // num_sentences
                embeddings = emb_data.reshape(num_sentences, emb_dim)
                
                # Return list of embeddings (one per sentence)
                return [emb for emb in embeddings]
        finally:
            # Clean up temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


def patch_segale_align(laser_dir: Path):
    """
    Patch segale_align.py to use LASER embed.sh directly.
    """
    laser_dir_str = str(laser_dir.resolve())
    backup_file = SEGALE_ALIGN_FILE.with_suffix('.py.backup_laser')
    
    # Create backup
    if not backup_file.exists():
        shutil.copy2(SEGALE_ALIGN_FILE, backup_file)
        print(f"✓ Created backup: {backup_file}")
    
    # Read original file
    with open(SEGALE_ALIGN_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add import for our wrapper at the top (after other imports)
    wrapper_code = f'''
# LASER embed.sh wrapper (added by patch_use_laser_directly.py)
import subprocess
import tempfile
import os
import numpy as np

class LaserEmbedWrapper:
    """Wrapper to use LASER embed.sh directly instead of laser_encoders package."""
    def __init__(self, laser_dir):
        self.laser_dir = Path(laser_dir)
        self.embed_sh = self.laser_dir / "tasks" / "embed" / "embed.sh"
        if not self.embed_sh.exists():
            raise FileNotFoundError(f"LASER embed.sh not found: {{self.embed_sh}}")
    
    def encode_sentences(self, sentences):
        """Encode sentences using LASER embed.sh script."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as input_file:
            input_path = input_file.name
            for sent in sentences:
                input_file.write(sent + '\\n')
        
        output_path = input_path + '.emb'
        
        try:
            # Run LASER embed.sh
            cmd = f'"{self.embed_sh}" "{input_path}" "{output_path}"'
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            # Read embeddings
            with open(output_path, 'rb') as f:
                emb_data = np.frombuffer(f.read(), dtype=np.float32)
                num_sentences = len(sentences)
                emb_dim = len(emb_data) // num_sentences
                embeddings = emb_data.reshape(num_sentences, emb_dim)
                return [emb for emb in embeddings]
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

'''
    
    # Find where to insert (after imports, before first function)
    import_end_pattern = "import datetime"
    if import_end_pattern in content:
        # Insert after the last import
        import_pos = content.rfind(import_end_pattern)
        next_line = content.find('\n', import_pos)
        if next_line != -1:
            content = content[:next_line+1] + wrapper_code + content[next_line+1:]
    
    # Replace the laser_encoders import and usage
    old_import = '''    if args.embedding_model is None:
        try:
            from laser_encoders import LaserEncoderPipeline
        except ImportError as e:
            raise ImportError(
                "laser_encoders is required when --embedding_model is not specified. Install it with `pip install laser_encoders==0.0.2`."
            ) from e
        tokenizer, model = (
            None,
            LaserEncoderPipeline(model_dir=LASER_DIR, laser="laser2"),
        )'''
    
    new_code = f'''    if args.embedding_model is None:
        # Use LASER embed.sh directly (no laser_encoders package needed)
        try:
            model = LaserEmbedWrapper(LASER_DIR)
            tokenizer = None
        except FileNotFoundError as e:
            raise ImportError(
                f"LASER embed.sh not found. Make sure LASER_DIR is set correctly. {{e}}"
            ) from e
        except Exception as e:
            raise ImportError(
                f"Failed to initialize LASER embed.sh wrapper: {{e}}"
            ) from e'''
    
    if old_import in content:
        content = content.replace(old_import, new_code)
        print(f"✓ Patched to use LASER embed.sh directly")
    else:
        print("⚠ Warning: Could not find exact pattern to replace")
        print("  You may need to manually edit segale_align.py")
        return False
    
    # Write patched file
    with open(SEGALE_ALIGN_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch segale_align.py to use LASER embed.sh directly (no laser_encoders needed)"
    )
    parser.add_argument(
        "--laser-dir",
        type=str,
        required=True,
        help="Path to LASER directory"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore original segale_align.py from backup"
    )
    
    args = parser.parse_args()
    
    backup_file = SEGALE_ALIGN_FILE.with_suffix('.py.backup_laser')
    
    if args.restore:
        if backup_file.exists():
            shutil.copy2(backup_file, SEGALE_ALIGN_FILE)
            print(f"✓ Restored original segale_align.py from backup")
            return 0
        else:
            print("✗ No backup file found")
            return 1
    
    laser_dir = Path(args.laser_dir)
    
    if not laser_dir.exists():
        print(f"✗ Error: LASER directory does not exist: {laser_dir}")
        return 1
    
    # Check for embed.sh
    embed_sh = laser_dir / "tasks" / "embed" / "embed.sh"
    if not embed_sh.exists():
        print(f"✗ Error: LASER embed.sh not found: {embed_sh}")
        print("  Make sure you've cloned the full LASER repository")
        return 1
    
    print(f"Patching segale_align.py to use LASER embed.sh directly...")
    print(f"  LASER directory: {laser_dir}")
    print(f"  embed.sh: {embed_sh}")
    
    if patch_segale_align(laser_dir):
        print("\n✓ Successfully patched!")
        print("\nNow you can use SEGALE without installing laser_encoders:")
        print("  python segale/test_segale.py --output-dir outputs/...")
        print("\nTo restore original:")
        print("  python segale/patch_use_laser_directly.py --laser-dir <dir> --restore")
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


#!/usr/bin/env python3
"""
Helper script to configure LASER path in segale_align.py.

This script patches segale_align.py to use a custom LASER directory path
instead of the default /opt/LASER.

Usage:
    python segale/setup_laser.py --laser-dir ~/Documents/Code/LASER
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEGALE_DIR = PROJECT_ROOT / "other_repos" / "SEGALE"
SEGALE_ALIGN_FILE = SEGALE_DIR / "segale_align.py"


def patch_laser_path(laser_dir: Path):
    """
    Patch segale_align.py to use the specified LASER directory.
    
    Args:
        laser_dir: Path to LASER directory
    """
    laser_dir_str = str(laser_dir.resolve())
    backup_file = SEGALE_ALIGN_FILE.with_suffix('.py.backup')
    
    # Create backup
    if not backup_file.exists():
        shutil.copy2(SEGALE_ALIGN_FILE, backup_file)
        print(f"✓ Created backup: {backup_file}")
    
    # Read original file
    with open(SEGALE_ALIGN_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace LASER_DIR definition
    # Find the line: LASER_DIR = "/opt/LASER"
    old_laser_dir = 'LASER_DIR = "/opt/LASER"'
    new_laser_dir = f'LASER_DIR = "{laser_dir_str}"'
    
    if old_laser_dir in content:
        content = content.replace(old_laser_dir, new_laser_dir)
        print(f"✓ Patched LASER_DIR to: {laser_dir_str}")
    else:
        # Try to find and replace if format is slightly different
        import re
        pattern = r'LASER_DIR\s*=\s*["\']/opt/LASER["\']'
        if re.search(pattern, content):
            content = re.sub(pattern, f'LASER_DIR = "{laser_dir_str}"', content)
            print(f"✓ Patched LASER_DIR to: {laser_dir_str}")
        else:
            print("⚠ Warning: Could not find LASER_DIR definition to patch")
            print("  You may need to manually edit segale_align.py line 37")
            return False
    
    # Write patched file
    with open(SEGALE_ALIGN_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Configure LASER path in segale_align.py"
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
    
    backup_file = SEGALE_ALIGN_FILE.with_suffix('.py.backup')
    
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
    
    # Check for LASER structure
    if not (laser_dir / "tasks").exists():
        print(f"⚠ Warning: {laser_dir} doesn't look like a LASER directory")
        print("  Expected to find 'tasks' subdirectory")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    print(f"Setting LASER_DIR to: {laser_dir}")
    
    if patch_laser_path(laser_dir):
        print("\n✓ LASER path configured successfully!")
        print("\nYou can also set the LASER environment variable instead:")
        print(f"  export LASER={laser_dir}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


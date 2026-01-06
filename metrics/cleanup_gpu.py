#!/usr/bin/env python3
"""
Utility script to clean up GPU memory.

This script helps release GPU memory that may be held by cached models
or processes from previous failed attempts.

Usage:
    python metrics/cleanup_gpu.py
    python metrics/cleanup_gpu.py --force  # Kill all Python processes using GPU
"""

import argparse
import subprocess
import sys


def clear_pytorch_cache():
    """Clear PyTorch CUDA cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ Cleared PyTorch CUDA cache")
            return True
    except ImportError:
        print("⚠ PyTorch not available")
    except Exception as e:
        print(f"⚠ Error clearing PyTorch cache: {e}")
    return False


def get_gpu_processes():
    """Get list of processes using GPU memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    pid = parts[0].strip()
                    name = parts[1].strip()
                    memory = parts[2].strip()
                    processes.append((pid, name, memory))
        return processes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def kill_gpu_processes(force=False):
    """Kill processes using GPU memory."""
    processes = get_gpu_processes()
    
    if not processes:
        print("✓ No processes found using GPU memory")
        return
    
    print(f"\nFound {len(processes)} process(es) using GPU:")
    for pid, name, memory in processes:
        print(f"  PID {pid}: {name} ({memory})")
    
    if not force:
        print("\n⚠ Use --force flag to kill these processes")
        return
    
    print("\nKilling processes...")
    killed = 0
    for pid, name, memory in processes:
        try:
            subprocess.run(['kill', '-9', pid], check=True)
            print(f"  ✓ Killed PID {pid} ({name})")
            killed += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to kill PID {pid}: {e}")
    
    print(f"\n✓ Killed {killed}/{len(processes)} processes")
    
    # Clear cache after killing processes
    clear_pytorch_cache()


def show_gpu_status():
    """Show current GPU memory status."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        print("\nGPU Status:")
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Could not get GPU status (nvidia-smi not available)")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up GPU memory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Kill all Python processes using GPU memory (use with caution!)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show GPU status only (don't clean)"
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_gpu_status()
        return
    
    print("Cleaning GPU memory...")
    print("="*60)
    
    # Clear PyTorch cache
    clear_pytorch_cache()
    
    # Show GPU status
    show_gpu_status()
    
    # Get and optionally kill processes
    if args.force:
        kill_gpu_processes(force=True)
    else:
        kill_gpu_processes(force=False)
    
    print("\n" + "="*60)
    print("Done!")
    print("\nIf memory is still full, try:")
    print("  1. Restart Python kernel/process")
    print("  2. Run with --force to kill GPU processes (use with caution)")
    print("  3. Check for zombie processes: nvidia-smi")


if __name__ == "__main__":
    main()


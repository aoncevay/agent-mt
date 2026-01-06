#!/bin/bash
# Aggressive GPU cleanup script
# Use this when normal cleanup doesn't work

echo "=== Aggressive GPU Cleanup ==="
echo ""

# 1. Show current GPU status
echo "1. Current GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# 2. Find processes using GPU device files
echo "2. Finding processes using GPU devices..."
GPU_PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)

if [ -z "$GPU_PIDS" ]; then
    echo "   No processes found via fuser"
else
    echo "   Found PIDs: $GPU_PIDS"
    for PID in $GPU_PIDS; do
        if [ -d "/proc/$PID" ]; then
            PROC_NAME=$(cat /proc/$PID/comm 2>/dev/null || echo "unknown")
            echo "   PID $PID: $PROC_NAME"
        fi
    done
fi
echo ""

# 3. Find zombie/stopped Python processes (especially evaluate_experiments.py)
echo "3. Finding zombie/stopped Python processes..."
ZOMBIE_PIDS=$(ps aux | grep -E "python.*evaluate_experiments|python.*metrics" | grep -v grep | awk '{print $2}' | sort -u)

if [ ! -z "$ZOMBIE_PIDS" ]; then
    echo "   Found Python processes: $ZOMBIE_PIDS"
    for PID in $ZOMBIE_PIDS; do
        if [ -d "/proc/$PID" ]; then
            PROC_NAME=$(cat /proc/$PID/comm 2>/dev/null || echo "unknown")
            PROC_STAT=$(cat /proc/$PID/stat 2>/dev/null | awk '{print $3}' || echo "unknown")
            echo "   PID $PID: $PROC_NAME (state: $PROC_STAT)"
        fi
    done
fi
echo ""

# 4. Kill processes using GPU
if [ ! -z "$GPU_PIDS" ]; then
    echo "4. Killing processes using GPU devices..."
    for PID in $GPU_PIDS; do
        if [ -d "/proc/$PID" ]; then
            kill -9 $PID 2>/dev/null && echo "   ✓ Killed PID $PID" || echo "   ✗ Failed to kill PID $PID"
        fi
    done
    sleep 2
fi

# 5. Kill zombie/stopped Python processes
if [ ! -z "$ZOMBIE_PIDS" ]; then
    echo "5. Killing zombie/stopped Python processes..."
    for PID in $ZOMBIE_PIDS; do
        if [ -d "/proc/$PID" ]; then
            # Try SIGTERM first (gentle)
            kill $PID 2>/dev/null && sleep 1
            # If still alive, force kill
            if [ -d "/proc/$PID" ]; then
                kill -9 $PID 2>/dev/null && echo "   ✓ Killed PID $PID" || echo "   ✗ Failed to kill PID $PID"
            else
                echo "   ✓ Killed PID $PID (SIGTERM)"
            fi
        fi
    done
    sleep 2
fi
echo ""

# 6. Clear PyTorch cache via Python
echo "6. Clearing PyTorch CUDA cache..."
python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('   ✓ Cache cleared')" 2>/dev/null || echo "   ⚠ PyTorch not available"
echo ""

# 7. Final GPU status
echo "7. Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# 6. If memory is still high, suggest reset
MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$MEMORY_USED" -gt 1000 ]; then
    echo "⚠ Memory still high: ${MEMORY_USED}MB"
    echo ""
    echo "If memory is still full, try:"
    echo "  1. Restart your Python kernel/process"
    echo "  2. Reset GPU (requires sudo): sudo nvidia-smi --gpu-reset -i 0"
    echo "  3. Check for zombie processes: ps aux | grep python"
fi

echo ""
echo "=== Done ==="


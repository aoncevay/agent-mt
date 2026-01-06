#!/bin/bash
# Quick script to kill all zombie/stopped evaluate_experiments.py processes

echo "=== Killing Zombie Python Processes ==="
echo ""

# Find all evaluate_experiments.py processes
PIDS=$(ps aux | grep "python.*evaluate_experiments" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No evaluate_experiments.py processes found"
    exit 0
fi

echo "Found processes:"
ps aux | grep "python.*evaluate_experiments" | grep -v grep
echo ""

# Ask for confirmation
read -p "Kill all these processes? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Kill processes
KILLED=0
for PID in $PIDS; do
    if [ -d "/proc/$PID" ]; then
        echo "Killing PID $PID..."
        kill -9 $PID 2>/dev/null && KILLED=$((KILLED + 1)) || echo "  Failed to kill $PID"
    fi
done

echo ""
echo "Killed $KILLED process(es)"
echo ""

# Clear GPU cache
echo "Clearing GPU cache..."
python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('Done')" 2>/dev/null || echo "PyTorch not available"

echo ""
echo "=== Done ==="
echo "Check GPU memory: nvidia-smi"


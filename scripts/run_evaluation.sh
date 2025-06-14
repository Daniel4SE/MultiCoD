#!/bin/bash
# Run Multi-CoD evaluation on BigCodeBench

set -e

echo "Starting Multi-CoD Pass@k Evaluation"
echo "===================================="

# Check environment
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    echo "Please run: export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Create directories
mkdir -p cache results logs

# Run evaluation
echo "Running evaluation..."
python -m src.main --config config.yaml 2>&1 | tee logs/evaluation_$(date +%Y%m%d_%H%M%S).log

# Find latest results directory
RESULTS_DIR=$(ls -td results/run_* | head -1)

# Run analysis
echo "Analyzing results..."
python scripts/analyze_results.py "$RESULTS_DIR"

echo "Evaluation complete!"
echo "Results available in: $RESULTS_DIR"
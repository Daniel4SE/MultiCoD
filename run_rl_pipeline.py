#!/usr/bin/env python3
"""
Run the complete RL training and evaluation pipeline on BigCodeBench v0.1.4
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def run_command(cmd, desc):
    """Run a command and handle errors"""
    print(f"\n🚀 {desc}")
    print(f"Running: {cmd}")
    print("-"*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Error: {desc} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {desc} completed in {elapsed:.1f} seconds")
    return True

def main():
    # Check if API key is set
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please run: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)
    
    print_banner("BigCodeBench v0.1.4 RL Pipeline")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("results_training").mkdir(exist_ok=True)
    Path("results_eval_rl").mkdir(exist_ok=True)
    
    # Phase 1: Check if we need to run initial data collection
    print_banner("Phase 1: Data Collection Check")
    
    # Check if we have existing training data
    existing_results = list(Path("results").glob("run_*/all_results.json"))
    if existing_results:
        print(f"✅ Found {len(existing_results)} existing result files")
        print("Skipping initial data collection")
    else:
        print("⚠️  No existing training data found")
        print("Running initial evaluation to collect training data...")
        
        # Run a small initial evaluation to collect training data
        if not run_command(
            "python -m src.main --config config.yaml --limit 50",
            "Initial data collection (50 tasks)"
        ):
            sys.exit(1)
    
    # Phase 2: Train RL selector from historical data
    print_banner("Phase 2: Train RL Selector")
    
    # Check if model already exists
    model_path = Path("models/rl_selector_v014.pt")
    if model_path.exists():
        print(f"⚠️  Model already exists at {model_path}")
        response = input("Retrain from scratch? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training phase")
        else:
            # Train from all available historical data
            results_pattern = "results/run_*"
            if not run_command(
                f"python -m src.rl_trainer {results_pattern}",
                "Training RL selector from historical data"
            ):
                print("⚠️  Training failed, but continuing...")
    else:
        # Train from all available historical data
        results_pattern = "results/run_*"
        if not run_command(
            f"python -m src.rl_trainer {results_pattern}",
            "Training RL selector from historical data"
        ):
            print("⚠️  No historical data available for training")
    
    # Phase 3: Run full evaluation with online RL training
    print_banner("Phase 3: Full Evaluation with RL Training")
    
    print("This will:")
    print("1. Generate 5 CoD solutions for each task")
    print("2. Use RL to select solutions (with exploration)")
    print("3. Evaluate all 5 solutions (for training)")
    print("4. Update RL model online")
    print("\nThis may take several hours for all BigCodeBench tasks.")
    
    response = input("\nProceed with full training run? (y/n): ")
    if response.lower() == 'y':
        if not run_command(
            "python -m src.main --config config_train_rl.yaml",
            "Full evaluation with RL training"
        ):
            print("❌ Training run failed")
            sys.exit(1)
    else:
        print("Skipping full training run")
    
    # Phase 4: Evaluation with trained RL selector
    print_banner("Phase 4: Evaluation with Trained RL Selector")
    
    print("This will:")
    print("1. Generate 5 CoD solutions for each task")
    print("2. Use trained RL to select the best solution")
    print("3. Evaluate ONLY the selected solution (5x speedup)")
    print("\nThis demonstrates the efficiency gains from RL selection.")
    
    response = input("\nProceed with RL evaluation? (y/n): ")
    if response.lower() == 'y':
        if not run_command(
            "python -m src.main --config config_eval_rl.yaml",
            "Evaluation with RL selector"
        ):
            print("❌ Evaluation failed")
            sys.exit(1)
    
    # Phase 5: Compare results
    print_banner("Phase 5: Results Comparison")
    
    # Check for evaluation results
    eval_metrics = Path("results_eval_rl") / "run_*" / "metrics.json"
    eval_files = list(Path("results_eval_rl").glob("run_*/metrics.json"))
    
    if eval_files:
        print(f"✅ Found {len(eval_files)} evaluation results")
        
        # Run comparison analysis
        if not run_command(
            f"python analyze_rl_results.py",
            "Analyzing RL selector performance"
        ):
            print("⚠️  Analysis script not found, skipping comparison")
    else:
        print("⚠️  No evaluation results found")
    
    print_banner("Pipeline Complete!")
    
    print("Summary:")
    print(f"- Model saved at: {model_path}")
    print(f"- Training results in: results_training/")
    print(f"- Evaluation results in: results_eval_rl/")
    print("\nNext steps:")
    print("1. Check results_eval_rl/*/metrics.json for Pass@k rates")
    print("2. Compare token usage between training and evaluation modes")
    print("3. Analyze RL selection accuracy")

if __name__ == "__main__":
    main() 
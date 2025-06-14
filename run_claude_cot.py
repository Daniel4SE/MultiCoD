#!/usr/bin/env python3
"""
Run Claude 3.5 with Chain-of-Thought prompting on BigCodeBench
"""

import sys
import subprocess

def main():
    # Run with a small limit for testing
    limit = 5  # Start with just 5 tasks for testing
    
    print(f"Running Claude 3.5 with Chain-of-Thought on {limit} BigCodeBench tasks...")
    print("This will show timing and token usage for each task.\n")
    
    # Run the modified claude35big.py
    cmd = [sys.executable, "claude35big.py", "--limit", str(limit)]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
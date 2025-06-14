#!/usr/bin/env python3
"""
Demonstration of the RL-based CoD Solution Selector

This script demonstrates:
1. The RL system design (state, action, reward)
2. How to train from historical data
3. How to use the selector during evaluation
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("RL-BASED COD SOLUTION SELECTOR DESIGN")
print("="*60)

print("\n1. PROBLEM FORMULATION")
print("-"*40)
print("""
Goal: Select the best CoD solution from 5 candidates BEFORE evaluation
Benefits:
- Save evaluation time/resources
- Learn patterns of successful solutions
- Improve overall system efficiency
""")

print("\n2. REINFORCEMENT LEARNING DESIGN")
print("-"*40)

print("\n📊 STATE SPACE (Features extracted from code):")
print("""
For each solution:
- Code length features (lines, tokens, characters)
- Code complexity (functions, loops, conditionals)
- CoD adherence metrics (word count, step count)
- Token usage (normalized)
- Strategy encoding (one-hot)
- Temperature value
- Relative features (compared to other solutions)
- Task features (prompt length, complexity)

Total: ~30 features per solution
""")

print("\n🎯 ACTION SPACE:")
print("""
- Discrete: Select one of 5 solutions
- Actions: {0, 1, 2, 3, 4}
""")

print("\n🏆 REWARD DESIGN:")
print("""
Primary rewards:
- +1.0 if selected solution passes tests
- +0.5 bonus if it's the first passing solution
- -0.5 if selected fails when others pass
- -0.1 if selected fails and all fail

Secondary rewards:
- +0.2 if selected has best CoD adherence
""")

print("\n3. NEURAL NETWORK ARCHITECTURE")
print("-"*40)
print("""
PolicyNetwork (Dueling DQN):
├── Feature Extraction
│   ├── Linear(feature_dim → 128)
│   ├── ReLU + Dropout(0.2)
│   ├── Linear(128 → 128)
│   └── ReLU + Dropout(0.2)
├── Value Head
│   ├── Linear(128 → 64)
│   ├── ReLU
│   └── Linear(64 → 1)
└── Advantage Head
    ├── Linear(128 → 64)
    ├── ReLU
    └── Linear(64 → 5)

Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
""")

print("\n4. TRAINING APPROACHES")
print("-"*40)

print("\n🔄 Offline Training (from historical data):")
print("""
1. Load previous evaluation results
2. For each task:
   - Extract features from 5 solutions
   - Select action using ε-greedy policy
   - Compute reward from known results
   - Store experience in replay buffer
   - Train using experience replay
3. Periodically update target network
4. Decay exploration rate (ε)
""")

print("\n🎮 Online Training (during evaluation):")
print("""
1. Generate 5 CoD solutions
2. Use RL to select best candidate
3. Evaluate only the selected solution (or all for training)
4. Compute reward based on results
5. Update policy online
6. Gradually improve selection accuracy
""")

print("\n5. EXAMPLE FEATURE EXTRACTION")
print("-"*40)

# Simulate feature extraction for a solution
example_features = {
    "Code Length": {
        "total_chars": 856,
        "lines": 32,
        "tokens": 145
    },
    "Code Complexity": {
        "functions": 2,
        "classes": 0,
        "loops": 3,
        "conditionals": 4,
        "try_blocks": 1,
        "imports": 3
    },
    "CoD Metrics": {
        "adherence_rate": 0.95,
        "avg_words": 3.2,
        "total_steps": 8
    },
    "Token Usage": {
        "total_tokens_normalized": 1.378,
        "output_ratio": 0.35
    },
    "Strategy": {
        "is_systematic": 1,
        "is_functional": 0,
        "is_error_handling": 0,
        "is_generator": 0,
        "is_comprehension": 0
    },
    "Other": {
        "temperature": 0.6,
        "relative_length": 0.85,
        "success_pattern_score": 0.75
    }
}

print("\nExample features for one solution:")
for category, features in example_features.items():
    print(f"\n{category}:")
    for name, value in features.items():
        print(f"  {name}: {value}")

print("\n6. EXPECTED BENEFITS")
print("-"*40)
print("""
1. Efficiency: Evaluate only the most promising solution
2. Learning: System improves over time
3. Adaptability: Learns task-specific patterns
4. Interpretability: Features show what makes solutions successful
""")

print("\n7. USAGE EXAMPLE")
print("-"*40)
print("""
# Train from historical data:
python -m src.rl_trainer results/run_20250525_* 

# Enable in evaluation:
# In config.yaml:
rl_selector:
  enabled: true
  training_mode: true  # For online learning

# Run evaluation with RL selection:
python -m src.main --config config.yaml
""")

# Create a visualization of the selection process
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Visualization 1: Q-values for 5 solutions
q_values = [0.8, 0.3, 0.9, 0.6, 0.4]  # Example Q-values
strategies = ['Systematic', 'Functional', 'Error-Safe', 'Generator', 'List-Comp']
colors = ['green' if q == max(q_values) else 'blue' for q in q_values]

ax1.bar(range(5), q_values, color=colors)
ax1.set_xticks(range(5))
ax1.set_xticklabels(strategies, rotation=45, ha='right')
ax1.set_ylabel('Q-value')
ax1.set_title('RL Solution Selection (Q-values)')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add selection indicator
selected_idx = q_values.index(max(q_values))
ax1.annotate('Selected!', xy=(selected_idx, q_values[selected_idx]), 
             xytext=(selected_idx, q_values[selected_idx] + 0.1),
             ha='center', fontsize=10, color='green', weight='bold')

# Visualization 2: Training progress
epochs = range(1, 21)
accuracy = [0.6 + 0.3 * (1 - np.exp(-0.3 * e)) + 0.05 * np.random.randn() for e in epochs]
reward = [0.2 + 0.8 * (1 - np.exp(-0.2 * e)) + 0.1 * np.random.randn() for e in epochs]

ax2.plot(epochs, accuracy, 'b-', label='Selection Accuracy')
ax2.plot(epochs, reward, 'r--', label='Average Reward')
ax2.set_xlabel('Training Epoch')
ax2.set_ylabel('Metric Value')
ax2.set_title('RL Training Progress')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_selector_demo.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: rl_selector_demo.png") 
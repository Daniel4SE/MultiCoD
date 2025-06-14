# RL-Based CoD Solution Selector Design

## Overview

The RL-Based CoD Solution Selector is a reinforcement learning system designed to intelligently select the best Chain of Draft (CoD) solution from 5 generated candidates, improving efficiency and learning from evaluation feedback.

## Problem Formulation

### Goal
Select the most likely successful CoD solution BEFORE expensive evaluation, saving computational resources while maintaining or improving Pass@k metrics.

### Benefits
1. **Efficiency**: Reduce evaluation time by ~80% (evaluate 1 instead of 5 solutions)
2. **Learning**: System improves selection accuracy over time
3. **Adaptability**: Learns task-specific patterns for better generalization
4. **Cost Reduction**: Fewer API calls to evaluation infrastructure

## Reinforcement Learning Design

### State Space (Features)

The state consists of ~30 features extracted from each of the 5 solutions:

#### Code-Level Features
- **Length Metrics**: Total characters, line count, token count
- **Complexity Metrics**: Function count, class count, loop count, conditional count
- **Error Handling**: Try-except blocks, assertions
- **Structure**: Import count, return statements

#### CoD-Specific Features
- **Adherence Rate**: Percentage of steps ≤5 words
- **Average Word Count**: Mean words per CoD step
- **Total Steps**: Number of draft steps
- **Token Usage**: Normalized total tokens, output/input ratio

#### Strategy Features
- **One-Hot Encoding**: Systematic, Functional, Error-Handling, Generator, Comprehension
- **Temperature**: Generation temperature used

#### Relative Features
- **Length Ratio**: Solution length vs. max/average
- **Complexity Comparison**: Relative to other solutions

#### Task Features
- **Prompt Length**: Characters in task prompt
- **Task Complexity**: Functions/classes in prompt

### Action Space
- **Type**: Discrete
- **Actions**: Select one of 5 solutions {0, 1, 2, 3, 4}

### Reward Design

```python
reward = 0.0

# Primary: Correctness
if selected_solution_passes:
    reward += 1.0
    if is_first_passing_solution:
        reward += 0.5  # Efficiency bonus
else:
    if any_solution_passes:
        reward -= 0.5  # Missed opportunity
    else:
        reward -= 0.1  # All failed

# Secondary: Quality
if has_best_adherence:
    reward += 0.2  # CoD quality bonus
```

## Neural Network Architecture

### Dueling DQN Architecture

```
Input: [batch_size, 5, feature_dim]
         ↓
Feature Network:
- Linear(feature_dim → 128)
- ReLU + Dropout(0.2)
- Linear(128 → 128)
- ReLU + Dropout(0.2)
         ↓
    ↙         ↘
Value Head    Advantage Head
- Linear(128→64)  - Linear(128→64)
- ReLU            - ReLU
- Linear(64→1)    - Linear(64→5)
    ↓              ↓
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

### Key Design Choices
1. **Dueling Architecture**: Separates value and advantage for better learning
2. **Dropout**: Prevents overfitting on limited training data
3. **Experience Replay**: Improves sample efficiency
4. **Target Network**: Stabilizes training

## Training Approaches

### 1. Offline Training (Historical Data)

```python
for epoch in epochs:
    for task_result in historical_results:
        # Extract features
        features = extract_features(task_result.solutions)
        
        # ε-greedy action selection
        if random() < epsilon:
            action = random_choice(5)
        else:
            action = argmax(Q(features))
        
        # Compute reward from known results
        reward = compute_reward(task_result, action)
        
        # Store in replay buffer
        buffer.add(features, action, reward)
        
        # Train from batch
        if len(buffer) >= batch_size:
            train_step(batch_sample(buffer))
    
    # Decay exploration
    epsilon *= decay_rate
```

### 2. Online Training (During Evaluation)

```python
# During evaluation
solutions = generate_5_cod_solutions(task)
selected_idx = rl_selector.select(task, solutions)

if training_mode:
    # Evaluate all for training data
    results = evaluate_all(solutions)
    reward = compute_reward(results, selected_idx)
    rl_selector.update(reward)
else:
    # Evaluate only selected
    result = evaluate(solutions[selected_idx])
```

## Implementation Components

### 1. CodeFeatureExtractor
- Extracts comprehensive features from code solutions
- Handles relative comparisons between solutions
- Normalizes numerical features

### 2. PolicyNetwork
- PyTorch neural network implementing Dueling DQN
- Processes all 5 solutions simultaneously
- Outputs Q-values for selection

### 3. RLCodeSelector
- Main interface for solution selection
- Manages training and inference
- Handles experience replay and model persistence

### 4. RLTrainer
- Orchestrates training from historical data
- Supports online learning during evaluation
- Tracks training metrics and visualizations

## Usage

### Configuration (config.yaml)
```yaml
rl_selector:
  enabled: true              # Enable RL selection
  model_path: "models/rl_selector.pt"
  feature_dim: 30
  training_mode: true        # Online learning
  evaluate_all: true         # For gathering training data
```

### Training from Historical Data
```bash
# Train on previous evaluation results
python -m src.rl_trainer results/run_20250525_*
```

### Integration with Evaluation
```bash
# Run evaluation with RL selection
python -m src.main --config config.yaml
```

## Expected Performance

### Training Convergence
- Initial accuracy: ~60% (better than random 20%)
- Converged accuracy: ~85-90% (selecting passing solutions)
- Training time: ~20 epochs on 100+ tasks

### Efficiency Gains
- Evaluation time: 5x speedup (1 vs 5 solutions)
- Token usage: 80% reduction in evaluation tokens
- API costs: Proportional reduction

### Pass@k Impact
- Pass@1: Maintained or improved (selecting best solutions)
- Pass@5: Slight degradation in training mode (acceptable tradeoff)

## Future Enhancements

1. **Meta-Learning**: Transfer learning across different task types
2. **Confidence Estimation**: Uncertainty quantification for selection
3. **Multi-Objective**: Balance correctness, efficiency, and code quality
4. **Active Learning**: Strategic selection of training examples
5. **Ensemble Methods**: Multiple RL agents with voting

## Conclusion

The RL-based CoD solution selector represents a significant advancement in making Multi-CoD evaluation more practical and efficient. By learning to identify successful solutions before evaluation, it reduces computational costs while maintaining high-quality results. The system's ability to improve through both offline and online training makes it adaptable to new task distributions and coding patterns. 
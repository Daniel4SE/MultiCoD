# BigCodeBench Multi-CoD Pass@k Evaluation

Evaluates Chain of Draft (CoD) methodology on BigCodeBench using 5 dynamically generated strategies per task.

## Features

- **Dynamic Strategy Generation**: Creates 5 task-specific CoD strategies using Claude
- **Multiple Attempts**: Generates code with each strategy
- **Pass@k Metrics**: Measures success rates for k=1 to 5 attempts
- **Comprehensive Analysis**: Strategy performance, adherence rates, and visualizations

## Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Run
```bash
python -m src.main --limit 10
```

### Components
- `src/strategy_generator.py`: Generate diverse strategies
- `src/cod_generator.py`: Generate code using CoD
- `src/evaluator.py`: Evaluate solutions
- `src/rl_selector.py`: RL-based solution selection (NEW!)
- `src/rl_trainer.py`: Train RL selector

### RL-Based Solution Selector

The system includes an intelligent RL-based selector that learns to choose the best CoD solution before evaluation:

#### Features
- **Efficiency**: Evaluate only the most promising solution (5x speedup)
- **Learning**: Improves selection accuracy over time
- **Features**: Extracts ~30 features from code (complexity, length, CoD adherence, etc.)
- **Architecture**: Dueling DQN with experience replay

#### Training
```bash
# Train from historical results
python -m src.rl_trainer results/run_*

# Enable in config.yaml
rl_selector:
  enabled: true
  training_mode: true  # For online learning
```

#### Design
- **State**: Code features (length, complexity, CoD metrics)
- **Action**: Select one of 5 solutions
- **Reward**: +1 for correct, +0.5 for first passing, -0.5 for missing

See `RL_SELECTOR_DESIGN.md` for complete details.
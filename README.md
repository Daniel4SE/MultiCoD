# MultiCoD: Reinforcement Learning-Based Selection in Chain-of-Draft for Code Generation

MultiCoD is a reinforcement learning-based framework for code generation that extends Chain-of-Draft (CoD) methodology. It learns to select the best candidate from a diverse set of CoD-generated solutions, achieving state-of-the-art performance across multiple benchmarks while significantly reducing token usage and improving generation efficiency.

## Overview

Large language models (LLMs) have become powerful tools for automating software engineering tasks, but often produce brittle or incorrect code. While Chain-of-Thought (CoT) prompting addresses this by generating intermediate reasoning steps, it requires excessive verbosity.

Chain-of-Draft (CoD) prompting offers a more efficient alternative, using succinct drafts (≤5 words per step) inspired by human problem-solving. MultiCoD builds on this by using reinforcement learning to select the optimal solution from multiple drafts, addressing CoD's inherent stochasticity.

## Key Features

- **Dynamic Strategy Generation**: Creates task-specific CoD strategies using Claude
- **Multiple Drafts**: Generates code with diverse strategies
- **RL-Based Selection**: Learns to choose the optimal solution without evaluation
- **Token Efficiency**: Uses 49-60% fewer tokens than CoT methods
- **Improved Performance**: Achieves state-of-the-art results across benchmarks
- **Comprehensive Metrics**: Pass@k scores, strategy performance, and analysis

## Performance Highlights

- **BigCodeBench**: 36.3% Pass@1 with Claude-3-7-Sonnet (new SOTA)
- **MBPP**: 94.5% accuracy, outperforming QualityFlow (94.2%)
- **SWE-bench Verified**: 64.0% resolution rate, with dramatic improvements for open-source models
- **Defects4J**: 80.1% Compilation Rate, 68.0% Pass@1, 70.2 BLEU score

### Performance Visualizations

The project includes comprehensive visualizations of performance across different benchmarks:

- Radar charts comparing prompting strategies across foundation models
- Bar charts showing performance progression across different prompting methods
- Performance vs. token consumption analysis demonstrating efficiency
- Detailed breakdown of improvements for both closed-source and open-source models

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
- `src/rl_selector.py`: RL-based solution selection
- `src/rl_trainer.py`: Train RL selector

## RL-Based Solution Selector

The system includes an intelligent RL-based selector that learns to choose the best CoD solution before evaluation:

### Features
- **Efficiency**: Evaluate only the most promising solution (5x speedup)
- **Learning**: Improves selection accuracy over time
- **Features**: Extracts ~30 features from code (complexity, length, CoD adherence, etc.)
- **Architecture**: Dueling DQN with experience replay

### Training
```bash
# Train from historical results
python -m src.rl_trainer results/run_*

# Enable in config.yaml
rl_selector:
  enabled: true
  training_mode: true  # For online learning
```

## Efficiency Analysis

MultiCoD shows significant efficiency improvements over Chain-of-Thought approaches:

### Generation Time (seconds)
| Benchmark | Standard | CoT | Chain-of-Draft | MultiCoD (best) | MultiCoD/CoT |
|-----------|----------|-----|----------------|-----------------|--------------|
| BigCodeBench | 5.2 | 12.8 | 6.4 | 6.8 | 0.53× |
| MBPP | 3.7 | 9.2 | 4.8 | 5.1 | 0.55× |
| SWE-bench | 14.6 | 35.9 | 18.2 | 19.3 | 0.54× |
| Defects4J | 17.3 | 42.1 | 20.9 | 22.4 | 0.53× |

### Token Consumption
| Benchmark | Standard | CoT | Chain-of-Draft | MultiCoD (best) | MultiCoD/CoT |
|-----------|----------|-----|----------------|-----------------|--------------|
| BigCodeBench | 410 | 986 | 583 | 596 | 0.60× |
| MBPP | 326 | 765 | 452 | 471 | 0.62× |
| SWE-bench | 1,243 | 3,894 | 1,927 | 1,952 | 0.50× |
| Defects4J | 1,572 | 4,618 | 2,204 | 2,245 | 0.49× |

The most dramatic savings occur in the reasoning component, where MultiCoD uses only 692 tokens compared to CoT's 2,451 tokens on SWE-bench—a 71.8% reduction.

See `RL_SELECTOR_DESIGN.md` for complete details.

## Repository

Source code is available at: https://anonymous.4open.science/r/MultiCoD
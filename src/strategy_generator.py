#!/usr/bin/env python3
"""Generate diverse Chain of Draft strategies for code generation"""

import json
import logging
from typing import Dict, List
import anthropic
from prompts.strategy_generation import STRATEGY_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """Generate diverse CoD strategies for BigCodeBench tasks"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate_strategies(self, task: Dict) -> List[Dict]:
        """Generate 5 task-specific strategies"""
        
        # Build task description
        task_description = task.get('description', '')
        if 'docstring' in task:
            task_description = task['docstring']
        
        # Extract function signature
        task_signature = task['prompt'].strip()
        
        prompt = STRATEGY_GENERATION_PROMPT.format(
            task_description=task_description,
            task_signature=task_signature
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract token usage for strategy generation
            self.last_token_usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Parse JSON response
            content = response.content[0].text
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            strategies_data = json.loads(json_str)
            strategies = strategies_data['strategies']
            
            # Validate we got 5 strategies
            if len(strategies) != 5:
                logger.warning(f"Got {len(strategies)} strategies instead of 5")
                # Pad or trim to exactly 5
                while len(strategies) < 5:
                    strategies.append(self._get_default_strategy(len(strategies)))
                strategies = strategies[:5]
            
            logger.info(f"Generated {len(strategies)} strategies for task {task['task_id']}")
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating strategies: {e}")
            self.last_token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            # Return default strategies on error
            return [self._get_default_strategy(i) for i in range(5)]
    
    def _get_default_strategy(self, index: int) -> Dict:
        """Get default strategy as fallback"""
        defaults = [
            {
                "name": "Systematic Approach",
                "focus": "Step-by-step implementation",
                "instruction": "Break down the problem systematically from imports to return",
                "key_priorities": ["clarity", "completeness", "correctness"]
            },
            {
                "name": "Core Functionality First",
                "focus": "Main logic implementation",
                "instruction": "Implement the core functionality first, then add supporting code",
                "key_priorities": ["functionality", "efficiency", "simplicity"]
            },
            {
                "name": "Error-Safe Implementation",
                "focus": "Robust error handling",
                "instruction": "Build with error handling and validation from the start",
                "key_priorities": ["safety", "validation", "error handling"]
            },
            {
                "name": "Test-Driven Approach",
                "focus": "Meeting test requirements",
                "instruction": "Analyze expected behavior and build to pass tests",
                "key_priorities": ["correctness", "test coverage", "edge cases"]
            },
            {
                "name": "Optimized Solution",
                "focus": "Performance and efficiency",
                "instruction": "Focus on optimal implementation with good performance",
                "key_priorities": ["performance", "memory efficiency", "scalability"]
            }
        ]
        return defaults[index % len(defaults)]

    def generate_custom_strategy(self, task: Dict, focus: str) -> str:
        """Generate a custom strategy based on specific focus area"""
        
        prompt = f"""
        Generate a Chain of Draft (CoD) strategy for solving this coding problem.
        The strategy should focus on: {focus}
        
        Problem: {task['prompt']}
        
        Rules for the strategy:
        1. Must encourage thinking in minimal draft steps
        2. Each step should be 5 words or less
        3. Strategy should be specific to the problem type
        
        Return only the strategy description in one sentence.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            strategy = response.content[0].text.strip()
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating custom strategy: {e}")
            return "Break down the problem into minimal steps, each step in 5 words max"  # Fallback to default 
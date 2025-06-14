#!/usr/bin/env python3
"""Main evaluation script for BigCodeBench Multi-CoD Pass@k"""

import os
import json
import time
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from src.strategy_generator import StrategyGenerator
from src.cod_generator import CoDGenerator
from src.evaluator import PassKEvaluator
from src.bigcodebench_loader import BigCodeBenchLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiCoDEvaluator:
    """Main evaluation pipeline"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize API key
        api_key = self.config['api']['anthropic_key']
        if api_key.startswith('${'):
            api_key = os.getenv(api_key[2:-1])
        
        if not api_key:
            raise ValueError("Anthropic API key not set")
        
        # Initialize components
        self.strategy_generator = StrategyGenerator(api_key)
        self.cod_generator = CoDGenerator(api_key)
        self.evaluator = PassKEvaluator(
            timeout=self.config['evaluation']['timeout_per_test'],
            use_docker=self.config['evaluation']['use_docker']
        )
        self.loader = BigCodeBenchLoader(
            subset=self.config['bigcodebench']['subset'],
            cache_dir=self.config['bigcodebench']['cache_dir'],
            version=self.config['bigcodebench'].get('version', 'v0.4.1')
        )
        
        # Initialize RL selector if enabled
        self.rl_selector = None
        self.rl_trainer = None
        if self.config['rl_selector']['enabled']:
            from src.rl_selector import RLCodeSelector
            from src.rl_trainer import RLTrainer
            
            self.rl_trainer = RLTrainer(
                model_save_path=self.config['rl_selector']['model_path'],
                feature_dim=self.config['rl_selector']['feature_dim']
            )
            self.rl_selector = self.rl_trainer.selector
            logger.info("RL selector enabled")
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config['output']['results_dir']) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Multi-CoD Evaluator")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        
        # Load tasks
        tasks = self.loader.load_dataset(
            max_tasks=self.config['bigcodebench']['max_tasks']
        )
        logger.info(f"Loaded {len(tasks)} tasks")
        
        # Results storage
        all_results = []
        pass_at_k_summary = {f'pass@{k}': 0 for k in range(1, 6)}
        
        # Process each task
        for task in tqdm(tasks, desc="Processing tasks"):
            try:
                result = self.process_task(task)
                if result:  # Only add if result is valid
                    all_results.append(result)
                    
                    # Update pass@k summary
                    for k in range(1, 6):
                        if result['pass_at_k'][f'pass@{k}']:
                            pass_at_k_summary[f'pass@{k}'] += 1
                    
                    # Save intermediate results
                    if self.config['output']['save_intermediate']:
                        self.save_task_result(result)
                
                # Rate limiting
                time.sleep(self.config['api']['rate_limit_delay'])
                
            except Exception as e:
                logger.error(f"Error processing task {task['task_id']}: {e}")
                continue
        
        # Calculate final metrics
        final_metrics = self.calculate_final_metrics(all_results, len(tasks))
        
        # Save all results
        self.save_final_results(all_results, final_metrics)
        
        # Print summary
        self.print_summary(final_metrics)
    
    def process_task(self, task: Dict) -> Optional[Dict]:
        """Process a single task with 5 CoD strategies"""
        logger.info(f"Processing task: {task['task_id']}")
        
        # Step 1: Generate 5 strategies
        strategies = self.strategy_generator.generate_strategies(task)
        strategy_token_usage = getattr(self.strategy_generator, 'last_token_usage', 
                                     {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
        
        # Step 2: Generate code with each strategy
        solutions = []
        temperatures = self.config['chain_of_draft']['temperatures']
        total_cod_tokens = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        
        for i, strategy in enumerate(strategies):
            temperature = temperatures[i % len(temperatures)]
            
            solution = self.cod_generator.generate_code(
                task, strategy, temperature
            )
            
            # Validate CoD steps
            if solution['success']:
                validation = self.cod_generator.validate_steps(solution['steps'])
                solution['validation'] = validation
                
                # Accumulate token usage
                if 'token_usage' in solution:
                    for key in total_cod_tokens:
                        total_cod_tokens[key] += solution['token_usage'][key]
            
            solutions.append(solution)
        
        # Calculate average tokens per CoD solution
        avg_cod_tokens = {
            'input_tokens': total_cod_tokens['input_tokens'] / 5,
            'output_tokens': total_cod_tokens['output_tokens'] / 5,
            'total_tokens': total_cod_tokens['total_tokens'] / 5
        }
        
        # Step 3: RL Selection (if enabled)
        rl_selection_info = None
        if self.rl_selector and not self.config['rl_selector']['evaluate_all']:
            # Use RL to select which solution to evaluate
            selected_idx, rl_info = self.rl_selector.select_solution(
                task, solutions, 
                training=self.config['rl_selector']['training_mode']
            )
            rl_selection_info = rl_info
            
            # Evaluate only the selected solution
            selected_solutions = [solutions[selected_idx]]
            evaluation_result = self.evaluator.evaluate_solutions(task, selected_solutions)
            
            # Save the actual evaluation result before modifying
            actual_result = evaluation_result['individual_results'][0] if evaluation_result['individual_results'] else {'passed': False, 'error': 'Evaluation failed'}
            
            # Map back to full results format
            evaluation_result['individual_results'] = [
                {'strategy': solutions[i]['strategy'], 'passed': False, 'error': 'Not evaluated'}
                for i in range(5)
            ]
            # Put the actual result in the correct position
            evaluation_result['individual_results'][selected_idx] = {
                'strategy': solutions[selected_idx]['strategy'],
                'passed': actual_result['passed'],
                'error': actual_result.get('error', '')
            }
            # Update pass@k based on the single evaluated solution
            evaluation_result['pass_at_k'] = {f'pass@{k}': actual_result['passed'] for k in range(1, 6)}
            
        else:
            # Evaluate all solutions (normal mode or training mode)
            evaluation_result = self.evaluator.evaluate_solutions(task, solutions)
        
        # Combine results
        result = {
            'task_id': task['task_id'],
            'strategies': strategies,
            'solutions': solutions,
            'evaluation': evaluation_result,
            'pass_at_k': evaluation_result['pass_at_k'],
            'timestamp': datetime.now().isoformat(),
            'token_usage': {
                'strategy_generation': strategy_token_usage,
                'cod_generation_total': total_cod_tokens,
                'cod_generation_average': avg_cod_tokens,
                'grand_total': {
                    'input_tokens': strategy_token_usage['input_tokens'] + total_cod_tokens['input_tokens'],
                    'output_tokens': strategy_token_usage['output_tokens'] + total_cod_tokens['output_tokens'],
                    'total_tokens': strategy_token_usage['total_tokens'] + total_cod_tokens['total_tokens']
                }
            }
        }
        
        if rl_selection_info:
            result['rl_selection'] = rl_selection_info
        
        # Step 4: Online RL training (if enabled)
        if self.rl_selector and self.config['rl_selector']['training_mode'] and self.rl_trainer:
            training_info = self.rl_trainer.online_training_step(task, solutions, result)
            result['rl_training'] = training_info
        
        return result
    
    def calculate_final_metrics(self, results: List[Dict], total_tasks: int) -> Dict:
        """Calculate final evaluation metrics"""
        
        # Pass@k rates
        pass_at_k_counts = {f'pass@{k}': 0 for k in range(1, 6)}
        for result in results:
            for k in range(1, 6):
                if result['pass_at_k'][f'pass@{k}']:
                    pass_at_k_counts[f'pass@{k}'] += 1
        
        pass_at_k_rates = {
            k: count / total_tasks 
            for k, count in pass_at_k_counts.items()
        }
        
        # Strategy performance
        strategy_stats = {}
        for result in results:
            for sol in result['evaluation']['individual_results']:
                strategy = sol['strategy']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'total': 0, 'passed': 0}
                strategy_stats[strategy]['total'] += 1
                if sol['passed']:
                    strategy_stats[strategy]['passed'] += 1
        
        strategy_performance = {
            strategy: stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            for strategy, stats in strategy_stats.items()
        }
        
        # CoD adherence
        total_steps = 0
        valid_steps = 0
        for result in results:
            for sol in result['solutions']:
                if 'validation' in sol:
                    total_steps += sol['validation']['total_steps']
                    valid_steps += sol['validation']['valid_steps']
        
        adherence_rate = valid_steps / total_steps if total_steps > 0 else 0
        
        # Token usage statistics
        total_tokens = {'input': 0, 'output': 0, 'total': 0}
        strategy_tokens = {'input': 0, 'output': 0, 'total': 0}
        cod_tokens = {'input': 0, 'output': 0, 'total': 0}
        
        for result in results:
            if 'token_usage' in result:
                # Strategy generation tokens
                strategy_tokens['input'] += result['token_usage']['strategy_generation']['input_tokens']
                strategy_tokens['output'] += result['token_usage']['strategy_generation']['output_tokens']
                strategy_tokens['total'] += result['token_usage']['strategy_generation']['total_tokens']
                
                # CoD generation tokens
                cod_tokens['input'] += result['token_usage']['cod_generation_total']['input_tokens']
                cod_tokens['output'] += result['token_usage']['cod_generation_total']['output_tokens']
                cod_tokens['total'] += result['token_usage']['cod_generation_total']['total_tokens']
                
                # Grand total
                total_tokens['input'] += result['token_usage']['grand_total']['input_tokens']
                total_tokens['output'] += result['token_usage']['grand_total']['output_tokens']
                total_tokens['total'] += result['token_usage']['grand_total']['total_tokens']
        
        # RL selector statistics
        rl_stats = None
        if self.rl_selector:
            rl_correct = sum(1 for r in results if 'rl_selection' in r and 
                           r['evaluation']['individual_results'][r['rl_selection']['selected_idx']]['passed'])
            rl_total = sum(1 for r in results if 'rl_selection' in r)
            
            rl_stats = {
                'selections_made': rl_total,
                'correct_selections': rl_correct,
                'accuracy': rl_correct / rl_total if rl_total > 0 else 0
            }
        
        num_tasks = len(results)
        token_usage_stats = {
            'total_tokens_used': total_tokens,
            'average_tokens_per_task': {
                'input': total_tokens['input'] / num_tasks if num_tasks > 0 else 0,
                'output': total_tokens['output'] / num_tasks if num_tasks > 0 else 0,
                'total': total_tokens['total'] / num_tasks if num_tasks > 0 else 0
            },
            'strategy_generation': {
                'total': strategy_tokens,
                'average_per_task': {
                    'input': strategy_tokens['input'] / num_tasks if num_tasks > 0 else 0,
                    'output': strategy_tokens['output'] / num_tasks if num_tasks > 0 else 0,
                    'total': strategy_tokens['total'] / num_tasks if num_tasks > 0 else 0
                }
            },
            'cod_generation': {
                'total': cod_tokens,
                'average_per_task': {
                    'input': cod_tokens['input'] / num_tasks if num_tasks > 0 else 0,
                    'output': cod_tokens['output'] / num_tasks if num_tasks > 0 else 0,
                    'total': cod_tokens['total'] / num_tasks if num_tasks > 0 else 0
                },
                'average_per_solution': {
                    'input': cod_tokens['input'] / (num_tasks * 5) if num_tasks > 0 else 0,
                    'output': cod_tokens['output'] / (num_tasks * 5) if num_tasks > 0 else 0,
                    'total': cod_tokens['total'] / (num_tasks * 5) if num_tasks > 0 else 0
                }
            }
        }
        
        metrics = {
            'total_tasks': total_tasks,
            'evaluated_tasks': len(results),
            'pass_at_k_counts': pass_at_k_counts,
            'pass_at_k_rates': pass_at_k_rates,
            'strategy_performance': strategy_performance,
            'cod_adherence_rate': adherence_rate,
            'token_usage_stats': token_usage_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        if rl_stats:
            metrics['rl_selector_stats'] = rl_stats
        
        return metrics
    
    def save_task_result(self, result: Dict):
        """Save individual task result"""
        task_id = result['task_id'].replace('/', '_')
        file_path = self.output_dir / f"task_{task_id}.json"
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_final_results(self, results: List[Dict], metrics: Dict):
        """Save all results and metrics"""
        
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        with open(self.output_dir / 'all_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for result in results:
            row = {
                'task_id': result['task_id'],
                **result['pass_at_k'],
                'first_passing_strategy': result['evaluation'].get('first_passing_strategy', 'None')
            }
            if 'rl_selection' in result:
                row['rl_selected_idx'] = result['rl_selection']['selected_idx']
                row['rl_selected_strategy'] = result['rl_selection']['selected_strategy']
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / 'summary.csv', index=False)
        
        # Save generated code if configured
        if self.config['evaluation']['save_generated_code']:
            code_dir = self.output_dir / 'generated_code'
            code_dir.mkdir(exist_ok=True)
            
            for result in results:
                task_id = result['task_id'].replace('/', '_')
                for i, sol in enumerate(result['solutions']):
                    if sol.get('code'):
                        file_path = code_dir / f"{task_id}_strategy_{i+1}.py"
                        with open(file_path, 'w') as f:
                            f.write(sol['code'])
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total tasks: {metrics['total_tasks']}")
        print(f"Evaluated tasks: {metrics['evaluated_tasks']}")
        print("\nPass@k Results:")
        print("-"*30)
        
        for k in range(1, 6):
            count = metrics['pass_at_k_counts'][f'pass@{k}']
            rate = metrics['pass_at_k_rates'][f'pass@{k}']
            print(f"Pass@{k}: {count}/{metrics['total_tasks']} ({rate:.2%})")
        
        print("\nStrategy Performance:")
        print("-"*30)
        for strategy, performance in sorted(
            metrics['strategy_performance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:  # Top 10 strategies
            print(f"{strategy}: {performance:.2%}")
        
        print(f"\nCoD Adherence Rate: {metrics['cod_adherence_rate']:.2%}")
        
        # RL selector statistics
        if 'rl_selector_stats' in metrics:
            print("\nRL Selector Performance:")
            print("-"*30)
            rl_stats = metrics['rl_selector_stats']
            print(f"Selections made: {rl_stats['selections_made']}")
            print(f"Correct selections: {rl_stats['correct_selections']}")
            print(f"Selection accuracy: {rl_stats['accuracy']:.2%}")
        
        # Token usage statistics
        if 'token_usage_stats' in metrics:
            print("\nToken Usage Statistics:")
            print("-"*30)
            stats = metrics['token_usage_stats']
            
            print(f"Total tokens used: {stats['total_tokens_used']['total']:,}")
            print(f"  Input tokens: {stats['total_tokens_used']['input']:,}")
            print(f"  Output tokens: {stats['total_tokens_used']['output']:,}")
            
            print(f"\nAverage tokens per task: {stats['average_tokens_per_task']['total']:.0f}")
            print(f"  Input: {stats['average_tokens_per_task']['input']:.0f}")
            print(f"  Output: {stats['average_tokens_per_task']['output']:.0f}")
            
            print(f"\nStrategy generation (per task): {stats['strategy_generation']['average_per_task']['total']:.0f} tokens")
            print(f"\nCoD generation (per task): {stats['cod_generation']['average_per_task']['total']:.0f} tokens")
            print(f"CoD generation (per solution): {stats['cod_generation']['average_per_solution']['total']:.0f} tokens")
        
        print(f"\nResults saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Multi-CoD on BigCodeBench'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tasks to evaluate'
    )
    
    args = parser.parse_args()
    
    # Override config if limit specified
    if args.limit:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['bigcodebench']['max_tasks'] = args.limit
        
        # Save temporary config
        temp_config = 'temp_config.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        evaluator = MultiCoDEvaluator(temp_config)
        os.unlink(temp_config)
    else:
        evaluator = MultiCoDEvaluator(args.config)
    
    # Run evaluation
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
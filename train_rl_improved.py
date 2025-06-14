#!/usr/bin/env python3
"""
Improved RL training script for Multi-CoD selector
Addresses the 25% performance issue with better training methodology
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rl_trainer import RLTrainer
from src.main import MultiCoDEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRLTraining:
    def __init__(self, config_path='config_train_rl.yaml'):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize trainer with better hyperparameters
        self.trainer = RLTrainer(
            model_save_path='models/rl_selector_improved.pt',
            learning_rate=5e-5,  # Lower learning rate for stability
            epsilon=0.3,  # Higher initial exploration
            epsilon_decay=0.997,  # Slower decay
            epsilon_min=0.05,  # Higher minimum exploration
            batch_size=64,  # Larger batch size
            buffer_size=50000,  # Much larger replay buffer
            update_frequency=10,  # Update every 10 steps
            target_update_frequency=500  # Update target network less frequently
        )
        
        self.training_history = {
            'rewards': [],
            'losses': [],
            'accuracies': [],
            'pass_rates': []
        }
    
    def collect_training_data(self, num_tasks=500):
        """Collect diverse training data from BigCodeBench"""
        logger.info(f"Collecting training data from {num_tasks} tasks...")
        
        # Use the standard evaluator to generate solutions
        evaluator = MultiCoDEvaluator(self.config_path)
        
        # Override to ensure we evaluate all solutions for training
        evaluator.config['rl_selector']['evaluate_all'] = True
        evaluator.config['bigcodebench']['max_tasks'] = num_tasks
        
        # Collect data
        all_experiences = []
        tasks = evaluator.loader.load_dataset(max_tasks=num_tasks)
        
        for i, task in enumerate(tqdm(tasks, desc="Collecting data")):
            try:
                # Generate strategies and solutions
                strategies = evaluator.strategy_generator.generate_strategies(task)
                solutions = []
                
                for j, strategy in enumerate(strategies):
                    temperature = evaluator.config['chain_of_draft']['temperatures'][j]
                    solution = evaluator.cod_generator.generate_code(task, strategy, temperature)
                    if solution['success']:
                        validation = evaluator.cod_generator.validate_steps(solution['steps'])
                        solution['validation'] = validation
                    solutions.append(solution)
                
                # Evaluate all solutions
                evaluation_result = evaluator.evaluator.evaluate_solutions(task, solutions)
                
                # Extract features for all solutions
                features_list = []
                for sol_idx, solution in enumerate(solutions):
                    features = self.trainer.selector.feature_extractor.extract_features(
                        task, solution, solutions
                    )
                    features_list.append(features)
                
                # Find the best solution(s)
                passing_indices = [
                    idx for idx, result in enumerate(evaluation_result['individual_results'])
                    if result['passed']
                ]
                
                # Create training experiences
                if passing_indices:
                    # Positive examples: all passing solutions
                    for idx in passing_indices:
                        experience = {
                            'task': task,
                            'solutions': solutions,
                            'features': features_list,
                            'selected_idx': idx,
                            'reward': 1.0,  # Base reward for passing
                            'passed': True
                        }
                        
                        # Bonus for first passing solution
                        if idx == passing_indices[0]:
                            experience['reward'] += 0.5
                        
                        # Bonus for best adherence among passing
                        adherences = [s.get('validation', {}).get('adherence_rate', 0) for s in solutions]
                        if idx < len(adherences) and adherences[idx] == max(adherences[i] for i in passing_indices):
                            experience['reward'] += 0.2
                        
                        all_experiences.append(experience)
                    
                    # Negative examples: all failing solutions (with penalty)
                    failing_indices = [i for i in range(5) if i not in passing_indices]
                    for idx in failing_indices:
                        experience = {
                            'task': task,
                            'solutions': solutions,
                            'features': features_list,
                            'selected_idx': idx,
                            'reward': -0.5,  # Penalty for selecting failing when passing exists
                            'passed': False
                        }
                        all_experiences.append(experience)
                else:
                    # No passing solutions - smaller penalty for all
                    for idx in range(5):
                        experience = {
                            'task': task,
                            'solutions': solutions,
                            'features': features_list,
                            'selected_idx': idx,
                            'reward': -0.1,  # Small penalty when nothing passes
                            'passed': False
                        }
                        all_experiences.append(experience)
                
                # Save intermediate data every 50 tasks
                if (i + 1) % 50 == 0:
                    self.save_experiences(all_experiences, f'training_data_{i+1}.json')
                    logger.info(f"Saved {len(all_experiences)} experiences")
                
            except Exception as e:
                logger.error(f"Error processing task {task['task_id']}: {e}")
                continue
        
        logger.info(f"Collected {len(all_experiences)} training experiences")
        return all_experiences
    
    def train_with_curriculum(self, experiences, epochs=10):
        """Train with curriculum learning - start with easy examples"""
        logger.info("Starting curriculum training...")
        
        # Sort experiences by difficulty (passing solutions first)
        experiences_sorted = sorted(experiences, key=lambda x: -x['reward'])
        
        # Split into curriculum stages
        stage_size = len(experiences_sorted) // 5
        stages = [
            experiences_sorted[:stage_size],  # Easy: mostly passing
            experiences_sorted[:stage_size*2],  # Add some harder
            experiences_sorted[:stage_size*3],  # More diverse
            experiences_sorted[:stage_size*4],  # Nearly full
            experiences_sorted  # Full dataset
        ]
        
        for stage_idx, stage_data in enumerate(stages):
            logger.info(f"Training stage {stage_idx + 1}/5 with {len(stage_data)} examples")
            
            for epoch in range(epochs):
                np.random.shuffle(stage_data)
                
                epoch_rewards = []
                epoch_losses = []
                correct_selections = 0
                total_selections = 0
                
                for exp in tqdm(stage_data, desc=f"Stage {stage_idx+1} Epoch {epoch+1}"):
                    # Extract state
                    features_tensor = torch.FloatTensor(exp['features']).unsqueeze(0)
                    
                    # Select action (with exploration)
                    selected_idx, _ = self.trainer.selector.select_solution(
                        exp['task'], exp['solutions'], training=True
                    )
                    
                    # Store experience
                    self.trainer.selector.store_experience(
                        exp['task'], exp['solutions'], 
                        exp['selected_idx'], exp['reward']
                    )
                    
                    # Update model
                    loss = self.trainer.selector.train_step()
                    if loss is not None:
                        epoch_losses.append(loss)
                    
                    # Track metrics
                    epoch_rewards.append(exp['reward'])
                    if selected_idx == exp['selected_idx'] and exp['passed']:
                        correct_selections += 1
                    total_selections += 1
                
                # Decay epsilon
                self.trainer.selector.decay_epsilon()
                
                # Log epoch statistics
                avg_reward = np.mean(epoch_rewards)
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                accuracy = correct_selections / total_selections if total_selections > 0 else 0
                
                logger.info(f"Stage {stage_idx+1} Epoch {epoch+1}: "
                          f"Avg Reward: {avg_reward:.3f}, "
                          f"Avg Loss: {avg_loss:.4f}, "
                          f"Accuracy: {accuracy:.2%}, "
                          f"Epsilon: {self.trainer.selector.epsilon:.3f}")
                
                self.training_history['rewards'].append(avg_reward)
                self.training_history['losses'].append(avg_loss)
                self.training_history['accuracies'].append(accuracy)
            
            # Update target network after each stage
            self.trainer.selector.update_target_network()
    
    def validate_on_holdout(self, num_tasks=50):
        """Validate on held-out tasks"""
        logger.info(f"Validating on {num_tasks} held-out tasks...")
        
        evaluator = MultiCoDEvaluator(self.config_path)
        evaluator.config['rl_selector']['evaluate_all'] = True
        evaluator.config['bigcodebench']['max_tasks'] = num_tasks
        evaluator.rl_selector = self.trainer.selector
        
        # Skip tasks used in training
        tasks = evaluator.loader.load_dataset(max_tasks=num_tasks + 500)[500:]
        
        correct = 0
        total = 0
        pass_count = 0
        
        for task in tqdm(tasks[:num_tasks], desc="Validation"):
            try:
                result = evaluator.process_task(task)
                if result and 'rl_selection' in result:
                    selected_idx = result['rl_selection']['selected_idx']
                    selected_passed = result['evaluation']['individual_results'][selected_idx]['passed']
                    any_passed = any(r['passed'] for r in result['evaluation']['individual_results'])
                    
                    if selected_passed:
                        correct += 1
                        pass_count += 1
                    elif any_passed:
                        # Selected failed when there was a passing solution
                        pass
                    else:
                        # No solutions passed - not a selection error
                        correct += 1
                    
                    total += 1
            except Exception as e:
                logger.error(f"Validation error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        pass_rate = pass_count / total if total > 0 else 0
        
        logger.info(f"Validation Results: Accuracy: {accuracy:.2%}, Pass Rate: {pass_rate:.2%}")
        return accuracy, pass_rate
    
    def save_experiences(self, experiences, filename):
        """Save experiences for later use"""
        save_data = []
        for exp in experiences:
            save_data.append({
                'task_id': exp['task']['task_id'],
                'selected_idx': exp['selected_idx'],
                'reward': exp['reward'],
                'passed': exp['passed'],
                'features': [f.tolist() if isinstance(f, np.ndarray) else f for f in exp['features']]
            })
        
        with open(f'training_data/{filename}', 'w') as f:
            json.dump(save_data, f)
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Rewards
        axes[0, 0].plot(self.training_history['rewards'])
        axes[0, 0].set_title('Average Reward per Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        
        # Losses
        axes[0, 1].plot(self.training_history['losses'])
        axes[0, 1].set_title('Average Loss per Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Accuracies
        axes[1, 0].plot(self.training_history['accuracies'])
        axes[1, 0].set_title('Selection Accuracy per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        
        # Pass rates (if available)
        if self.training_history['pass_rates']:
            axes[1, 1].plot(self.training_history['pass_rates'])
            axes[1, 1].set_title('Pass Rate per Validation')
            axes[1, 1].set_xlabel('Validation Round')
            axes[1, 1].set_ylabel('Pass Rate')
        
        plt.tight_layout()
        plt.savefig('models/improved_training_history.png')
        logger.info("Saved training history plot")

def main():
    """Run improved RL training"""
    
    # Create directories
    Path('training_data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = ImprovedRLTraining()
    
    # Phase 1: Collect diverse training data
    logger.info("Phase 1: Data Collection")
    experiences = trainer.collect_training_data(num_tasks=200)  # Start with 200 tasks
    
    # Phase 2: Curriculum training
    logger.info("Phase 2: Curriculum Training")
    trainer.train_with_curriculum(experiences, epochs=5)
    
    # Phase 3: Validation
    logger.info("Phase 3: Validation")
    val_accuracy, val_pass_rate = trainer.validate_on_holdout(num_tasks=30)
    trainer.training_history['pass_rates'].append(val_pass_rate)
    
    # Phase 4: Fine-tuning if needed
    if val_accuracy < 0.8:
        logger.info("Phase 4: Fine-tuning with more data")
        more_experiences = trainer.collect_training_data(num_tasks=100)
        trainer.train_with_curriculum(more_experiences, epochs=3)
        
        # Re-validate
        val_accuracy, val_pass_rate = trainer.validate_on_holdout(num_tasks=30)
        trainer.training_history['pass_rates'].append(val_pass_rate)
    
    # Save final model
    trainer.trainer.save_model('models/rl_selector_improved.pt')
    
    # Plot results
    trainer.plot_training_history()
    
    logger.info(f"Training complete! Final validation accuracy: {val_accuracy:.2%}")
    logger.info("Model saved to models/rl_selector_improved.pt")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Training script for RL-based CoD solution selector
Can train from historical evaluation results or online during evaluation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.rl_selector import RLCodeSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLTrainer:
    """Trainer for the RL code selector"""
    
    def __init__(self, 
                 model_save_path: str = "models/rl_selector.pt",
                 feature_dim: int = 26):
        
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.selector = RLCodeSelector(
            feature_dim=feature_dim,
            learning_rate=1e-4,
            epsilon=0.2,  # Start with higher exploration
            model_path=str(self.model_save_path) if self.model_save_path.exists() else None
        )
        
        self.training_history = {
            'rewards': [],
            'losses': [],
            'correct_selections': [],
            'epsilon': []
        }
    
    def train_from_historical_data(self, results_dirs: List[str], 
                                  epochs: int = 10,
                                  batch_size: int = 32):
        """Train from historical evaluation results"""
        logger.info("Loading historical data...")
        
        # Load all task results
        all_task_results = []
        for results_dir in results_dirs:
            results_path = Path(results_dir)
            all_results_file = results_path / "all_results.json"
            
            if all_results_file.exists():
                with open(all_results_file, 'r') as f:
                    results = json.load(f)
                    all_task_results.extend(results)
        
        logger.info(f"Loaded {len(all_task_results)} task results for training")
        
        if not all_task_results:
            logger.warning("No historical data found!")
            return
        
        # Training loop
        for epoch in range(epochs):
            epoch_rewards = []
            epoch_correct = 0
            epoch_total = 0
            losses = []
            
            # Shuffle data
            np.random.shuffle(all_task_results)
            
            for task_result in tqdm(all_task_results, desc=f"Epoch {epoch+1}/{epochs}"):
                # Skip if no solutions
                if 'solutions' not in task_result or len(task_result['solutions']) != 5:
                    continue
                
                # Reconstruct task info
                task = {
                    'task_id': task_result['task_id'],
                    'prompt': task_result.get('prompt', ''),
                    'description': task_result.get('description', '')
                }
                
                solutions = task_result['solutions']
                
                # Select solution using current policy
                selected_idx, info = self.selector.select_solution(
                    task, solutions, training=True
                )
                
                # Compute reward based on actual results
                reward = self.selector.compute_reward(task_result, selected_idx)
                epoch_rewards.append(reward)
                
                # Check if selection was correct (selected a passing solution)
                if task_result['evaluation']['individual_results'][selected_idx]['passed']:
                    epoch_correct += 1
                epoch_total += 1
                
                # Store experience
                self.selector.store_experience(task, solutions, selected_idx, reward)
                
                # Train every few steps
                if len(self.selector.buffer) >= batch_size and epoch_total % 4 == 0:
                    loss = self.selector.train_step(batch_size)
                    if loss is not None:
                        losses.append(loss)
            
            # Update target network periodically
            if epoch % 2 == 0:
                self.selector.update_target_network()
            
            # Decay epsilon
            self.selector.decay_epsilon(decay_rate=0.9)
            
            # Record statistics
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            avg_loss = np.mean(losses) if losses else 0
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            
            self.training_history['rewards'].append(avg_reward)
            self.training_history['losses'].append(avg_loss)
            self.training_history['correct_selections'].append(accuracy)
            self.training_history['epsilon'].append(self.selector.epsilon)
            
            logger.info(f"Epoch {epoch+1}: Avg Reward={avg_reward:.3f}, "
                       f"Accuracy={accuracy:.2%}, Loss={avg_loss:.4f}, "
                       f"Epsilon={self.selector.epsilon:.3f}")
        
        # Save model
        self.selector.save_model(str(self.model_save_path))
        self.plot_training_history()
    
    def online_training_step(self, task: Dict, solutions: List[Dict], 
                           task_result: Dict) -> Dict:
        """Perform one online training step during evaluation"""
        # Select solution
        selected_idx, info = self.selector.select_solution(
            task, solutions, training=True
        )
        
        # Compute reward after evaluation
        reward = self.selector.compute_reward(task_result, selected_idx)
        
        # Store experience
        self.selector.store_experience(task, solutions, selected_idx, reward)
        
        # Train if we have enough data
        loss = None
        if len(self.selector.buffer) >= 32:
            loss = self.selector.train_step(batch_size=32)
        
        # Update statistics
        self.selector.stats['total_selections'] += 1
        if task_result['evaluation']['individual_results'][selected_idx]['passed']:
            self.selector.stats['correct_selections'] += 1
        
        self.selector.stats['episodes'] += 1
        
        # Decay epsilon
        if self.selector.stats['episodes'] % 10 == 0:
            self.selector.decay_epsilon()
        
        # Update target network periodically
        if self.selector.stats['episodes'] % 50 == 0:
            self.selector.update_target_network()
            self.selector.save_model(str(self.model_save_path))
        
        return {
            'selected_idx': selected_idx,
            'reward': reward,
            'loss': loss,
            'info': info
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history['rewards']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Rewards
        axes[0, 0].plot(self.training_history['rewards'])
        axes[0, 0].set_title('Average Reward per Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        
        # Losses
        if self.training_history['losses']:
            axes[0, 1].plot(self.training_history['losses'])
            axes[0, 1].set_title('Average Loss per Epoch')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
        
        # Accuracy
        axes[1, 0].plot(self.training_history['correct_selections'])
        axes[1, 0].set_title('Selection Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        
        # Epsilon
        axes[1, 1].plot(self.training_history['epsilon'])
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plot_path = self.model_save_path.parent / 'training_history.png'
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved to {plot_path}")
    
    def evaluate_selector(self, test_results_dir: str) -> Dict:
        """Evaluate the trained selector on test data"""
        results_path = Path(test_results_dir)
        all_results_file = results_path / "all_results.json"
        
        if not all_results_file.exists():
            logger.error(f"Results file not found: {all_results_file}")
            return {}
        
        with open(all_results_file, 'r') as f:
            test_results = json.load(f)
        
        correct_selections = 0
        total_selections = 0
        rewards = []
        
        # Set to evaluation mode (no exploration)
        original_epsilon = self.selector.epsilon
        self.selector.epsilon = 0.0
        
        for task_result in test_results:
            if 'solutions' not in task_result or len(task_result['solutions']) != 5:
                continue
            
            task = {
                'task_id': task_result['task_id'],
                'prompt': task_result.get('prompt', ''),
                'description': task_result.get('description', '')
            }
            
            solutions = task_result['solutions']
            
            # Select solution
            selected_idx, info = self.selector.select_solution(
                task, solutions, training=False
            )
            
            # Check if correct
            if task_result['evaluation']['individual_results'][selected_idx]['passed']:
                correct_selections += 1
            
            total_selections += 1
            
            # Compute reward
            reward = self.selector.compute_reward(task_result, selected_idx)
            rewards.append(reward)
        
        # Restore epsilon
        self.selector.epsilon = original_epsilon
        
        accuracy = correct_selections / total_selections if total_selections > 0 else 0
        avg_reward = np.mean(rewards) if rewards else 0
        
        results = {
            'accuracy': accuracy,
            'avg_reward': avg_reward,
            'correct_selections': correct_selections,
            'total_selections': total_selections
        }
        
        logger.info(f"Evaluation Results: Accuracy={accuracy:.2%}, "
                   f"Avg Reward={avg_reward:.3f}")
        
        return results

if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        # Train from historical results
        results_dirs = sys.argv[1:]
        trainer = RLTrainer()
        trainer.train_from_historical_data(results_dirs, epochs=20)
    else:
        print("Usage: python rl_trainer.py <results_dir1> [results_dir2] ...")
        print("Example: python rl_trainer.py results/run_20250525_234900") 
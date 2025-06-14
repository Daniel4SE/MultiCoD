#!/usr/bin/env python3
"""
Reinforcement Learning module for selecting the best CoD solution
Uses a contextual bandit approach with neural network policy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from collections import deque
import re

logger = logging.getLogger(__name__)

class CodeFeatureExtractor:
    """Extract features from code solutions for RL state representation"""
    
    def extract_features(self, task: Dict, solution: Dict, all_solutions: List[Dict]) -> np.ndarray:
        """Extract features from a solution in context of all solutions"""
        features = []
        
        # 1. Solution-specific features
        code = solution.get('code', '')
        
        # Code length features
        features.append(len(code))  # Total length
        features.append(code.count('\n'))  # Number of lines
        features.append(len(code.split()))  # Number of tokens
        
        # Code complexity features
        features.append(code.count('def '))  # Number of functions
        features.append(code.count('class '))  # Number of classes
        features.append(code.count('for ') + code.count('while '))  # Loop count
        features.append(code.count('if ') + code.count('elif '))  # Conditional count
        features.append(code.count('try:'))  # Exception handling
        features.append(code.count('import '))  # Import count
        
        # CoD adherence features
        validation = solution.get('validation', {})
        features.append(validation.get('adherence_rate', 0))
        features.append(validation.get('avg_words', 0))
        features.append(validation.get('total_steps', 0))
        
        # Token usage features (normalized)
        token_usage = solution.get('token_usage', {})
        features.append(token_usage.get('total_tokens', 0) / 1000)  # Normalized
        features.append(token_usage.get('output_tokens', 0) / token_usage.get('total_tokens', 1))
        
        # Strategy encoding (one-hot style)
        strategy_name = solution.get('strategy', '')
        strategy_keywords = ['systematic', 'functional', 'error', 'generator', 'comprehension']
        for keyword in strategy_keywords:
            features.append(1.0 if keyword.lower() in strategy_name.lower() else 0.0)
        
        # Temperature feature
        features.append(solution.get('temperature', 0.5))
        
        # 2. Relative features (compared to other solutions)
        all_code_lengths = [len(s.get('code', '')) for s in all_solutions]
        if all_code_lengths:
            # Relative length
            features.append(len(code) / max(all_code_lengths))
            features.append(len(code) / (sum(all_code_lengths) / len(all_code_lengths)))
        else:
            features.extend([0.0, 0.0])
        
        # 3. Task features
        task_prompt = task.get('prompt', '')
        features.append(len(task_prompt))
        features.append(task_prompt.count('def '))
        features.append(task_prompt.count('class '))
        
        # Success probability estimate based on patterns
        success_patterns = ['return', 'yield', 'raise', 'assert']
        pattern_score = sum(1 for p in success_patterns if p in code) / len(success_patterns)
        features.append(pattern_score)
        
        return np.array(features, dtype=np.float32)

class PolicyNetwork(nn.Module):
    """Neural network policy for solution selection"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Shared feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value head (estimates expected reward)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage head (estimates relative advantage of each action)
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 solutions
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, 5, feature_dim] tensor of features for 5 solutions
        Returns:
            q_values: [batch_size, 5] Q-values for each solution
            features: [batch_size, 5, hidden_dim] extracted features
        """
        batch_size = features.shape[0]
        
        # Process all solutions
        flat_features = features.view(-1, self.feature_dim)
        hidden = self.feature_net(flat_features)
        hidden = hidden.view(batch_size, 5, -1)
        
        # Compute value and advantages
        # Use mean pooling across solutions for value estimation
        pooled = hidden.mean(dim=1)
        value = self.value_head(pooled)  # [batch_size, 1]
        
        # Compute advantages for each solution
        advantages = []
        for i in range(5):
            adv = self.advantage_head(hidden[:, i, :])[:, i].unsqueeze(1)
            advantages.append(adv)
        advantages = torch.cat(advantages, dim=1)  # [batch_size, 5]
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values, hidden

class RLCodeSelector:
    """RL-based selector for choosing the best CoD solution"""
    
    def __init__(self, 
                 feature_dim: int = 30,
                 learning_rate: float = 1e-4,
                 epsilon: float = 0.1,
                 gamma: float = 0.99,
                 buffer_size: int = 10000,
                 model_path: Optional[str] = None):
        
        self.feature_extractor = CodeFeatureExtractor()
        self.feature_dim = feature_dim
        self.policy_net = PolicyNetwork(feature_dim)
        self.target_net = PolicyNetwork(feature_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'correct_selections': 0,
            'total_selections': 0,
            'avg_reward': 0.0
        }
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def select_solution(self, task: Dict, solutions: List[Dict], 
                       training: bool = False) -> Tuple[int, Dict]:
        """
        Select the best solution using the RL policy
        
        Returns:
            selected_idx: Index of selected solution
            info: Dictionary with selection info
        """
        # Extract features for all solutions
        features_list = []
        for i, solution in enumerate(solutions):
            features = self.feature_extractor.extract_features(task, solution, solutions)
            features_list.append(features)
        
        features_tensor = torch.FloatTensor(features_list).unsqueeze(0)  # [1, 5, feature_dim]
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            selected_idx = np.random.randint(0, 5)
            action_type = "exploration"
        else:
            with torch.no_grad():
                q_values, _ = self.policy_net(features_tensor)
                selected_idx = q_values[0].argmax().item()
                action_type = "exploitation"
        
        info = {
            'selected_idx': selected_idx,
            'selected_strategy': solutions[selected_idx].get('strategy', 'Unknown'),
            'action_type': action_type,
            'features': features_list[selected_idx].tolist(),
            'q_values': q_values[0].tolist() if 'q_values' in locals() else None
        }
        
        return selected_idx, info
    
    def compute_reward(self, task_result: Dict, selected_idx: int) -> float:
        """
        Compute reward based on evaluation results
        
        Reward design:
        - +1.0 if selected solution passed
        - +0.5 if selected first passing solution
        - +0.2 if selected solution has best adherence
        - -0.5 if selected solution failed when others passed
        - -0.1 otherwise
        """
        individual_results = task_result['evaluation']['individual_results']
        selected_result = individual_results[selected_idx]
        
        reward = 0.0
        
        # Primary reward: did it pass?
        if selected_result['passed']:
            reward += 1.0
            
            # Bonus: was it the first passing solution?
            first_passing_idx = None
            for i, result in enumerate(individual_results):
                if result['passed']:
                    first_passing_idx = i
                    break
            
            if selected_idx == first_passing_idx:
                reward += 0.5
        else:
            # Penalty if we selected a failing solution when others passed
            any_passed = any(r['passed'] for r in individual_results)
            if any_passed:
                reward -= 0.5
            else:
                reward -= 0.1
        
        # Secondary reward: adherence quality
        solutions = task_result['solutions']
        selected_adherence = solutions[selected_idx].get('validation', {}).get('adherence_rate', 0)
        best_adherence = max(s.get('validation', {}).get('adherence_rate', 0) for s in solutions)
        
        if selected_adherence == best_adherence and selected_adherence > 0:
            reward += 0.2
        
        return reward
    
    def store_experience(self, task: Dict, solutions: List[Dict], 
                        selected_idx: int, reward: float, done: bool = True):
        """Store experience in replay buffer"""
        features_list = []
        for solution in solutions:
            features = self.feature_extractor.extract_features(task, solution, solutions)
            features_list.append(features)
        
        experience = {
            'features': np.array(features_list),
            'action': selected_idx,
            'reward': reward,
            'done': done
        }
        
        self.buffer.append(experience)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step"""
        if len(self.buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Prepare batch tensors
        features_batch = torch.FloatTensor([exp['features'] for exp in batch])
        actions_batch = torch.LongTensor([exp['action'] for exp in batch])
        rewards_batch = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # Compute Q-values
        q_values, _ = self.policy_net(features_batch)
        q_values_selected = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
        
        # Compute targets (no next state since episodes are single-step)
        targets = rewards_batch
        
        # Compute loss
        loss = nn.MSELoss()(q_values_selected, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path: str):
        """Save model and statistics"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'epsilon': self.epsilon
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and statistics"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.stats = checkpoint['stats']
        self.epsilon = checkpoint.get('epsilon', 0.1)
        logger.info(f"Model loaded from {path}")
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon) 
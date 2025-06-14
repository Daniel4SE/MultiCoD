#!/usr/bin/env python3
"""Analyze and visualize Multi-CoD evaluation results"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(results_dir: Path):
    """Load evaluation results"""
    metrics_file = results_dir / 'metrics.json'
    results_file = results_dir / 'all_results.json'
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return metrics, results

def plot_pass_at_k(metrics: Dict, output_dir: Path):
    """Plot pass@k rates"""
    k_values = list(range(1, 6))
    pass_rates = [metrics['pass_at_k_rates'][f'pass@{k}'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, pass_rates, 'b-o', linewidth=2, markersize=10)
    
    # Add value labels
    for k, rate in zip(k_values, pass_rates):
        plt.text(k, rate + 0.01, f'{rate:.1%}', ha='center', va='bottom')
    
    plt.xlabel('k (Number of Attempts)', fontsize=12)
    plt.ylabel('Pass@k Rate', fontsize=12)
    plt.title('Chain of Draft Pass@k Results on BigCodeBench', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(pass_rates) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_strategy_performance(metrics: Dict, output_dir: Path):
    """Plot strategy performance comparison"""
    strategies = list(metrics['strategy_performance'].keys())
    performances = list(metrics['strategy_performance'].values())
    
    # Sort by performance
    sorted_pairs = sorted(zip(strategies, performances), key=lambda x: x[1], reverse=True)
    strategies, performances = zip(*sorted_pairs)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(strategies)), performances)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Performance by Strategy Type', fontsize=14)
    plt.xticks(range(len(strategies)), strategies, rotation=45, ha='right')
    
    # Add value labels
    for i, (strategy, perf) in enumerate(zip(strategies, performances)):
        plt.text(i, perf + 0.005, f'{perf:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cod_steps(results: List[Dict], output_dir: Path):
    """Analyze Chain of Draft steps"""
    all_steps = []
    adherence_by_strategy = {}
    
    for result in results:
        for sol in result['solutions']:
            if 'validation' in sol:
                all_steps.append({
                    'strategy': sol['strategy'],
                    'total_steps': sol['validation']['total_steps'],
                    'avg_words': sol['validation']['avg_words'],
                    'adherence_rate': sol['validation']['adherence_rate']
                })
                
                strategy = sol['strategy']
                if strategy not in adherence_by_strategy:
                    adherence_by_strategy[strategy] = []
                adherence_by_strategy[strategy].append(sol['validation']['adherence_rate'])
    
    # Plot adherence by strategy
    plt.figure(figsize=(12, 6))
    
    strategies = list(adherence_by_strategy.keys())
    adherence_data = [adherence_by_strategy[s] for s in strategies]
    
    plt.boxplot(adherence_data, labels=strategies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('5-Word Rule Adherence Rate')
    plt.title('Chain of Draft Step Adherence by Strategy')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cod_adherence.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(metrics: Dict, results: List[Dict], output_dir: Path):
    """Generate detailed text report"""
    report = []
    report.append("="*60)
    report.append("MULTI-COD EVALUATION REPORT")
    report.append("="*60)
    report.append(f"\nTotal Tasks: {metrics['total_tasks']}")
    report.append(f"Successfully Evaluated: {metrics['evaluated_tasks']}")
    report.append(f"\nEvaluation Timestamp: {metrics['timestamp']}")
    
    report.append("\n" + "-"*60)
    report.append("PASS@K RESULTS")
    report.append("-"*60)
    
    improvement_data = []
    for k in range(1, 6):
        count = metrics['pass_at_k_counts'][f'pass@{k}']
        rate = metrics['pass_at_k_rates'][f'pass@{k}']
        report.append(f"Pass@{k}: {count}/{metrics['total_tasks']} ({rate:.2%})")
        
        if k > 1:
            prev_rate = metrics['pass_at_k_rates'][f'pass@{k-1}']
            improvement = ((rate - prev_rate) / prev_rate * 100) if prev_rate > 0 else 0
            improvement_data.append(f"  Improvement over Pass@{k-1}: +{improvement:.1f}%")
    
    report.extend(improvement_data)
    
    report.append("\n" + "-"*60)
    report.append("STRATEGY PERFORMANCE RANKING")
    report.append("-"*60)
    
    sorted_strategies = sorted(
        metrics['strategy_performance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (strategy, performance) in enumerate(sorted_strategies[:10], 1):
        report.append(f"{i}. {strategy}: {performance:.2%}")
    
    report.append("\n" + "-"*60)
    report.append("CHAIN OF DRAFT METRICS")
    report.append("-"*60)
    report.append(f"Overall 5-Word Rule Adherence: {metrics['cod_adherence_rate']:.2%}")
    
    # Save report
    report_text = '\n'.join(report)
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)

def main():
    parser = argparse.ArgumentParser(description='Analyze Multi-CoD results')
    parser.add_argument('results_dir', type=str, help='Results directory path')
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    # Create analysis output directory
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    # Load results
    metrics, results = load_results(results_dir)
    
    # Generate visualizations
    plot_pass_at_k(metrics, analysis_dir)
    plot_strategy_performance(metrics, analysis_dir)
    analyze_cod_steps(results, analysis_dir)
    
    # Generate report
    generate_report(metrics, results, analysis_dir)
    
    print(f"\nAnalysis complete! Results saved to: {analysis_dir}")

if __name__ == "__main__":
    main()
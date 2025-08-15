"""
Visualization Module for Language-Arithmetic Dissociation Study
Provides plotting functions for results analysis and visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import seaborn as sns

class Visualizer:
    def __init__(self):
        """Initialize the visualizer with default styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_accuracy_comparison(self, results: Dict[str, Dict[str, float]], save_path: str = None):
        """
        Plot accuracy comparison between different tasks with and without ablation
        
        Args:
            results: Dictionary with task names as keys and dicts with 'original' and 'ablated' accuracies
            save_path: Optional path to save the plot
        """
        tasks = list(results.keys())
        original_accuracies = [results[task]['original'] for task in tasks]
        ablated_accuracies = [results[task]['ablated'] for task in tasks]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, original_accuracies, width, label='Original Model', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, ablated_accuracies, width, label='Ablated Model', 
                      color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Task Type')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Comparison: Original vs Ablated Model')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_accuracy_drop(self, results: Dict[str, Dict[str, float]], save_path: str = None):
        """
        Plot the accuracy drop (difference) between original and ablated models
        
        Args:
            results: Dictionary with task names as keys and dicts with 'original' and 'ablated' accuracies
            save_path: Optional path to save the plot
        """
        tasks = list(results.keys())
        accuracy_drops = []
        
        for task in tasks:
            original = results[task]['original']
            ablated = results[task]['ablated']
            drop = original - ablated
            accuracy_drops.append(drop)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if drop > 0 else 'green' for drop in accuracy_drops]
        bars = ax.bar(tasks, accuracy_drops, color=colors, alpha=0.7)
        
        ax.set_xlabel('Task Type')
        ax.set_ylabel('Accuracy Drop (Original - Ablated)')
        ax.set_title('Performance Drop After Arithmetic Neuron Ablation')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, drop in zip(bars, accuracy_drops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   f'{drop:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_neuron_activations(self, hidden_states: Dict[str, np.ndarray], 
                               layer_idx: int, save_path: str = None):
        """
        Plot neuron activation patterns for different input types
        
        Args:
            hidden_states: Dictionary with 'language' and 'arithmetic' hidden states
            layer_idx: Layer index for the plot title
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Language activations
        lang_activations = hidden_states['language']
        im1 = ax1.imshow(lang_activations.T, aspect='auto', cmap='Blues')
        ax1.set_title(f'Language Activations - Layer {layer_idx}')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Neuron Index')
        plt.colorbar(im1, ax=ax1)
        
        # Arithmetic activations
        arith_activations = hidden_states['arithmetic']
        im2 = ax2.imshow(arith_activations.T, aspect='auto', cmap='Reds')
        ax2.set_title(f'Arithmetic Activations - Layer {layer_idx}')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Neuron Index')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_activation_differences(self, language_hidden: np.ndarray, arithmetic_hidden: np.ndarray,
                                  layer_idx: int, top_k: int = 20, save_path: str = None):
        """
        Plot the difference in activations between arithmetic and language inputs
        
        Args:
            language_hidden: Hidden states for language inputs
            arithmetic_hidden: Hidden states for arithmetic inputs
            layer_idx: Layer index for the plot title
            top_k: Number of top neurons to highlight
            save_path: Optional path to save the plot
        """
        # Compute mean activations
        lang_mean = np.mean(language_hidden, axis=0)
        arith_mean = np.mean(arithmetic_hidden, axis=0)
        
        # Compute differences
        activation_diff = arith_mean - lang_mean
        
        # Get top-k neurons
        top_indices = np.argsort(activation_diff)[-top_k:]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot all activation differences
        ax1.bar(range(len(activation_diff)), activation_diff, alpha=0.6, color='lightblue')
        ax1.set_title(f'Activation Differences (Arithmetic - Language) - Layer {layer_idx}')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Activation Difference')
        ax1.grid(True, alpha=0.3)
        
        # Highlight top-k neurons
        ax1.bar(top_indices, activation_diff[top_indices], color='red', alpha=0.8, label=f'Top {top_k} neurons')
        ax1.legend()
        
        # Plot top-k neurons separately
        ax2.bar(range(top_k), activation_diff[top_indices], color='red', alpha=0.8)
        ax2.set_title(f'Top {top_k} Arithmetic-Sensitive Neurons')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Activation Difference')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pca_visualization(self, hidden_states: Dict[str, np.ndarray], 
                             layer_idx: int, n_components: int = 2, save_path: str = None):
        """
        Create PCA visualization of hidden states
        
        Args:
            hidden_states: Dictionary with 'language' and 'arithmetic' hidden states
            layer_idx: Layer index for the plot title
            n_components: Number of PCA components
            save_path: Optional path to save the plot
        """
        # Combine all hidden states
        all_states = np.vstack([hidden_states['language'], hidden_states['arithmetic']])
        
        # Create labels
        n_lang = hidden_states['language'].shape[0]
        n_arith = hidden_states['arithmetic'].shape[0]
        labels = ['Language'] * n_lang + ['Arithmetic'] * n_arith
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced_states = pca.fit_transform(all_states)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        lang_points = reduced_states[:n_lang]
        arith_points = reduced_states[n_lang:]
        
        ax.scatter(lang_points[:, 0], lang_points[:, 1], c='blue', label='Language', alpha=0.7, s=50)
        ax.scatter(arith_points[:, 0], arith_points[:, 1], c='red', label='Arithmetic', alpha=0.7, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(f'PCA Visualization of Hidden States - Layer {layer_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return pca, reduced_states
    
    def plot_tsne_visualization(self, hidden_states: Dict[str, np.ndarray], 
                              layer_idx: int, save_path: str = None):
        """
        Create t-SNE visualization of hidden states
        
        Args:
            hidden_states: Dictionary with 'language' and 'arithmetic' hidden states
            layer_idx: Layer index for the plot title
            save_path: Optional path to save the plot
        """
        # Combine all hidden states
        all_states = np.vstack([hidden_states['language'], hidden_states['arithmetic']])
        
        # Create labels
        n_lang = hidden_states['language'].shape[0]
        n_arith = hidden_states['arithmetic'].shape[0]
        labels = ['Language'] * n_lang + ['Arithmetic'] * n_arith
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_states)-1))
        reduced_states = tsne.fit_transform(all_states)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        lang_points = reduced_states[:n_lang]
        arith_points = reduced_states[n_lang:]
        
        ax.scatter(lang_points[:, 0], lang_points[:, 1], c='blue', label='Language', alpha=0.7, s=50)
        ax.scatter(arith_points[:, 0], arith_points[:, 1], c='red', label='Arithmetic', alpha=0.7, s=50)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(f't-SNE Visualization of Hidden States - Layer {layer_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return tsne, reduced_states
    
    def plot_layer_comparison(self, layer_results: Dict[int, Dict[str, float]], save_path: str = None):
        """
        Plot accuracy comparison across different layers
        
        Args:
            layer_results: Dictionary with layer indices as keys and accuracy dicts as values
            save_path: Optional path to save the plot
        """
        layers = sorted(layer_results.keys())
        original_accuracies = [layer_results[layer]['original'] for layer in layers]
        ablated_accuracies = [layer_results[layer]['ablated'] for layer in layers]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_accuracies, width, label='Original Model', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, ablated_accuracies, width, label='Ablated Model', 
                      color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Comparison Across Layers')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_report(self, results: Dict, save_path: str = None):
        """
        Create a comprehensive summary report with all visualizations
        
        Args:
            results: Dictionary containing all experimental results
            save_path: Optional path to save the report
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        tasks = list(results['task_accuracies'].keys())
        original_acc = [results['task_accuracies'][task]['original'] for task in tasks]
        ablated_acc = [results['task_accuracies'][task]['ablated'] for task in tasks]
        
        x = np.arange(len(tasks))
        ax1.bar(x - 0.2, original_acc, 0.4, label='Original', color='skyblue')
        ax1.bar(x + 0.2, ablated_acc, 0.4, label='Ablated', color='lightcoral')
        ax1.set_title('Task Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tasks)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy drop
        ax2 = fig.add_subplot(gs[0, 2])
        drops = [original_acc[i] - ablated_acc[i] for i in range(len(tasks))]
        colors = ['red' if drop > 0 else 'green' for drop in drops]
        ax2.bar(tasks, drops, color=colors, alpha=0.7)
        ax2.set_title('Performance Drop')
        ax2.set_ylabel('Accuracy Drop')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Layer comparison (if available)
        if 'layer_results' in results:
            ax3 = fig.add_subplot(gs[1, :])
            layers = sorted(results['layer_results'].keys())
            layer_original = [results['layer_results'][layer]['original'] for layer in layers]
            layer_ablated = [results['layer_results'][layer]['ablated'] for layer in layers]
            
            ax3.plot(layers, layer_original, 'o-', label='Original', color='blue')
            ax3.plot(layers, layer_ablated, 's-', label='Ablated', color='red')
            ax3.set_title('Performance Across Layers')
            ax3.set_xlabel('Layer Index')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Neuron activation differences (if available)
        if 'activation_differences' in results:
            ax4 = fig.add_subplot(gs[2, :])
            diffs = results['activation_differences']
            ax4.hist(diffs, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_title('Distribution of Activation Differences')
            ax4.set_xlabel('Arithmetic - Language Activation')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Language-Arithmetic Dissociation Study: Summary Report', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 
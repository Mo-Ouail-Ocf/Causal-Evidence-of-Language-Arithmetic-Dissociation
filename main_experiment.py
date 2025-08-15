"""
Main Experiment Script for Language-Arithmetic Dissociation Study
Orchestrates the complete causal evidence experiment
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Any
from tqdm import tqdm

from data_utils.data_generator import DataGenerator
from model.model_wrapper import ModelWrapper
from viz.visualization import Visualizer

class CausalEvidenceExperiment:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 device: str = "auto", top_k_neurons: int = 20):
        """
        Initialize the causal evidence experiment
        
        Args:
            model_name: HuggingFace model name to use
            device: Device to run experiments on
            top_k_neurons: Number of top neurons to ablate
        """
        self.model_name = model_name
        self.device = device
        self.top_k_neurons = top_k_neurons
        
        # Initialize components
        self.data_generator = DataGenerator()
        self.model_wrapper = ModelWrapper(model_name, device)
        self.visualizer = Visualizer()
        
        # Results storage
        self.results = {}
        
        print(f"Initialized experiment with model: {model_name}")
        print(f"Top-k neurons to ablate: {top_k_neurons}")
    
    def step1_identify_arithmetic_neurons(self, layer_idx: int = None) -> Dict[int, List[int]]:
        """
        STEP 1: Identify arithmetic-sensitive neurons
        
        Args:
            layer_idx: Specific layer to analyze (None for all layers)
            
        Returns:
            Dictionary mapping layer indices to arithmetic-sensitive neuron indices
        """
        print("\n" + "="*60)
        print("STEP 1: Identifying Arithmetic-Sensitive Neurons")
        print("="*60)
        
        # Get datasets
        language_texts = self.data_generator.get_language_texts()
        arithmetic_prompts = self.data_generator.get_arithmetic_prompts()
        
        print(f"Language texts: {len(language_texts)} samples")
        print(f"Arithmetic prompts: {len(arithmetic_prompts)} samples")
        
        # Identify arithmetic-sensitive neurons
        # Limit to actual number of layers in the model
        if hasattr(self.model_wrapper.model, 'model') and hasattr(self.model_wrapper.model.model, 'layers'):
            num_layers = len(self.model_wrapper.model.model.layers)
        elif hasattr(self.model_wrapper.model, 'transformer') and hasattr(self.model_wrapper.model.transformer, 'h'):
            num_layers = len(self.model_wrapper.model.transformer.h)
        else:
            num_layers = 32  # fallback
        
        # Only analyze first few layers to avoid memory issues
        max_layers_to_analyze = min(5, num_layers)
        
        arithmetic_neurons = self.model_wrapper.identify_arithmetic_neurons(
            language_texts, arithmetic_prompts, self.top_k_neurons, 
            layer_idx if layer_idx is not None else None,  # None means analyze all layers, but we'll limit in the method
            max_layers=5  # Limit to first 5 layers
        )
        
        # Store results
        self.results['arithmetic_neurons'] = arithmetic_neurons
        
        print(f"\nIdentified arithmetic-sensitive neurons:")
        for layer, neurons in arithmetic_neurons.items():
            print(f"  Layer {layer}: {len(neurons)} neurons")
        
        return arithmetic_neurons
    
    def step2_create_ablation_masks(self, arithmetic_neurons: Dict[int, List[int]]):
        """
        STEP 2: Create ablation masks for identified neurons
        """
        print("\n" + "="*60)
        print("STEP 2: Creating Ablation Masks")
        print("="*60)
        
        self.model_wrapper.create_ablation_masks(arithmetic_neurons)
        print("Ablation masks created successfully!")
    
    def step3_evaluate_performance(self) -> Dict[str, Dict[str, float]]:
        """
        STEP 3: Evaluate model performance with and without ablation
        
        Returns:
            Dictionary with task accuracies for original and ablated models
        """
        print("\n" + "="*60)
        print("STEP 3: Evaluating Performance")
        print("="*60)
        
        # Get all datasets
        datasets = self.data_generator.get_all_datasets()
        expected_answers = {
            "language_texts": self.data_generator.get_expected_answers("language_texts"),
            "arithmetic_prompts": self.data_generator.get_expected_answers("arithmetic_prompts"),
            "word_problems": self.data_generator.get_expected_answers("word_problems")
        }
        
        task_accuracies = {}
        
        # Test original model
        print("\nTesting original model...")
        self.model_wrapper.apply_ablation(enable=False)  # Ensure no ablation
        
        for task_name, texts in datasets.items():
            print(f"  Evaluating {task_name}...")
            accuracy = self.model_wrapper.evaluate_accuracy(texts, expected_answers[task_name])
            task_accuracies[task_name] = {'original': accuracy}
            print(f"    Original accuracy: {accuracy:.3f}")
        
        # Test ablated model
        print("\nTesting ablated model...")
        self.model_wrapper.apply_ablation(enable=True)
        
        for task_name, texts in datasets.items():
            print(f"  Evaluating {task_name}...")
            accuracy = self.model_wrapper.evaluate_accuracy(texts, expected_answers[task_name])
            task_accuracies[task_name]['ablated'] = accuracy
            print(f"    Ablated accuracy: {accuracy:.3f}")
        
        # Store results
        self.results['task_accuracies'] = task_accuracies
        
        return task_accuracies
    
    def step4_layer_analysis(self, max_layers: int = 5) -> Dict[int, Dict[str, float]]:
        """
        STEP 4: Analyze performance across different layers
        
        Args:
            max_layers: Maximum number of layers to analyze
            
        Returns:
            Dictionary with layer-wise performance results
        """
        print("\n" + "="*60)
        print("STEP 4: Layer-wise Analysis")
        print("="*60)
        
        # Get datasets
        arithmetic_prompts = self.data_generator.get_arithmetic_prompts()
        expected_answers = self.data_generator.get_expected_answers("arithmetic_prompts")
        
        layer_results = {}
        
        # Analyze each layer individually
        # Get the actual number of layers in the model
        if hasattr(self.model_wrapper.model, 'model') and hasattr(self.model_wrapper.model.model, 'layers'):
            num_layers = len(self.model_wrapper.model.model.layers)
        elif hasattr(self.model_wrapper.model, 'transformer') and hasattr(self.model_wrapper.model.transformer, 'h'):
            num_layers = len(self.model_wrapper.model.transformer.h)
        else:
            num_layers = 32  # fallback
        
        for layer_idx in range(min(max_layers, num_layers)):
            print(f"\nAnalyzing layer {layer_idx}...")
            
            # Identify neurons for this layer
            arithmetic_neurons = self.model_wrapper.identify_arithmetic_neurons(
                self.data_generator.get_language_texts(),
                arithmetic_prompts,
                self.top_k_neurons,
                layer_idx,
                max_layers=5
            )
            
            if layer_idx not in arithmetic_neurons:
                print(f"  No arithmetic neurons found in layer {layer_idx}, skipping...")
                continue
            
            # Create ablation masks for this layer
            self.model_wrapper.create_ablation_masks(arithmetic_neurons)
            
            # Test original performance
            self.model_wrapper.apply_ablation(enable=False)
            original_accuracy = self.model_wrapper.evaluate_accuracy(arithmetic_prompts, expected_answers)
            
            # Test ablated performance
            self.model_wrapper.apply_ablation(enable=True)
            ablated_accuracy = self.model_wrapper.evaluate_accuracy(arithmetic_prompts, expected_answers)
            
            layer_results[layer_idx] = {
                'original': original_accuracy,
                'ablated': ablated_accuracy
            }
            
            print(f"  Layer {layer_idx}: Original={original_accuracy:.3f}, Ablated={ablated_accuracy:.3f}")
        
        # Store results
        self.results['layer_results'] = layer_results
        
        return layer_results
    
    def step5_visualization_analysis(self, layer_idx: int = 0):
        """
        STEP 5: Create visualizations and analysis
        
        Args:
            layer_idx: Layer to analyze for visualizations
        """
        print("\n" + "="*60)
        print("STEP 5: Visualization and Analysis")
        print("="*60)
        
        # Extract hidden states for visualization
        language_texts = self.data_generator.get_language_texts()
        arithmetic_prompts = self.data_generator.get_arithmetic_prompts()
        
        print("Extracting hidden states for visualization...")
        language_hidden = self.model_wrapper.extract_hidden_states(language_texts, layer_idx)
        arithmetic_hidden = self.model_wrapper.extract_hidden_states(arithmetic_prompts, layer_idx)
        
        if layer_idx in language_hidden and layer_idx in arithmetic_hidden:
            # Convert to numpy for visualization
            lang_states = language_hidden[layer_idx].cpu().numpy()
            arith_states = arithmetic_hidden[layer_idx].cpu().numpy()
            
            hidden_states = {
                'language': lang_states,
                'arithmetic': arith_states
            }
            
            # Create visualizations
            print("Creating visualizations...")
            
            # 1. Neuron activation patterns
            self.visualizer.plot_neuron_activations(hidden_states, layer_idx, 
                                                   save_path=f"neuron_activations_layer_{layer_idx}.png")
            
            # 2. Activation differences
            self.visualizer.plot_activation_differences(lang_states, arith_states, layer_idx, 
                                                       self.top_k_neurons,
                                                       save_path=f"activation_differences_layer_{layer_idx}.png")
            
            # 3. PCA visualization
            self.visualizer.plot_pca_visualization(hidden_states, layer_idx,
                                                  save_path=f"pca_visualization_layer_{layer_idx}.png")
            
            # 4. t-SNE visualization
            self.visualizer.plot_tsne_visualization(hidden_states, layer_idx,
                                                   save_path=f"tsne_visualization_layer_{layer_idx}.png")
            
            # Store activation differences for summary
            lang_mean = np.mean(lang_states, axis=0)
            arith_mean = np.mean(arith_states, axis=0)
            activation_diff = arith_mean - lang_mean
            self.results['activation_differences'] = activation_diff.tolist()
    
    def step6_create_summary_visualizations(self):
        """
        STEP 6: Create summary visualizations
        """
        print("\n" + "="*60)
        print("STEP 6: Summary Visualizations")
        print("="*60)
        
        # Create summary plots
        if 'task_accuracies' in self.results:
            self.visualizer.plot_accuracy_comparison(self.results['task_accuracies'],
                                                    save_path="accuracy_comparison.png")
            self.visualizer.plot_accuracy_drop(self.results['task_accuracies'],
                                              save_path="accuracy_drop.png")
        
        if 'layer_results' in self.results:
            self.visualizer.plot_layer_comparison(self.results['layer_results'],
                                                 save_path="layer_comparison.png")
        
        # Create comprehensive summary report
        self.visualizer.create_summary_report(self.results, save_path="summary_report.png")
    
    def run_complete_experiment(self, max_layers: int = 5, layer_idx: int = 0):
        """
        Run the complete causal evidence experiment
        
        Args:
            max_layers: Maximum number of layers for layer analysis
            layer_idx: Layer index for detailed analysis
        """
        print("🚀 Starting Language-Arithmetic Dissociation Study")
        print("="*80)
        
        try:
            # Step 1: Identify arithmetic-sensitive neurons
            arithmetic_neurons = self.step1_identify_arithmetic_neurons()
            
            # Step 2: Create ablation masks
            self.step2_create_ablation_masks(arithmetic_neurons)
            
            # Step 3: Evaluate performance
            task_accuracies = self.step3_evaluate_performance()
            
            # Step 4: Layer analysis
            layer_results = self.step4_layer_analysis(max_layers)
            
            # Step 5: Visualization analysis
            self.step5_visualization_analysis(layer_idx)
            
            # Step 6: Summary visualizations
            self.step6_create_summary_visualizations()
            
            # Save results
            self.save_results()
            
            print("\n" + "="*80)
            print("✅ Experiment completed successfully!")
            print("="*80)
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ Experiment failed with error: {str(e)}")
            raise
    
    def save_results(self, filename: str = "experiment_results.json"):
        """
        Save experiment results to JSON file
        
        Args:
            filename: Output filename
        """
        # Convert numpy arrays to lists for JSON serialization
        results_copy = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_copy[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        results_copy[key][subkey] = {}
                        for subsubkey, subsubvalue in subvalue.items():
                            if isinstance(subsubvalue, (np.ndarray, np.integer, np.floating)):
                                results_copy[key][subkey][subsubkey] = subsubvalue.tolist() if hasattr(subsubvalue, 'tolist') else float(subsubvalue)
                            else:
                                results_copy[key][subkey][subsubkey] = subsubvalue
                    elif isinstance(subvalue, (np.ndarray, np.integer, np.floating)):
                        results_copy[key][subkey] = subvalue.tolist() if hasattr(subvalue, 'tolist') else float(subvalue)
                    else:
                        results_copy[key][subkey] = subvalue
            elif isinstance(value, (np.ndarray, np.integer, np.floating)):
                results_copy[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
            else:
                results_copy[key] = value
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of the experiment results"""
        print("\n📊 EXPERIMENT SUMMARY")
        print("-" * 40)
        
        if 'task_accuracies' in self.results:
            print("\nTask Performance:")
            for task, accuracies in self.results['task_accuracies'].items():
                original = accuracies['original']
                ablated = accuracies['ablated']
                drop = original - ablated
                print(f"  {task}:")
                print(f"    Original: {original:.3f}")
                print(f"    Ablated:  {ablated:.3f}")
                print(f"    Drop:     {drop:.3f}")
        
        if 'arithmetic_neurons' in self.results:
            print(f"\nArithmetic-Sensitive Neurons:")
            total_neurons = sum(len(neurons) for neurons in self.results['arithmetic_neurons'].values())
            print(f"  Total neurons identified: {total_neurons}")
            for layer, neurons in self.results['arithmetic_neurons'].items():
                print(f"    Layer {layer}: {len(neurons)} neurons")
        
        # Calculate dissociation evidence
        if 'task_accuracies' in self.results:
            arithmetic_drop = 0
            language_drop = 0
            
            if 'arithmetic_prompts' in self.results['task_accuracies']:
                arithmetic_drop = (self.results['task_accuracies']['arithmetic_prompts']['original'] - 
                                 self.results['task_accuracies']['arithmetic_prompts']['ablated'])
            
            if 'language_texts' in self.results['task_accuracies']:
                language_drop = (self.results['task_accuracies']['language_texts']['original'] - 
                               self.results['task_accuracies']['language_texts']['ablated'])
            
            print(f"\nDissociation Evidence:")
            print(f"  Arithmetic performance drop: {arithmetic_drop:.3f}")
            print(f"  Language performance drop: {language_drop:.3f}")
            print(f"  Dissociation strength: {arithmetic_drop - language_drop:.3f}")

def main():
    """Main function to run the experiment"""
    # Configuration
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Using Qwen2.5-7B-Instruct model
    device = "auto"  # Will use CUDA if available, otherwise CPU
    top_k_neurons = 20  # Number of neurons to ablate
    
    # Create and run experiment
    experiment = CausalEvidenceExperiment(
        model_name=model_name,
        device=device,
        top_k_neurons=top_k_neurons
    )
    
    # Run the complete experiment
    experiment.run_complete_experiment(max_layers=5, layer_idx=0)

if __name__ == "__main__":
    main() 
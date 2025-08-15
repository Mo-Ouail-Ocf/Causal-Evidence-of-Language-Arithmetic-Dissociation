"""
Multi-Dataset Experiment for Language-Arithmetic Dissociation Study
Runs experiments for each dataset category separately and creates individual result folders
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Any
from tqdm import tqdm
import shutil

from data_utils.real_dataset_loader import RealDatasetLoader
from model.model_wrapper import ModelWrapper
from viz.visualization import Visualizer

class MultiDatasetExperiment:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 device: str = "auto", top_k_neurons: int = 20):
        """
        Initialize the multi-dataset experiment
        
        Args:
            model_name: HuggingFace model name to use
            device: Device to run experiments on
            top_k_neurons: Number of top neurons to ablate
        """
        self.model_name = model_name
        self.device = device
        self.top_k_neurons = top_k_neurons
        
        # Initialize components
        self.data_loader = RealDatasetLoader()
        self.model_wrapper = ModelWrapper(model_name, device)
        self.visualizer = Visualizer()
        
        # Results storage
        self.all_results = {}
        
        print(f"Initialized multi-dataset experiment with model: {model_name}")
        print(f"Top-k neurons to ablate: {top_k_neurons}")
    
    def run_single_dataset_experiment(self, dataset_name: str, output_dir: str, questions: List[str], answers: List[str]):
        """
        Run experiment for a single dataset
        
        Args:
            dataset_name: Name of the dataset (GSM8K, SQuAD, etc.)
            output_dir: Directory to save results
            questions: List of questions for this dataset
            answers: List of expected answers
        """
        print(f"\n{'='*60}")
        print(f"Running experiment for dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Dataset size: {len(questions)} samples")
        
        # Step 1: Extract hidden states
        print(f"\nStep 1: Extracting hidden states for {dataset_name}...")
        hidden_states = self.model_wrapper.extract_hidden_states(questions, layer_idx=0)
        
        if 0 not in hidden_states:
            print(f"❌ Failed to extract hidden states for {dataset_name}")
            return None
        
        print(f"✅ Hidden states extracted: {hidden_states[0].shape}")
        
        # Step 2: Identify arithmetic-sensitive neurons (compare with SQuAD as baseline)
        print(f"\nStep 2: Identifying arithmetic-sensitive neurons...")
        
        # Get SQuAD dataset for comparison (language baseline)
        all_datasets = self.data_loader.get_all_datasets(max_samples=len(questions))
        if "SQUAD" in all_datasets:
            squad_questions, _ = all_datasets["SQUAD"]
        else:
            # Fallback: use first few questions as language baseline
            squad_questions = questions[:5]
        
        arithmetic_neurons = self.model_wrapper.identify_arithmetic_neurons(
            squad_questions, questions, self.top_k_neurons, layer_idx=0, max_layers=1
        )
        
        if not arithmetic_neurons or 0 not in arithmetic_neurons:
            print(f"❌ No arithmetic neurons found for {dataset_name}")
            return None
        
        print(f"✅ Found {len(arithmetic_neurons[0])} arithmetic-sensitive neurons")
        
        # Step 3: Create ablation masks
        print(f"\nStep 3: Creating ablation masks...")
        self.model_wrapper.create_ablation_masks(arithmetic_neurons)
        print("✅ Ablation masks created")
        
        # Step 4: Evaluate performance
        print(f"\nStep 4: Evaluating performance...")
        
        # Test original model
        self.model_wrapper.apply_ablation(enable=False)
        original_accuracy = self.model_wrapper.evaluate_accuracy(questions, answers)
        print(f"  Original accuracy: {original_accuracy:.3f}")
        
        # Test ablated model
        self.model_wrapper.apply_ablation(enable=True)
        ablated_accuracy = self.model_wrapper.evaluate_accuracy(questions, answers)
        print(f"  Ablated accuracy: {ablated_accuracy:.3f}")
        
        # Calculate performance drop
        performance_drop = original_accuracy - ablated_accuracy
        print(f"  Performance drop: {performance_drop:.3f}")
        
        # Step 5: Create visualizations
        print(f"\nStep 5: Creating visualizations...")
        
        # Prepare results
        results = {
            "dataset_name": dataset_name,
            "original_accuracy": original_accuracy,
            "ablated_accuracy": ablated_accuracy,
            "performance_drop": performance_drop,
            "arithmetic_neurons": arithmetic_neurons,
            "hidden_states_shape": hidden_states[0].shape,
            "sample_questions": questions[:3],  # Save first 3 questions as examples
            "sample_answers": answers[:3]
        }
        
        # Create activation difference visualization
        if 0 in hidden_states:
            squad_hidden = self.model_wrapper.extract_hidden_states(squad_questions, layer_idx=0)
            if 0 in squad_hidden:
                squad_states = squad_hidden[0].cpu().numpy()
                dataset_states = hidden_states[0].cpu().numpy()
                
                self.visualizer.plot_activation_differences(
                    squad_states, dataset_states, 0, self.top_k_neurons,
                    save_path=os.path.join(output_dir, f"{dataset_name}_activation_differences.png")
                )
                
                # Create PCA visualization
                hidden_states_dict = {
                    'language': squad_states,
                    'dataset': dataset_states
                }
                self.visualizer.plot_pca_visualization(
                    hidden_states_dict, 0,
                    save_path=os.path.join(output_dir, f"{dataset_name}_pca_visualization.png")
                )
        
        # Save results
        results_file = os.path.join(output_dir, f"{dataset_name}_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    results_copy[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (np.ndarray, list)):
                            results_copy[key][subkey] = subvalue.tolist() if hasattr(subvalue, 'tolist') else subvalue
                        else:
                            results_copy[key][subkey] = subvalue
                elif isinstance(value, (np.ndarray, list)):
                    results_copy[key] = value.tolist() if hasattr(value, 'tolist') else value
                else:
                    results_copy[key] = value
            
            json.dump(results_copy, f, indent=2)
        
        print(f"✅ Results saved to {results_file}")
        
        return results
    
    def run_all_experiments(self, max_samples: int = 20):
        """
        Run experiments for all datasets
        
        Args:
            max_samples: Maximum number of samples per dataset
        """
        print("🚀 Starting Multi-Dataset Language-Arithmetic Dissociation Study")
        print("="*80)
        
        # Load all datasets
        print(f"\n📚 Loading real datasets (max {max_samples} samples each)...")
        try:
            all_datasets = self.data_loader.get_all_datasets(max_samples=max_samples)
        except Exception as e:
            print(f"❌ Failed to load datasets: {e}")
            return
        
        # Define dataset characteristics
        datasets_info = {
            "GSM8K": {"description": "Complex word problems", "has_math": True},
            "SQUAD": {"description": "Reading comprehension", "has_math": False},
            "MATH_QA": {"description": "Math question answering", "has_math": True},
            "MAWPS": {"description": "Math word problems", "has_math": True},
            "COMMONSENSE_QA": {"description": "Commonsense reasoning", "has_math": False}
        }
        
        # Create main results directory
        main_results_dir = f"results_{self.model_name.split('/')[-1]}"
        os.makedirs(main_results_dir, exist_ok=True)
        
        # Run experiments for each available dataset
        for dataset_name, (questions, answers) in all_datasets.items():
            if dataset_name in datasets_info:
                info = datasets_info[dataset_name]
                print(f"\n📊 Dataset: {dataset_name} - {info['description']}")
                
                # Create dataset-specific directory
                dataset_dir = os.path.join(main_results_dir, dataset_name)
                
                # Run experiment
                results = self.run_single_dataset_experiment(dataset_name, dataset_dir, questions, answers)
                
                if results:
                    self.all_results[dataset_name] = results
                    print(f"✅ Experiment completed for {dataset_name}")
                else:
                    print(f"❌ Experiment failed for {dataset_name}")
            else:
                print(f"\n📊 Dataset: {dataset_name} (no info available)")
                
                # Create dataset-specific directory
                dataset_dir = os.path.join(main_results_dir, dataset_name)
                
                # Run experiment
                results = self.run_single_dataset_experiment(dataset_name, dataset_dir, questions, answers)
                
                if results:
                    self.all_results[dataset_name] = results
                    print(f"✅ Experiment completed for {dataset_name}")
                else:
                    print(f"❌ Experiment failed for {dataset_name}")
        
        # Create summary report
        self.create_summary_report(main_results_dir)
        
        print(f"\n{'='*80}")
        print("✅ All experiments completed!")
        print(f"{'='*80}")
        
        # Print summary
        self.print_summary()
    
    def create_summary_report(self, main_results_dir: str):
        """
        Create a summary report comparing all datasets
        """
        print(f"\n📊 Creating summary report...")
        
        # Prepare summary data
        summary_data = {}
        for dataset_name, results in self.all_results.items():
            summary_data[dataset_name] = {
                "original": results["original_accuracy"],
                "ablated": results["ablated_accuracy"],
                "performance_drop": results["performance_drop"]
            }
        
        # Create summary visualizations
        if len(summary_data) > 1:
            self.visualizer.plot_accuracy_comparison(
                summary_data, 
                save_path=os.path.join(main_results_dir, "summary_accuracy_comparison.png")
            )
            
            self.visualizer.plot_accuracy_drop(
                summary_data,
                save_path=os.path.join(main_results_dir, "summary_accuracy_drop.png")
            )
        
        # Save summary results
        summary_file = os.path.join(main_results_dir, "summary_results.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✅ Summary report saved to {main_results_dir}")
    
    def print_summary(self):
        """Print a summary of all experiment results"""
        print("\n📊 EXPERIMENT SUMMARY")
        print("-" * 60)
        
        print(f"{'Dataset':<15} {'Original':<10} {'Ablated':<10} {'Drop':<10} {'Dissociation':<12}")
        print("-" * 60)
        
        for dataset_name, results in self.all_results.items():
            original = results["original_accuracy"]
            ablated = results["ablated_accuracy"]
            drop = results["performance_drop"]
            
            # Calculate dissociation strength (compared to SQuAD baseline)
            if "SQUAD" in self.all_results:
                squad_drop = self.all_results["SQUAD"]["performance_drop"]
                dissociation = drop - squad_drop
            else:
                dissociation = drop
            
            print(f"{dataset_name:<15} {original:<10.3f} {ablated:<10.3f} {drop:<10.3f} {dissociation:<12.3f}")
        
        print("-" * 60)
        
        # Find strongest dissociation
        if len(self.all_results) > 1:
            strongest_dissociation = max(self.all_results.items(), 
                                       key=lambda x: x[1]["performance_drop"])
            print(f"\n🎯 Strongest dissociation: {strongest_dissociation[0]} "
                  f"(drop: {strongest_dissociation[1]['performance_drop']:.3f})")
            
            # Find most arithmetic-sensitive dataset
            math_datasets = ["GSM8K", "MATH_QA", "MAWPS"]
            math_results = {k: v for k, v in self.all_results.items() if k in math_datasets}
            
            if math_results:
                most_math_sensitive = max(math_results.items(), 
                                        key=lambda x: x[1]["performance_drop"])
                print(f"🧮 Most arithmetic-sensitive: {most_math_sensitive[0]} "
                      f"(drop: {most_math_sensitive[1]['performance_drop']:.3f})")

def main():
    """Main function to run the multi-dataset experiment"""
    # Configuration
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "auto"
    top_k_neurons = 20
    max_samples = 20  # Number of samples per dataset
    
    # Create and run experiment
    experiment = MultiDatasetExperiment(
        model_name=model_name,
        device=device,
        top_k_neurons=top_k_neurons
    )
    
    # Run all experiments
    experiment.run_all_experiments(max_samples=max_samples)

if __name__ == "__main__":
    main() 
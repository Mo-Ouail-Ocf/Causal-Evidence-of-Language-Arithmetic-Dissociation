"""
Example Usage Script for Language-Arithmetic Dissociation Study
Demonstrates how to use the system step by step
"""

import torch
import numpy as np
from data_utils.data_generator import DataGenerator
from model.model_wrapper import ModelWrapper
from viz.visualization import Visualizer

def example_step_by_step():
    """
    Example demonstrating step-by-step usage of the system
    """
    print("🔬 Language-Arithmetic Dissociation Study - Example Usage")
    print("="*70)
    
    # Step 1: Initialize components
    print("\n📦 Step 1: Initializing Components")
    print("-" * 40)
    
    # Create data generator
    data_gen = DataGenerator()
    print("✅ Data generator created")
    
    # Create model wrapper (using Qwen2.5-7B-Instruct for demonstration)
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Using Qwen2.5-7B-Instruct model
    model_wrapper = ModelWrapper(model_name, device="auto")
    print(f"✅ Model wrapper created with {model_name}")
    
    # Create visualizer
    visualizer = Visualizer()
    print("✅ Visualizer created")
    
    # Step 2: Prepare datasets
    print("\n📚 Step 2: Preparing Datasets")
    print("-" * 40)
    
    # Get datasets
    language_texts = data_gen.get_language_texts()[:3]  # Use first 3 for demo
    arithmetic_prompts = data_gen.get_arithmetic_prompts()[:3]
    word_problems = data_gen.get_word_problems()[:3]
    
    print(f"Language texts: {len(language_texts)} samples")
    print(f"Arithmetic prompts: {len(arithmetic_prompts)} samples")
    print(f"Word problems: {len(word_problems)} samples")
    
    # Show sample data
    print(f"\nSample language: '{language_texts[0]}'")
    print(f"Sample arithmetic: '{arithmetic_prompts[0]}'")
    print(f"Sample word problem: '{word_problems[0]}'")
    
    # Step 3: Extract hidden states
    print("\n🧠 Step 3: Extracting Hidden States")
    print("-" * 40)
    
    print("Extracting hidden states for language texts...")
    language_hidden = model_wrapper.extract_hidden_states(language_texts, layer_idx=0)
    
    print("Extracting hidden states for arithmetic prompts...")
    arithmetic_hidden = model_wrapper.extract_hidden_states(arithmetic_prompts, layer_idx=0)
    
    if 0 in language_hidden and 0 in arithmetic_hidden:
        print(f"✅ Hidden states extracted successfully")
        print(f"Language hidden states shape: {language_hidden[0].shape}")
        print(f"Arithmetic hidden states shape: {arithmetic_hidden[0].shape}")
    else:
        print("❌ Hidden state extraction failed")
        return
    
    # Step 4: Identify arithmetic-sensitive neurons
    print("\n🔍 Step 4: Identifying Arithmetic-Sensitive Neurons")
    print("-" * 40)
    
    arithmetic_neurons = model_wrapper.identify_arithmetic_neurons(
        language_texts, arithmetic_prompts, top_k=5, layer_idx=0
    )
    
    if arithmetic_neurons and 0 in arithmetic_neurons:
        neurons = arithmetic_neurons[0]
        print(f"✅ Found {len(neurons)} arithmetic-sensitive neurons in layer 0")
        print(f"Neuron indices: {neurons}")
    else:
        print("❌ No arithmetic neurons found")
        return
    
    # Step 5: Create ablation masks
    print("\n🎭 Step 5: Creating Ablation Masks")
    print("-" * 40)
    
    model_wrapper.create_ablation_masks(arithmetic_neurons)
    print("✅ Ablation masks created")
    
    # Step 6: Test original model performance
    print("\n📊 Step 6: Testing Original Model Performance")
    print("-" * 40)
    
    # Get expected answers
    arith_answers = data_gen.get_expected_answers("arithmetic_prompts")[:3]
    word_answers = data_gen.get_expected_answers("word_problems")[:3]
    
    # Ensure no ablation is applied
    model_wrapper.apply_ablation(enable=False)
    
    # Test arithmetic performance
    original_arith_acc = model_wrapper.evaluate_accuracy(arithmetic_prompts, arith_answers)
    print(f"Original arithmetic accuracy: {original_arith_acc:.3f}")
    
    # Test word problem performance
    original_word_acc = model_wrapper.evaluate_accuracy(word_problems, word_answers)
    print(f"Original word problem accuracy: {original_word_acc:.3f}")
    
    # Step 7: Test ablated model performance
    print("\n🔧 Step 7: Testing Ablated Model Performance")
    print("-" * 40)
    
    # Apply ablation
    model_wrapper.apply_ablation(enable=True)
    
    # Test arithmetic performance
    ablated_arith_acc = model_wrapper.evaluate_accuracy(arithmetic_prompts, arith_answers)
    print(f"Ablated arithmetic accuracy: {ablated_arith_acc:.3f}")
    
    # Test word problem performance
    ablated_word_acc = model_wrapper.evaluate_accuracy(word_problems, word_answers)
    print(f"Ablated word problem accuracy: {ablated_word_acc:.3f}")
    
    # Step 8: Analyze results
    print("\n📈 Step 8: Analyzing Results")
    print("-" * 40)
    
    # Calculate performance drops
    arith_drop = original_arith_acc - ablated_arith_acc
    word_drop = original_word_acc - ablated_word_acc
    dissociation_strength = arith_drop - word_drop
    
    print(f"Arithmetic performance drop: {arith_drop:.3f}")
    print(f"Word problem performance drop: {word_drop:.3f}")
    print(f"Dissociation strength: {dissociation_strength:.3f}")
    
    # Interpret results
    if dissociation_strength > 0.1:
        print("🎯 Strong evidence of dissociation!")
    elif dissociation_strength > 0.05:
        print("📊 Moderate evidence of dissociation")
    else:
        print("⚠️ Weak or no evidence of dissociation")
    
    # Step 9: Create visualizations
    print("\n📊 Step 9: Creating Visualizations")
    print("-" * 40)
    
    # Prepare results for visualization
    results = {
        "arithmetic_prompts": {
            "original": original_arith_acc,
            "ablated": ablated_arith_acc
        },
        "word_problems": {
            "original": original_word_acc,
            "ablated": ablated_word_acc
        }
    }
    
    # Create accuracy comparison plot
    visualizer.plot_accuracy_comparison(results, save_path="example_accuracy_comparison.png")
    print("✅ Accuracy comparison plot saved")
    
    # Create accuracy drop plot
    visualizer.plot_accuracy_drop(results, save_path="example_accuracy_drop.png")
    print("✅ Accuracy drop plot saved")
    
    # Create activation difference visualization
    if 0 in language_hidden and 0 in arithmetic_hidden:
        lang_states = language_hidden[0].cpu().numpy()
        arith_states = arithmetic_hidden[0].cpu().numpy()
        
        visualizer.plot_activation_differences(
            lang_states, arith_states, 0, top_k=5,
            save_path="example_activation_differences.png"
        )
        print("✅ Activation differences plot saved")
    
    print("\n🎉 Example completed successfully!")
    print("Check the generated PNG files for visualizations.")

def example_custom_model():
    """
    Example showing how to use a different model
    """
    print("\n🔄 Example: Using a Different Model")
    print("="*50)
    
    # You can easily switch to other models
    models_to_try = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",  # Smaller alternative
        # "mistralai/Mistral-7B-Instruct-v0.2",  # Large model, requires more resources
        # "meta-llama/Llama-2-7b-chat-hf",       # Requires access
    ]
    
    for model_name in models_to_try:
        print(f"\nTrying model: {model_name}")
        try:
            wrapper = ModelWrapper(model_name, device="cpu")
            print(f"✅ Successfully loaded {model_name}")
            
            # Test basic functionality
            response = wrapper.generate_response("Hello", max_length=10)
            print(f"Test response: '{response}'")
            break
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            continue

def example_custom_dataset():
    """
    Example showing how to create custom datasets
    """
    print("\n📝 Example: Custom Dataset Creation")
    print("="*50)
    
    # Create custom datasets
    custom_language = [
        "The weather is sunny today.",
        "I love reading books.",
        "Music makes me happy."
    ]
    
    custom_arithmetic = [
        "2+3=?",
        "4+5=?",
        "6+7=?"
    ]
    
    custom_answers = ["5", "9", "13"]
    
    print("Custom language texts:")
    for text in custom_language:
        print(f"  - {text}")
    
    print("\nCustom arithmetic prompts:")
    for prompt, answer in zip(custom_arithmetic, custom_answers):
        print(f"  - {prompt} -> {answer}")
    
    # You can use these with the model wrapper
    print("\nYou can use these custom datasets with the model wrapper:")
    print("wrapper.identify_arithmetic_neurons(custom_language, custom_arithmetic)")

def main():
    """
    Main function to run all examples
    """
    print("🚀 Language-Arithmetic Dissociation Study - Examples")
    print("="*70)
    
    try:
        # Run main example
        example_step_by_step()
        
        # Run additional examples
        example_custom_model()
        example_custom_dataset()
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)
        
        print("\n📚 What you learned:")
        print("1. How to initialize the system components")
        print("2. How to extract hidden states from the model")
        print("3. How to identify arithmetic-sensitive neurons")
        print("4. How to apply neuron ablation")
        print("5. How to evaluate performance differences")
        print("6. How to create visualizations")
        print("7. How to use different models and datasets")
        
        print("\n💡 Next steps:")
        print("1. Run 'python main_experiment.py' for the full experiment")
        print("2. Modify parameters in main_experiment.py for different configurations")
        print("3. Add your own datasets or models")
        print("4. Explore the generated visualizations and results")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("This might be due to model download issues or resource constraints.")
        print("Try running 'python test_experiment.py' for a simpler test.")

if __name__ == "__main__":
    main() 
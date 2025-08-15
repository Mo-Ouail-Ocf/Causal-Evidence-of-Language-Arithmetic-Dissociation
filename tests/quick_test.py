"""
Quick Test Script for Language-Arithmetic Dissociation Study
Uses smaller models for faster testing and validation
"""

import torch
import numpy as np
from data_utils.data_generator import DataGenerator
from model.model_wrapper import ModelWrapper
from viz.visualization import Visualizer

def quick_test_with_small_model():
    """
    Quick test using a smaller model for faster validation
    """
    print("🚀 Quick Test with Small Model")
    print("="*50)
    
    # Use a smaller model for quick testing
    model_name = "microsoft/DialoGPT-small"  # ~117M parameters
    # Alternative small models:
    # "microsoft/DialoGPT-medium" (~345M parameters)
    # "distilgpt2" (~82M parameters)
    
    try:
        # Initialize components
        print("📦 Initializing components...")
        data_gen = DataGenerator()
        model_wrapper = ModelWrapper(model_name, device="cpu")  # Use CPU for small model
        visualizer = Visualizer()
        
        # Use very small datasets for quick testing
        language_texts = data_gen.get_language_texts()[:3]
        arithmetic_prompts = data_gen.get_arithmetic_prompts()[:3]
        word_problems = data_gen.get_word_problems()[:3]
        
        print(f"Using {len(language_texts)} samples each for quick test")
        
        # Step 1: Extract hidden states
        print("\n🧠 Step 1: Extracting hidden states...")
        language_hidden = model_wrapper.extract_hidden_states(language_texts, layer_idx=0)
        arithmetic_hidden = model_wrapper.extract_hidden_states(arithmetic_prompts, layer_idx=0)
        
        if 0 not in language_hidden or 0 not in arithmetic_hidden:
            print("❌ Hidden state extraction failed")
            return
        
        print("✅ Hidden states extracted successfully")
        
        # Step 2: Identify arithmetic neurons
        print("\n🔍 Step 2: Identifying arithmetic neurons...")
        arithmetic_neurons = model_wrapper.identify_arithmetic_neurons(
            language_texts, arithmetic_prompts, top_k=3, layer_idx=0
        )
        
        if not arithmetic_neurons or 0 not in arithmetic_neurons:
            print("❌ No arithmetic neurons found")
            return
        
        print(f"✅ Found {len(arithmetic_neurons[0])} arithmetic neurons")
        
        # Step 3: Create ablation masks
        print("\n🎭 Step 3: Creating ablation masks...")
        model_wrapper.create_ablation_masks(arithmetic_neurons)
        print("✅ Ablation masks created")
        
        # Step 4: Test performance
        print("\n📊 Step 4: Testing performance...")
        
        # Get expected answers
        arith_answers = data_gen.get_expected_answers("arithmetic_prompts")[:3]
        word_answers = data_gen.get_expected_answers("word_problems")[:3]
        
        # Test original model
        model_wrapper.apply_ablation(enable=False)
        original_arith_acc = model_wrapper.evaluate_accuracy(arithmetic_prompts, arith_answers)
        original_word_acc = model_wrapper.evaluate_accuracy(word_problems, word_answers)
        
        # Test ablated model
        model_wrapper.apply_ablation(enable=True)
        ablated_arith_acc = model_wrapper.evaluate_accuracy(arithmetic_prompts, arith_answers)
        ablated_word_acc = model_wrapper.evaluate_accuracy(word_problems, word_answers)
        
        # Report results
        print("\n📈 Quick Test Results:")
        print(f"Arithmetic Prompts:")
        print(f"  Original: {original_arith_acc:.3f}")
        print(f"  Ablated:  {ablated_arith_acc:.3f}")
        print(f"  Drop:     {original_arith_acc - ablated_arith_acc:.3f}")
        
        print(f"Word Problems:")
        print(f"  Original: {original_word_acc:.3f}")
        print(f"  Ablated:  {ablated_word_acc:.3f}")
        print(f"  Drop:     {original_word_acc - ablated_word_acc:.3f}")
        
        # Calculate dissociation strength
        dissociation = (original_arith_acc - ablated_arith_acc) - (original_word_acc - ablated_word_acc)
        print(f"\nDissociation Strength: {dissociation:.3f}")
        
        if dissociation > 0.05:
            print("🎯 Evidence of dissociation detected!")
        else:
            print("⚠️ Weak or no dissociation evidence (expected for small model)")
        
        # Create simple visualization
        results = {
            "arithmetic_prompts": {"original": original_arith_acc, "ablated": ablated_arith_acc},
            "word_problems": {"original": original_word_acc, "ablated": ablated_word_acc}
        }
        
        visualizer.plot_accuracy_comparison(results, save_path="quick_test_results.png")
        print("✅ Quick test visualization saved")
        
        print("\n🎉 Quick test completed successfully!")
        print("The system is working correctly. You can now run the full experiment with Qwen2.5-7B-Instruct.")
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        print("This might be due to model download issues or resource constraints.")

def main():
    """Main function for quick test"""
    print("🧪 Language-Arithmetic Dissociation Study - Quick Test")
    print("="*60)
    print("This test uses a small model for fast validation.")
    print("For the full experiment with Qwen2.5-7B-Instruct, run: python main_experiment.py")
    print("="*60)
    
    quick_test_with_small_model()
    
    print("\n" + "="*60)
    print("💡 Next Steps:")
    print("1. If quick test passed, run: python main_experiment.py")
    print("2. For examples, run: python example_usage.py")
    print("3. Check generated visualizations")
    print("="*60)

if __name__ == "__main__":
    main() 
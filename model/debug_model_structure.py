"""
Debug script to examine model structure
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_model_structure(model_name="microsoft/DialoGPT-small"):
    """Debug the model structure to understand layer names"""
    
    print(f"Debugging model structure for: {model_name}")
    print("="*50)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {dir(model)}")
    
    # Check transformer structure
    if hasattr(model, 'transformer'):
        print(f"\nTransformer attributes: {dir(model.transformer)}")
        
        if hasattr(model.transformer, 'h'):
            print(f"Number of layers: {len(model.transformer.h)}")
            
            # Examine first layer
            first_layer = model.transformer.h[0]
            print(f"\nFirst layer attributes: {dir(first_layer)}")
            
            if hasattr(first_layer, 'mlp'):
                mlp = first_layer.mlp
                print(f"MLP attributes: {dir(mlp)}")
                
                # Check all attributes of MLP
                for attr in dir(mlp):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(mlp, attr)
                            if hasattr(value, 'out_features'):
                                print(f"  {attr}: out_features = {value.out_features}")
                            elif hasattr(value, 'weight'):
                                print(f"  {attr}: weight shape = {value.weight.shape}")
                            else:
                                print(f"  {attr}: {type(value)}")
                        except:
                            print(f"  {attr}: <error accessing>")
            else:
                print("No MLP found in first layer")
        else:
            print("No 'h' attribute found in transformer")
    else:
        print("No transformer attribute found")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    debug_model_structure() 
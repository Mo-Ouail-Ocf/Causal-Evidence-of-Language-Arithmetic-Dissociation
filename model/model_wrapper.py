"""
Model Wrapper for Language-Arithmetic Dissociation Study
Handles model loading, hidden state extraction, and neuron ablation
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

class ModelWrapper:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "auto"):
        """
        Initialize the model wrapper
        
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix padding token issue for models that don't have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Store ablation masks
        self.ablation_masks = {}
        self.original_weights = {}
        
        print("Model loaded successfully!")
    
    def extract_hidden_states(self, texts: List[str], layer_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract hidden states from the model for given texts
        
        Args:
            texts: List of input texts
            layer_idx: Specific layer to extract from (None for all layers)
            
        Returns:
            Dictionary containing hidden states for each layer
        """
        hidden_states = {} # [layer index , torch.Tensor(N,H)]
        # For each layer , for each text sample : mean of hidden representations
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting hidden states"):
                # Tokenize input
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                # 'input_ids' : [1,L] , 'attention_mask': shape (1, L)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs with output_hidden_states=True
                outputs = self.model(**inputs, output_hidden_states=True) # CausalLMOutputWithPast


                # outputs.hidden_states : tuple of length num_layers+1 ,
                #  index 0 : embed 
                # index i > 0 : output after tranf block 
                
                # Extract hidden states from each layer
                for layer, hidden_state in enumerate(outputs.hidden_states):
                    # hidden_state : (batch_size, seq_len, hidden_dim) [1,L,H]
                    if layer_idx is not None and layer != layer_idx:
                        continue
                    
                    # Average across tokens (excluding padding)
                    attention_mask = inputs['attention_mask'] 
                    masked_hidden = hidden_state * attention_mask.unsqueeze(-1) # [1,L,H]
                    avg_hidden = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) # [1,H] per-sample, per-layer mean hidden vector.
                    
                    if layer not in hidden_states:
                        hidden_states[layer] = []
                    # append per layer per sample hidden mean embedding vector
                    hidden_states[layer].append(avg_hidden.squeeze(0))
        
        # Stack all hidden states for each layer
        for layer in hidden_states:
            # [N,H]
            hidden_states[layer] = torch.stack(hidden_states[layer])
        
        return hidden_states 
    
    def identify_arithmetic_neurons(self, language_texts: List[str], arithmetic_texts: List[str], 
                                  top_k: int = 20, layer_idx: Optional[int] = None, max_layers: int = 5) -> Dict[int, List[int]]:
        """
        Identify neurons that respond strongly to arithmetic vs language inputs
        
        Args:
            language_texts: List of language texts
            arithmetic_texts: List of arithmetic texts
            top_k: Number of top neurons to select
            layer_idx: Specific layer to analyze (None for all layers)
            
        Returns:
            Dictionary mapping layer indices to lists of arithmetic-sensitive neuron indices
        """
        print("Extracting hidden states for language texts...")
        language_hidden = self.extract_hidden_states(language_texts, layer_idx)
        
        print("Extracting hidden states for arithmetic texts...")
        arithmetic_hidden = self.extract_hidden_states(arithmetic_texts, layer_idx)
        
        arithmetic_neurons = {}
        
        for layer in language_hidden:
            if layer_idx is not None and layer != layer_idx:
                continue
            
            # Limit the number of layers to analyze to avoid memory issues
            if layer_idx is None and layer >= max_layers:
                continue
                
            # Compute mean activations
            lang_mean = language_hidden[layer].mean(dim=0)  # Average across samples
            arith_mean = arithmetic_hidden[layer].mean(dim=0)
            
            # Find neurons more active for arithmetic
            activation_diff = arith_mean - lang_mean
            
            # Get top-k neurons
            top_indices = torch.topk(activation_diff, k=min(top_k, len(activation_diff))).indices
            
            arithmetic_neurons[layer] = top_indices.cpu().numpy().tolist()
            
            print(f"Layer {layer}: Selected {len(arithmetic_neurons[layer])} arithmetic-sensitive neurons")
        
        return arithmetic_neurons
    
    def create_ablation_masks(self, arithmetic_neurons: Dict[int, List[int]]):
        """
        Create ablation masks for the identified arithmetic-sensitive neurons
        
        Args:
            arithmetic_neurons: Dictionary mapping layer indices to neuron indices
        """
        self.ablation_masks = {}
        
        for layer_idx, neuron_indices in arithmetic_neurons.items():
            # Get the MLP layer (handling different model architectures)
            mlp_layer = None
            output_size = None
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # For models like Mistral, LLaMA
                mlp_layer = self.model.model.layers[layer_idx].mlp
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # For GPT-style models
                mlp_layer = self.model.transformer.h[layer_idx].mlp
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                # For DialoGPT-style models
                mlp_layer = self.model.transformer.h[layer_idx].mlp
            else:
                print(f"Warning: Could not find MLP layer for layer {layer_idx}")
                continue
            
            # Create mask for the MLP output (handling different architectures)
            if hasattr(mlp_layer, 'down_proj'):
                # For models with up_proj -> act_fn -> down_proj structure
                output_size = mlp_layer.down_proj.out_features
            elif hasattr(mlp_layer, 'c_proj'):
                # For GPT-style models (including DialoGPT)
                # Conv1D layers have weight shape [out_features, in_features]
                output_size = mlp_layer.c_proj.weight.shape[0]
            else:
                print(f"Warning: Could not determine output size for layer {layer_idx}")
                continue

            # output_Size : total neurons in MLP output
            
            # Create mask (1 for keep, 0 for ablate)
            mask = torch.ones(output_size, device=self.device)
            mask[neuron_indices] = 0.0
            
            self.ablation_masks[layer_idx] = mask
            
            print(f"Created ablation mask for layer {layer_idx}: {sum(mask == 0)} neurons to ablate")
    
    def apply_ablation(self, enable: bool = True):
        """
        Apply or remove neuron ablation
        
        Args:
            enable: Whether to enable ablation (True) or remove it (False)
        """
        if not self.ablation_masks:
            print("No ablation masks found. Run create_ablation_masks first.")
            return
        
        for layer_idx, mask in self.ablation_masks.items():
            # Get the MLP layer
            mlp_layer = None
            output_layer = None
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                mlp_layer = self.model.model.layers[layer_idx].mlp
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                mlp_layer = self.model.transformer.h[layer_idx].mlp
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                # For DialoGPT-style models
                mlp_layer = self.model.transformer.h[layer_idx].mlp
            else:
                continue
            
            # Get the output projection layer
            if hasattr(mlp_layer, 'down_proj'):
                output_layer = mlp_layer.down_proj
            elif hasattr(mlp_layer, 'c_proj'):
                output_layer = mlp_layer.c_proj
            else:
                continue
            
            if enable:
                # Store original weights if not already stored
                if layer_idx not in self.original_weights:
                    self.original_weights[layer_idx] = output_layer.weight.clone() # output_size, in_features

                # Apply ablation by zeroing out weights for ablated neurons
                ablated_weights = output_layer.weight.clone()
                
                # Handle different layer types
                if hasattr(output_layer, 'out_features'):
                    # Linear layers: [out_features, in_features]
                    # For Qwen models: down_proj has shape [3584, 18944], out_features=3584
                    # We want to zero out rows (output features) where mask == 0
                    ablated_weights[mask == 0, :] = 0.0
                else:
                    # Conv1D layers: [out_features, in_features] but indexed differently
                    # For GPT-style models
                    ablated_weights[mask == 0, :] = 0.0
                
                output_layer.weight.data = ablated_weights
                
                print(f"Applied ablation to layer {layer_idx}")
            else:
                # Restore original weights
                if layer_idx in self.original_weights:
                    output_layer.weight.data = self.original_weights[layer_idx]
                    print(f"Removed ablation from layer {layer_idx}")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate a response for a given prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated response text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()  # Remove the input prompt
    
    def evaluate_accuracy(self, texts: List[str], expected_answers: List[str]) -> float:
        """
        Evaluate accuracy on arithmetic tasks
        
        Args:
            texts: List of input texts
            expected_answers: List of expected answers
            
        Returns:
            Accuracy score
        """
        correct = 0
        total = 0
        
        for text, expected in zip(texts, expected_answers):
            if not expected:  # Skip language texts
                continue
                
            response = self.generate_response(text)
            
            # Simple answer extraction (look for numbers in response)
            import re
            numbers = re.findall(r'\d+', response)
            
            if numbers and numbers[0] == expected:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0 
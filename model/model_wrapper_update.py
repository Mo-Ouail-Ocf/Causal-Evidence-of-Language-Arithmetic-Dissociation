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
import re

class ModelWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
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
            torch_dtype=torch.float16,
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

    def extract_hidden_states(
        self,
        texts: List[str],
        layer_idx: Optional[int] = None,
        batch_size: int = 8,
        agg_mode: str = "mean",  # 'mean', 'max', or 'last_token'
    ) -> Dict[int, torch.Tensor]:
        """
        Extract aggregated hidden states for each text from the model.

        Args:
            texts: List of input strings.
            layer_idx: Optional[int], specific layer to extract from.
            batch_size: Processing batch size.
            agg_mode: Aggregation mode for token representations:
                - 'mean': Mean over all valid tokens (default).
                - 'max': Max over all valid tokens.
                - 'last_token': Last non-padded token's hidden state.

        Returns:
            Dict[layer_idx, torch.Tensor]: Mapping layer index → (N, H) tensor.
        """
        from math import ceil
        hidden_states: Dict[int, list] = {}

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
                batch_texts = texts[start:start + batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs, output_hidden_states=True)

                for layer, hidden_state in enumerate(outputs.hidden_states):
                    if layer_idx is not None and layer != layer_idx:
                        continue

                    attention_mask = inputs['attention_mask'].to(hidden_state.dtype)  # (B, L)
                    masked_hidden = hidden_state * attention_mask.unsqueeze(-1)       # (B, L, H)

                    if agg_mode == "mean":
                        token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
                        agg_hidden = masked_hidden.sum(dim=1) / token_counts                 # (B, H)

                    elif agg_mode == "max":
                        # Set masked positions to very low value before max
                        masked_hidden = masked_hidden + (attention_mask.unsqueeze(-1) - 1) * 1e9
                        agg_hidden, _ = masked_hidden.max(dim=1)  # (B, H)

                    elif agg_mode == "last_token":
                        last_indices = attention_mask.sum(dim=1) - 1  # (B,)
                        agg_hidden = hidden_state[torch.arange(hidden_state.size(0)), last_indices]  # (B, H)

                    else:
                        raise ValueError(f"Invalid agg_mode: {agg_mode}")

                    if layer not in hidden_states:
                        hidden_states[layer] = []
                    hidden_states[layer].append(agg_hidden)

        # Stack along batch dimension
        for layer in hidden_states:
            hidden_states[layer] = torch.cat(hidden_states[layer], dim=0)  # (N, H)

        return hidden_states

    def identify_arithmetic_neurons(
        self,
        language_texts: List[str],
        arithmetic_texts: List[str],
        top_k: int = 20,
        layer_idx: Optional[int] = None,
        max_layers: int = 5,
        use_cohens_d: bool = True
    ) -> Dict[int, List[int]]:
        """
        Identify neurons that respond strongly to arithmetic vs. language inputs.

        Args:
            language_texts: List of plain language texts
            arithmetic_texts: List of arithmetic/mathematical texts
            top_k: Number of neurons to select per layer
            layer_idx: Specific layer to analyze (None → analyze up to `max_layers`)
            max_layers: Max number of layers to process if `layer_idx` is None
            use_cohens_d: If True, uses Cohen's d for effect size

        Returns:
            Dictionary mapping layer index → list of top arithmetic-sensitive neuron indices
        """
        print("Extracting hidden states for language texts...")
        language_hidden = self.extract_hidden_states(language_texts, layer_idx)

        print("Extracting hidden states for arithmetic texts...")
        arithmetic_hidden = self.extract_hidden_states(arithmetic_texts, layer_idx)

        arithmetic_neurons = {}

        for layer in sorted(language_hidden.keys()):
            # Layer filtering
            if layer_idx is not None and layer != layer_idx:
                continue
            if layer_idx is None and layer >= max_layers:
                continue

            lang_acts = language_hidden[layer]   # shape [n_lang, hidden_dim]
            arith_acts = arithmetic_hidden[layer] # shape [n_arith, hidden_dim]

            # Compute metric per neuron
            if use_cohens_d:
                # Cohen's d: (mean1 - mean2) / pooled std
                lang_mean, arith_mean = lang_acts.mean(0), arith_acts.mean(0)
                lang_std, arith_std = lang_acts.std(0, unbiased=False), arith_acts.std(0, unbiased=False)
                pooled_std = torch.sqrt((lang_std**2 + arith_std**2) / 2)
                pooled_std = torch.where(pooled_std == 0, torch.ones_like(pooled_std), pooled_std)
                effect_size = (arith_mean - lang_mean) / pooled_std
                metric = effect_size
            else:
                metric = arith_acts.mean(0) - lang_acts.mean(0)

            # Select top-k neurons
            top_indices = torch.topk(metric, k=min(top_k, metric.numel())).indices
            arithmetic_neurons[layer] = top_indices.cpu().numpy().tolist()

            print(f"Layer {layer}: Selected {len(top_indices)} arithmetic-sensitive neurons")

        return arithmetic_neurons


    def create_ablation_masks(self, arithmetic_neurons: Dict[int, List[int]]) -> Dict[int, torch.Tensor]:
        """
        Create binary ablation masks for selected neurons in each layer's MLP output.
        
        Shapes:
            arithmetic_neurons:
                - Dict[layer_idx, List[neuron_idx]]
                - neuron_idx in [0, output_size-1]
            Output:
                - Dict[layer_idx, mask_tensor]
                - mask_tensor: shape (output_size,), dtype=torch.float32
                Values: 1.0 (keep neuron), 0.0 (ablate neuron)

        Args:
            arithmetic_neurons: Mapping from layer index to list of neuron indices to ablate.

        Returns:
            ablation_masks: Same as self.ablation_masks (also stored in instance)
        """
        self.ablation_masks: Dict[int, torch.Tensor] = {}

        for layer_idx, neuron_indices in arithmetic_neurons.items():
            # Get MLP layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                mlp_layer = self.model.model.layers[layer_idx].mlp
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                mlp_layer = self.model.transformer.h[layer_idx].mlp
            else:
                raise ValueError(f"Could not find MLP layer for layer {layer_idx}")

            # Determine output size
            if hasattr(mlp_layer, 'down_proj'):
                output_size = mlp_layer.down_proj.out_features
            elif hasattr(mlp_layer, 'c_proj'):
                output_size = mlp_layer.c_proj.weight.shape[0]
            else:
                raise ValueError(f"Could not determine output size for layer {layer_idx}")

            # Create mask
            mask = torch.ones(output_size, device=self.device, dtype=torch.float32)
            mask[neuron_indices] = 0.0

            self.ablation_masks[layer_idx] = mask

        return self.ablation_masks

    def apply_ablation(self, enable: bool = True):
        """
        Apply or remove neuron ablation.

        Shapes:
            self.ablation_masks[layer_idx]: (output_size,) float32 mask [1.0 keep, 0.0 ablate]
            output_layer.weight: (output_size, in_features)

        Args:
            enable (bool): If True, zero out selected neurons; if False, restore original weights.
        """
        if not getattr(self, "ablation_masks", None):
            raise ValueError("No ablation masks found. Run create_ablation_masks first.")

        for layer_idx, mask in self.ablation_masks.items():
            # 1. Locate MLP layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                mlp_layer = self.model.model.layers[layer_idx].mlp
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                mlp_layer = self.model.transformer.h[layer_idx].mlp
            else:
                print(f"[WARN] Could not find MLP for layer {layer_idx}")
                continue

            # 2. Get projection layer
            if hasattr(mlp_layer, 'down_proj'):
                output_layer = mlp_layer.down_proj
            elif hasattr(mlp_layer, 'c_proj'):
                output_layer = mlp_layer.c_proj
            else:
                print(f"[WARN] Could not find projection layer for {layer_idx}")
                continue

            # 3. Shape check
            output_size = output_layer.weight.shape[0]
            if mask.shape[0] != output_size:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match output size {output_size} in layer {layer_idx}"
                )

            # 4. Apply or restore
            if enable:
                if not hasattr(self, "original_weights"):
                    self.original_weights = {}
                if layer_idx not in self.original_weights:
                    self.original_weights[layer_idx] = output_layer.weight.clone()

                ablated_weights = output_layer.weight.clone()
                ablated_weights[mask == 0] = 0.0
                output_layer.weight.data = ablated_weights
                print(f"[INFO] Applied ablation to layer {layer_idx}")
            else:
                if hasattr(self, "original_weights") and layer_idx in self.original_weights:
                    output_layer.weight.data = self.original_weights[layer_idx]
                    print(f"[INFO] Restored original weights for layer {layer_idx}")

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate a model response for a given text prompt.

        Args:
            prompt (str): Input text.
            max_new_tokens (int): Max tokens to generate.

        Returns:
            str: Generated continuation text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]  # seq_len

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Slice only the generated part
        generated_ids = outputs[0, input_length:]  # shape [gen_len]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


    def evaluate_accuracy(self, texts: List[str], expected_answers: List[Union[str, int, float]]) -> float:
        """
        Evaluate model accuracy on arithmetic tasks.
        
        Improvements over original:
            - Handles negative numbers and decimals
            - Compares numerically (avoids string mismatch like "05" vs "5")
            - Matches *any* number in output, not just the first
            - Ignores irrelevant output text
            - Skips empty expected answers
        
        Args:
            texts: List of input prompts
            expected_answers: List of expected numeric answers (as str/int/float)
        
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        correct = 0
        total = 0

        for text, expected in zip(texts, expected_answers):
            if expected is None or expected == "":
                continue  # skip non-arithmetic tasks

            # Ensure numeric type for expected
            try:
                expected_num = float(expected)
            except ValueError:
                continue  # skip if expected isn't numeric at all

            response = self.generate_response(text)

            # Extract numbers: handles integers, decimals, negatives
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response)

            # Compare any number in the output
            match_found = any(float(num) == expected_num for num in numbers)

            if match_found:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

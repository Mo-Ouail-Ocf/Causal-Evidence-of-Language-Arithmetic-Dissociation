# Causal Evidence of Language–Arithmetic Dissociation in LLMs

This project implements a comprehensive study to provide causal evidence for the dissociation between language and arithmetic processing in Large Language Models (LLMs). The study uses neuron ablation techniques to demonstrate that specific neurons are causally involved in arithmetic processing but not in general language processing.

## 🧠 Research Goal

The main objective is to:
1. **Load a language model** (e.g., Mistral, LLaMA, or Qwen)
2. **Identify neurons** that respond strongly to arithmetic vs. language inputs
3. **Ablate arithmetic-sensitive neurons** and observe performance drops on arithmetic tasks
4. **Evaluate dissociation causally** by comparing performance on different task types
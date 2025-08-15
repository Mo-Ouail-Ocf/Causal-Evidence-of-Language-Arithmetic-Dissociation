"""
Real Dataset Loader for Language-Arithmetic Dissociation Study
Downloads and loads real datasets from HuggingFace
"""

import torch
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Any, Tuple
import re
import random

class RealDatasetLoader:
    def __init__(self):
        """Initialize the real dataset loader"""
        self.datasets = {}
        self.cache = {}
        
    def load_gsm8k_dataset(self, split: str = "train", max_samples: int = 20) -> Tuple[List[str], List[str]]:
        """
        Load GSM8K dataset from HuggingFace
        
        Args:
            split: Dataset split ('train', 'test', 'validation')
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (questions, answers)
        """
        print(f"Loading GSM8K dataset ({split} split)...")
        
        try:
            # Load GSM8K dataset
            dataset = load_dataset("gsm8k", "main", split=split)
            
            # Extract questions and answers
            questions = []
            answers = []
            
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                question = example['question']
                answer = example['answer']
                
                # Extract the final numerical answer from the answer text
                # GSM8K answers are in format: "Let's solve this step by step: ... \n#### 42"
                final_answer = self._extract_final_answer(answer)
                
                if final_answer:
                    questions.append(question)
                    answers.append(final_answer)
            
            if not questions:
                raise ValueError("No valid questions found in GSM8K dataset")
            
            print(f"✅ Loaded {len(questions)} GSM8K samples")
            return questions, answers
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GSM8K dataset: {e}")
    
    def load_math_dataset(self, dataset_name: str = "math_qa", split: str = "train", max_samples: int = 20) -> Tuple[List[str], List[str]]:
        """
        Load math dataset (MathQA, MAWPS, etc.)
        
        Args:
            dataset_name: Name of the math dataset
            split: Dataset split
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (questions, answers)
        """
        print(f"Loading {dataset_name} dataset ({split} split)...")
        
        try:
            # Try different math datasets
            if dataset_name == "math_qa":
                dataset = load_dataset("math_qa", split=split)
                questions = []
                answers = []
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    question = example['Problem']
                    answer = example['correct']
                    
                    questions.append(question)
                    answers.append(str(answer))
            
            elif dataset_name == "mawps":
                dataset = load_dataset("mawps", split=split)
                questions = []
                answers = []
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    question = example['sQuestion']
                    answer = example['lSolutions'][0] if example['lSolutions'] else "0"
                    
                    questions.append(question)
                    answers.append(str(answer))
            
            else:
                # Try generic math dataset
                dataset = load_dataset(dataset_name, split=split)
                questions = []
                answers = []
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    # Try to find question and answer fields
                    if 'question' in example:
                        question = example['question']
                    elif 'text' in example:
                        question = example['text']
                    else:
                        continue
                    
                    if 'answer' in example:
                        answer = str(example['answer'])
                    elif 'label' in example:
                        answer = str(example['label'])
                    else:
                        answer = "0"
                    
                    questions.append(question)
                    answers.append(answer)
            
            if not questions:
                raise ValueError(f"No valid questions found in {dataset_name} dataset")
            
            print(f"✅ Loaded {len(questions)} {dataset_name} samples")
            return questions, answers
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {dataset_name} dataset: {e}")
    
    def load_language_dataset(self, dataset_name: str = "squad", split: str = "train", max_samples: int = 20) -> Tuple[List[str], List[str]]:
        """
        Load language dataset (SQuAD, etc.)
        
        Args:
            dataset_name: Name of the language dataset
            split: Dataset split
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (questions, answers)
        """
        print(f"Loading {dataset_name} dataset ({split} split)...")
        
        try:
            if dataset_name == "squad":
                dataset = load_dataset("squad", split=split)
                questions = []
                answers = []
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    question = example['question']
                    answer = example['answers']['text'][0] if example['answers']['text'] else ""
                    
                    questions.append(question)
                    answers.append(answer)
            
            elif dataset_name == "commonsense_qa":
                dataset = load_dataset("commonsense_qa", split=split)
                questions = []
                answers = []
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    question = example['question']
                    answer = example['choices']['text'][example['answerKey']]
                    
                    questions.append(question)
                    answers.append(answer)
            
            else:
                # Try generic dataset
                dataset = load_dataset(dataset_name, split=split)
                questions = []
                answers = []
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    if 'question' in example:
                        question = example['question']
                    elif 'text' in example:
                        question = example['text']
                    else:
                        continue
                    
                    if 'answer' in example:
                        answer = str(example['answer'])
                    elif 'label' in example:
                        answer = str(example['label'])
                    else:
                        answer = ""
                    
                    questions.append(question)
                    answers.append(answer)
            
            if not questions:
                raise ValueError(f"No valid questions found in {dataset_name} dataset")
            
            print(f"✅ Loaded {len(questions)} {dataset_name} samples")
            return questions, answers
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {dataset_name} dataset: {e}")
    
    def _extract_final_answer(self, answer_text: str) -> str:
        """
        Extract the final numerical answer from GSM8K answer text
        
        Args:
            answer_text: The full answer text from GSM8K
            
        Returns:
            The final numerical answer as string
        """
        # GSM8K format: "... \n#### 42"
        lines = answer_text.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('####'):
                return line.replace('####', '').strip()
        
        # Fallback: try to find any number at the end
        numbers = re.findall(r'\d+', answer_text)
        if numbers:
            return numbers[-1]
        
        return "0"
    
    def get_all_datasets(self, max_samples: int = 20) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Load all datasets - only real datasets, no fallbacks
        
        Args:
            max_samples: Maximum samples per dataset
            
        Returns:
            Dictionary of dataset name -> (questions, answers)
        """
        print("🔄 Loading real datasets from HuggingFace...")
        
        datasets = {}
        
        # Load real datasets - fail if any can't be loaded
        try:
            # GSM8K - Complex word problems
            print("\n📊 Loading GSM8K (complex word problems)...")
            gsm8k_questions, gsm8k_answers = self.load_gsm8k_dataset(max_samples=max_samples)
            datasets["GSM8K"] = (gsm8k_questions, gsm8k_answers)
            
        except Exception as e:
            raise RuntimeError(f"GSM8K dataset is required but failed to load: {e}")
        
        try:
            # SQuAD - Language questions
            print("\n📊 Loading SQuAD (reading comprehension)...")
            squad_questions, squad_answers = self.load_language_dataset("squad", max_samples=max_samples)
            datasets["SQUAD"] = (squad_questions, squad_answers)
            
        except Exception as e:
            raise RuntimeError(f"SQuAD dataset is required but failed to load: {e}")
        
        try:
            # MathQA - Math questions
            print("\n📊 Loading MathQA (math question answering)...")
            math_questions, math_answers = self.load_math_dataset("math_qa", max_samples=max_samples)
            datasets["MATH_QA"] = (math_questions, math_answers)
            
        except Exception as e:
            print(f"⚠️ MathQA failed to load: {e}")
            print("⚠️ MathQA dataset is deprecated, skipping...")
        
        try:
            # MAWPS - Math word problems
            print("\n📊 Loading MAWPS (math word problems)...")
            mawps_questions, mawps_answers = self.load_math_dataset("mawps", max_samples=max_samples)
            datasets["MAWPS"] = (mawps_questions, mawps_answers)
            
        except Exception as e:
            print(f"⚠️ MAWPS failed to load: {e}")
            print("⚠️ MAWPS dataset may not be available, skipping...")
        
        try:
            # CommonsenseQA - Language reasoning
            print("\n📊 Loading CommonsenseQA (commonsense reasoning)...")
            csqa_questions, csqa_answers = self.load_language_dataset("commonsense_qa", max_samples=max_samples)
            datasets["COMMONSENSE_QA"] = (csqa_questions, csqa_answers)
            
        except Exception as e:
            print(f"⚠️ CommonsenseQA failed to load: {e}")
            print("⚠️ CommonsenseQA dataset may not be available, skipping...")
        
        # Verify we have at least the essential datasets
        if not datasets:
            raise RuntimeError("No datasets could be loaded!")
        
        if "GSM8K" not in datasets:
            raise RuntimeError("GSM8K dataset is essential but could not be loaded!")
        
        if "SQUAD" not in datasets:
            raise RuntimeError("SQuAD dataset is essential but could not be loaded!")
        
        print(f"\n✅ Successfully loaded {len(datasets)} real datasets:")
        for name, (questions, answers) in datasets.items():
            print(f"  {name}: {len(questions)} samples")
        
        return datasets

if __name__ == "__main__":
    # Test the real dataset loader
    loader = RealDatasetLoader()
    
    try:
        datasets = loader.get_all_datasets(max_samples=5)
        
        print("\nSample data:")
        for name, (questions, answers) in datasets.items():
            print(f"\n{name}:")
            print(f"  Q: {questions[0]}")
            print(f"  A: {answers[0]}")
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("This ensures we only use real datasets, not synthetic fallbacks.") 
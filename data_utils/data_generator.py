"""
Data Generator for Language-Arithmetic Dissociation Study
Creates three datasets: language texts, arithmetic prompts, and word problems
"""

import random
from typing import List, Dict, Any

class DataGenerator:
    def __init__(self):
        # Language texts - pure language prompts
        self.language_texts = [
            "The sky is blue and beautiful.",
            "Birds sing sweetly in the morning.",
            "The ocean waves crash against the shore.",
            "Trees sway gently in the breeze.",
            "Mountains rise majestically in the distance.",
            "The sun sets behind the horizon.",
            "Stars twinkle in the night sky.",
            "Rain falls softly on the ground.",
            "Flowers bloom in the spring garden.",
            "The wind whispers through the leaves.",
            "Clouds drift across the blue sky.",
            "The moon shines brightly tonight.",
            "Rivers flow towards the sea.",
            "Butterflies dance in the meadow.",
            "The forest is full of life.",
            "Snow covers the mountain peaks.",
            "The desert stretches for miles.",
            "Crickets chirp in the evening.",
            "The lake reflects the sky above.",
            "Grass grows green in the field."
        ]
        
        # Arithmetic prompts - basic math operations
        self.arithmetic_prompts = [
            "1+2=?",
            "3+4=?",
            "5+6=?",
            "7+8=?",
            "9+10=?",
            "2+3=?",
            "4+5=?",
            "6+7=?",
            "8+9=?",
            "10+11=?",
            "1+3=?",
            "2+4=?",
            "3+5=?",
            "4+6=?",
            "5+7=?",
            "6+8=?",
            "7+9=?",
            "8+10=?",
            "9+11=?",
            "10+12=?"
        ]
        
        # Word problems - small arithmetic word problems
        self.word_problems = [
            "Tom has 3 apples, gets 2 more. How many?",
            "Mary has 5 books, buys 3 more. How many?",
            "John has 4 candies, eats 1. How many?",
            "Sarah has 6 pencils, gives 2 away. How many?",
            "Mike has 7 toys, gets 3 more. How many?",
            "Lisa has 2 cookies, bakes 4 more. How many?",
            "David has 8 marbles, loses 3. How many?",
            "Emma has 9 stickers, gives 2 to friends. How many?",
            "Alex has 1 dollar, earns 5 more. How many?",
            "Sophie has 10 balloons, pops 2. How many?",
            "James has 3 cars, buys 4 more. How many?",
            "Anna has 5 flowers, picks 2 more. How many?",
            "Robert has 6 coins, finds 3 more. How many?",
            "Grace has 7 beads, uses 2 for necklace. How many?",
            "Henry has 8 blocks, adds 1 more. How many?",
            "Zoe has 4 dolls, gets 3 more. How many?",
            "Lucas has 9 cards, gives 4 away. How many?",
            "Mia has 2 pets, adopts 3 more. How many?",
            "Noah has 5 friends, meets 2 new ones. How many?",
            "Ava has 6 pictures, draws 2 more. How many?"
        ]
    
    def get_language_texts(self) -> List[str]:
        """Return language texts dataset"""
        return self.language_texts.copy()
    
    def get_arithmetic_prompts(self) -> List[str]:
        """Return arithmetic prompts dataset"""
        return self.arithmetic_prompts.copy()
    
    def get_word_problems(self) -> List[str]:
        """Return word problems dataset"""
        return self.word_problems.copy()
    
    def get_all_datasets(self) -> Dict[str, List[str]]:
        """Return all three datasets"""
        return {
            "language_texts": self.get_language_texts(),
            "arithmetic_prompts": self.get_arithmetic_prompts(),
            "word_problems": self.get_word_problems()
        }
    
    def get_expected_answers(self, dataset_name: str) -> List[str]:
        """Get expected answers for arithmetic and word problems"""
        if dataset_name == "arithmetic_prompts":
            return ["3", "7", "11", "15", "19", "5", "9", "13", "17", "21", 
                   "4", "6", "8", "10", "12", "14", "16", "18", "20", "22"]
        elif dataset_name == "word_problems":
            return ["5", "8", "3", "4", "10", "6", "5", "7", "6", "8",
                   "7", "7", "9", "5", "9", "7", "5", "5", "7", "8"]
        else:
            return [""] * len(self.language_texts)  # No specific answers for language texts

if __name__ == "__main__":
    # Test the data generator
    generator = DataGenerator()
    datasets = generator.get_all_datasets()
    
    print("Dataset sizes:")
    for name, data in datasets.items():
        print(f"{name}: {len(data)} samples")
    
    print("\nSample language text:", datasets["language_texts"][0])
    print("Sample arithmetic prompt:", datasets["arithmetic_prompts"][0])
    print("Sample word problem:", datasets["word_problems"][0]) 
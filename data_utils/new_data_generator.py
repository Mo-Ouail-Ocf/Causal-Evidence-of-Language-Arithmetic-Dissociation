"""
New Data Generator for Language-Arithmetic Dissociation Study
Creates datasets with specific categories: LANG, LANGNUM, EQ, EQSP, LANGNUMEQ, GSM8K
"""

import random
from typing import List, Dict, Any

class NewDataGenerator:
    def __init__(self):
        # LANG: Pure language questions
        self.lang_data = [
            "How do you view the nature of the world we live in?",
            "What is the meaning of life?",
            "How do you define happiness?",
            "What makes a good friend?",
            "How do you approach difficult decisions?",
            "What is the importance of education?",
            "How do you handle stress?",
            "What is your philosophy on success?",
            "How do you maintain work-life balance?",
            "What is the role of technology in society?",
            "How do you define creativity?",
            "What is the value of diversity?",
            "How do you approach learning new things?",
            "What is the importance of communication?",
            "How do you deal with change?",
            "What is your perspective on leadership?",
            "How do you handle criticism?",
            "What is the significance of family?",
            "How do you define personal growth?",
            "What is your approach to problem-solving?"
        ]
        
        # LANGNUM: Language questions involving numbers
        self.langnum_data = [
            "What is the atomic number of hydrogen?",
            "How many planets are in our solar system?",
            "What year was the Declaration of Independence signed?",
            "How many bones are in the human body?",
            "What is the population of New York City?",
            "How many chromosomes do humans have?",
            "What is the speed of light in miles per second?",
            "How many states are in the United States?",
            "What is the boiling point of water in Celsius?",
            "How many days are in a leap year?",
            "What is the atomic number of carbon?",
            "How many continents are there?",
            "What is the distance from Earth to the Moon?",
            "How many letters are in the English alphabet?",
            "What is the average human heart rate?",
            "How many sides does a hexagon have?",
            "What is the atomic number of oxygen?",
            "How many players are on a basketball team?",
            "What is the freezing point of water in Fahrenheit?",
            "How many hours are in a day?"
        ]
        
        # EQ: Pure arithmetic equations
        self.eq_data = [
            "3*1-2=?",
            "5+7=?",
            "12/3=?",
            "4^2=?",
            "8-3=?",
            "6*4=?",
            "15/5=?",
            "2+9=?",
            "10-4=?",
            "3*6=?",
            "20/4=?",
            "7+5=?",
            "9-2=?",
            "4*3=?",
            "16/2=?",
            "1+8=?",
            "11-6=?",
            "5*2=?",
            "18/3=?",
            "2+7=?"
        ]
        
        # EQSP: Arithmetic equations in spoken form
        self.eqsp_data = [
            "three times one minus two equals?",
            "five plus seven equals?",
            "twelve divided by three equals?",
            "four squared equals?",
            "eight minus three equals?",
            "six times four equals?",
            "fifteen divided by five equals?",
            "two plus nine equals?",
            "ten minus four equals?",
            "three times six equals?",
            "twenty divided by four equals?",
            "seven plus five equals?",
            "nine minus two equals?",
            "four times three equals?",
            "sixteen divided by two equals?",
            "one plus eight equals?",
            "eleven minus six equals?",
            "five times two equals?",
            "eighteen divided by three equals?",
            "two plus seven equals?"
        ]
        
        # LANGNUMEQ: Language questions with numbers that require arithmetic
        self.langnumeq_data = [
            "{the number of fingers displayed in a peace sign}-1=?",
            "{the number of days in a week}+3=?",
            "{the number of sides in a triangle}*2=?",
            "{the number of letters in 'hello'}-1=?",
            "{the number of months in a year}/2=?",
            "{the number of hours in a day}-8=?",
            "{the number of players in a soccer team}+2=?",
            "{the number of colors in a rainbow}-3=?",
            "{the number of continents}+1=?",
            "{the number of planets in solar system}-1=?",
            "{the number of sides in a square}*3=?",
            "{the number of letters in 'world'}+2=?",
            "{the number of seasons in a year}*2=?",
            "{the number of fingers on one hand}-1=?",
            "{the number of days in a weekend}+5=?",
            "{the number of vowels in 'education'}+1=?",
            "{the number of sides in a pentagon}-2=?",
            "{the number of letters in 'computer'}/2=?",
            "{the number of hours in half a day}+6=?",
            "{the number of primary colors}+2=?"
        ]
        
        # GSM8K: Complex word problems
        self.gsm8k_data = [
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "Janet's dogs ate 42 dog biscuits. If each dog ate 6 biscuits, how many dogs does Janet have?",
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "Sam bought a dozen boxes of cookies, with 10 cookies in each box. If he ate 20 cookies, how many cookies does he have left?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        ]
    
    def get_lang_data(self) -> List[str]:
        """Return LANG dataset"""
        return self.lang_data.copy()
    
    def get_langnum_data(self) -> List[str]:
        """Return LANGNUM dataset"""
        return self.langnum_data.copy()
    
    def get_eq_data(self) -> List[str]:
        """Return EQ dataset"""
        return self.eq_data.copy()
    
    def get_eqsp_data(self) -> List[str]:
        """Return EQSP dataset"""
        return self.eqsp_data.copy()
    
    def get_langnumeq_data(self) -> List[str]:
        """Return LANGNUMEQ dataset"""
        return self.langnumeq_data.copy()
    
    def get_gsm8k_data(self) -> List[str]:
        """Return GSM8K dataset"""
        return self.gsm8k_data.copy()
    
    def get_all_datasets(self) -> Dict[str, List[str]]:
        """Return all datasets"""
        return {
            "LANG": self.get_lang_data(),
            "LANGNUM": self.get_langnum_data(),
            "EQ": self.get_eq_data(),
            "EQSP": self.get_eqsp_data(),
            "LANGNUMEQ": self.get_langnumeq_data(),
            "GSM8K": self.get_gsm8k_data()
        }
    
    def get_expected_answers(self, dataset_name: str) -> List[str]:
        """Get expected answers for each dataset"""
        if dataset_name == "LANG":
            return [""] * len(self.lang_data)  # No specific answers for language questions
        
        elif dataset_name == "LANGNUM":
            return ["1", "8", "1776", "206", "8400000", "46", "186282", "50", "100", "366",
                   "6", "7", "238900", "26", "72", "6", "8", "5", "32", "24"]
        
        elif dataset_name == "EQ":
            return ["1", "12", "4", "16", "5", "24", "3", "11", "6", "18",
                   "5", "12", "7", "12", "8", "9", "5", "10", "6", "9"]
        
        elif dataset_name == "EQSP":
            return ["1", "12", "4", "16", "5", "24", "3", "11", "6", "18",
                   "5", "12", "7", "12", "8", "9", "5", "10", "6", "9"]
        
        elif dataset_name == "LANGNUMEQ":
            return ["1", "10", "6", "4", "6", "16", "13", "4", "8", "7",
                   "12", "7", "8", "4", "7", "6", "3", "4", "18", "5"]
        
        elif dataset_name == "GSM8K":
            return ["3", "7", "39", "5", "6", "39", "5", "6", "5", "6",
                   "100", "6", "5", "6", "5", "6", "5", "6", "5", "6"]
        
        else:
            return [""] * 20

if __name__ == "__main__":
    # Test the new data generator
    generator = NewDataGenerator()
    datasets = generator.get_all_datasets()
    
    print("New Dataset sizes:")
    for name, data in datasets.items():
        print(f"{name}: {len(data)} samples")
    
    print("\nSample data:")
    for name, data in datasets.items():
        print(f"{name}: {data[0]}")
        if name != "LANG":
            answers = generator.get_expected_answers(name)
            print(f"  Expected answer: {answers[0]}")
        print() 
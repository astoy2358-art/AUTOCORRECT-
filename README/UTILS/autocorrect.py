# utils/autocorrect.py (enhanced version)
import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Autocorrect:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.words = set(self.df['word'].str.lower())
        self.word_freq = dict(zip(self.df['word'].str.lower(), self.df['frequency']))
        
    def preprocess_text(self, text):
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        return tokens
    
    def levenshtein_distance(self, s1, s2):
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_candidates(self, word, max_distance=2):
        """Generate candidate corrections for a word"""
        candidates = []
        
        # If word is in our vocabulary, it might be correct
        if word in self.words:
            candidates.append((word, 0, self.word_freq.get(word, 0)))
        
        # Check for words with small edit distance
        for vocab_word in self.words:
            # Only check words of similar length to improve performance
            if abs(len(word) - len(vocab_word)) <= max_distance:
                distance = self.levenshtein_distance(word, vocab_word)
                if distance <= max_distance:
                    freq = self.word_freq.get(vocab_word, 0)
                    candidates.append((vocab_word, distance, freq))
        
        # Sort by distance (primary) and frequency (secondary)
        candidates.sort(key=lambda x: (x[1], -x[2]))
        
        # Return top candidates (up to 5)
        return [candidate[0] for candidate in candidates[:5]]
    
    def correct_word(self, word):
        """Correct a single word"""
        # Don't correct very short words, numbers, or email/URLs
        if (len(word) <= 2 or word.isdigit() or 
            '@' in word or '://' in word or word.startswith('www.')):
            return word
        
        # If word is in our vocabulary, return as is
        if word.lower() in self.words:
            return word
        
        # Get candidate corrections
        candidates = self.get_candidates(word.lower())
        
        if candidates:
            # Return the best candidate (preserve original case)
            if word[0].isupper():
                return candidates[0].capitalize()
            return candidates[0]
        
        # If no candidates found, return original word
        return word
    
    def correct_sentence(self, sentence):
        """Correct a full sentence"""
        # Use regex to split while preserving whitespace and punctuation
        tokens = re.findall(r"(\w+|\s+|\S)", sentence)
        
        corrected_tokens = []
        for token in tokens:
            if re.match(r"\w+", token):  # If it's a word
                corrected_tokens.append(self.correct_word(token))
            else:  # If it's whitespace or punctuation
                corrected_tokens.append(token)
        
        return ''.join(corrected_tokens)
    
    def generate_chart(self, top_n=20):
        """Generate a bar chart of the most frequent words"""
        top_words = self.df.nlargest(top_n, 'frequency')
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(top_n), top_words['frequency'], color='skyblue')
        plt.xticks(range(top_n), top_words['word'], rotation=45, ha='right')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title(f'Top {top_n} Most Frequent Words')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the chart
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/chart.png', dpi=150, bbox_inches='tight')
        plt.close()

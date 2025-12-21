"""
Intelligent Tutoring System Engine
Implements mistake tracking and adaptive word generation
AI Components: Rule-Based System + Constraint Satisfaction Search
"""

import random
import json
from collections import defaultdict

class MistakeTracker:
    """
    AI Component #2: Rule-Based System (Week 4)
    Tracks user mistakes and applies inference rules
    """
    
    def __init__(self):
        self.mistake_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.total_attempts = defaultdict(int)
        
    def record_attempt(self, letter, was_correct):
        """Record user's attempt at a letter"""
        self.total_attempts[letter] += 1
        
        if was_correct:
            self.success_counts[letter] += 1
        else:
            self.mistake_counts[letter] += 1
    
    def get_error_rate(self, letter):
        """Calculate error rate for a letter"""
        if self.total_attempts[letter] == 0:
            return 0.0
        return self.mistake_counts[letter] / self.total_attempts[letter]
    
    def get_high_error_letters(self, threshold=0.3, min_attempts=3):
        """
        INFERENCE RULE:
        IF error_rate(letter) > threshold AND attempts >= min_attempts
        THEN classify as 'high_error_letter'
        """
        high_error = []
        
        for letter in self.total_attempts:
            if self.total_attempts[letter] >= min_attempts:
                error_rate = self.get_error_rate(letter)
                if error_rate > threshold:
                    high_error.append((letter, error_rate))
        
        # Sort by error rate (highest first)
        high_error.sort(key=lambda x: x[1], reverse=True)
        return [letter for letter, _ in high_error]
    
    def should_activate_word_mode(self, threshold=0.3):
        """
        INFERENCE RULE:
        IF exists letter with error_rate > threshold
        THEN activate adaptive word mode
        """
        high_error = self.get_high_error_letters(threshold)
        return len(high_error) >= 2  # At least 2 problem letters
    
    def get_stats(self):
        """Get statistics for display"""
        stats = {}
        for letter in self.total_attempts:
            stats[letter] = {
                'attempts': self.total_attempts[letter],
                'successes': self.success_counts[letter],
                'mistakes': self.mistake_counts[letter],
                'error_rate': self.get_error_rate(letter)
            }
        return stats


class DictionarySearch:
    """
    AI Component #3: Constraint Satisfaction Search (Week 2)
    Finds words containing specific letters (constraint satisfaction problem)
    """
    
    def __init__(self):
        # Common 3-4 letter words for practice
        self.dictionary = [
            # 3-letter words
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT',
            'BED', 'RED', 'LED', 'FED', 'PET', 'WET', 'MET', 'GET', 'SET', 'LET',
            'BIG', 'DIG', 'FIG', 'PIG', 'WIG', 'JAG', 'LAG', 'NAG', 'RAG', 'TAG', 'WAG',
            'BAD', 'DAD', 'HAD', 'LAD', 'MAD', 'PAD', 'SAD',
            'BAN', 'CAN', 'DAN', 'FAN', 'MAN', 'PAN', 'RAN', 'TAN', 'VAN',
            # 4-letter words
            'CARD', 'CART', 'CAST', 'FAST', 'LAST', 'MAST', 'PAST', 'VAST',
            'BEAR', 'DEAR', 'FEAR', 'GEAR', 'HEAR', 'NEAR', 'PEAR', 'TEAR', 'WEAR', 'YEAR',
            'BELL', 'CELL', 'DELL', 'FELL', 'JELL', 'SELL', 'TELL', 'WELL', 'YELL',
            'BOAT', 'COAT', 'GOAT', 'MOAT',
            'FACE', 'LACE', 'MACE', 'PACE', 'RACE',
            'FROG', 'FROM',
        ]
    
    def find_word_with_letters(self, required_letters, max_word_length=4):
        """
        SEARCH ALGORITHM: Constraint Satisfaction
        
        Constraints:
        1. Word must contain ALL required_letters
        2. Word length <= max_word_length (for difficulty control)
        
        Search strategy: Greedy (prefer shorter words)
        """
        candidates = []
        
        for word in self.dictionary:
            # Check constraint #1: contains all required letters
            if all(letter in word for letter in required_letters):
                # Check constraint #2: length limit
                if len(word) <= max_word_length:
                    candidates.append(word)
        
        if not candidates:
            # BACKTRACKING: Relax constraint - find words with MOST letters
            for word in self.dictionary:
                match_count = sum(1 for letter in required_letters if letter in word)
                if match_count >= len(required_letters) - 1:  # Allow 1 missing
                    candidates.append(word)
        
        if not candidates:
            # FALLBACK: Random word
            return random.choice(self.dictionary)
        
        # Return shortest word (easier for user)
        candidates.sort(key=len)
        return candidates[0]
    
    def generate_practice_word(self, high_error_letters):
        """
        Generate word targeting user's weak points
        """
        if not high_error_letters:
            return random.choice(self.dictionary)
        
        # Take top 2-3 problem letters
        target_letters = high_error_letters[:min(3, len(high_error_letters))]
        
        return self.find_word_with_letters(target_letters)


class IntelligentTutorSystem:
    """
    Complete ITS combining both AI components
    """
    
    def __init__(self):
        self.mistake_tracker = MistakeTracker()
        self.dictionary_search = DictionarySearch()
        self.current_mode = 'random'  # 'random', 'adaptive_word', or 'selected'
        self.current_word = None
        self.current_word_index = 0
        
        # Gamification: No Repeat Logic
        self.all_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        # Removed dynamic letters J/Z for now as they require motion
        self.available_letters = list(self.all_letters)
        random.shuffle(self.available_letters)
        self.forced_letter = None
        
    def set_forced_letter(self, letter):
        """User selected a specific letter from menu"""
        self.current_mode = 'selected'
        self.forced_letter = letter
        print(f"[ITS] User selected: {letter}")

    def reset_random_pool(self):
        """Reset the pool of letters for random mode"""
        self.available_letters = list(self.all_letters)
        random.shuffle(self.available_letters)
        print("[ITS] Resetting random letter pool")

    def record_attempt(self, letter, was_correct):
        """Process user attempt"""
        self.mistake_tracker.record_attempt(letter, was_correct)
        
        # Check if should switch to adaptive mode
        if self.mistake_tracker.should_activate_word_mode():
            if self.current_mode == 'random':
                self.activate_word_mode()
    
    def activate_word_mode(self):
        """Switch to adaptive word mode"""
        self.current_mode = 'adaptive_word'
        high_error = self.mistake_tracker.get_high_error_letters()
        self.current_word = self.dictionary_search.generate_practice_word(high_error)
        self.current_word_index = 0
        print(f"\n[ITS] Activating Word Mode: {self.current_word}")
        print(f"[ITS] Targeting weak letters: {high_error}")
    
    def get_next_letter(self):
        """
        ADAPTIVE LETTER SELECTION
        Returns next letter based on current mode
        """
        # 1. Selected Mode (User forced a letter)
        if self.current_mode == 'selected':
            return self.forced_letter

        # 2. Random Mode (No Repeats)
        if self.current_mode == 'random':
            if not self.available_letters:
                self.reset_random_pool()
            
            return self.available_letters.pop()
        
        # 3. Word Mode
        elif self.current_mode == 'adaptive_word':
            # Word mode - spell out the practice word
            if self.current_word_index >= len(self.current_word):
                # Word completed, generate new word
                high_error = self.mistake_tracker.get_high_error_letters()
                self.current_word = self.dictionary_search.generate_practice_word(high_error)
                self.current_word_index = 0
                print(f"\n[ITS] New word: {self.current_word}")
            
            letter = self.current_word[self.current_word_index]
            self.current_word_index += 1
            return letter
    
    def get_progress_info(self):
        """Get info for UI display"""
        info = {
            'mode': self.current_mode,
            'current_word': self.current_word,
            'word_progress': self.current_word_index if self.current_word else 0,
            'high_error_letters': self.mistake_tracker.get_high_error_letters()[:5]
        }
        return info
    
    def save_session(self, filename='its_session.json'):
        """Save session statistics"""
        stats = self.mistake_tracker.get_stats()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_session(self, filename='its_session.json'):
        """Load previous session"""
        try:
            with open(filename, 'r') as f:
                stats = json.load(f)
            
            for letter, data in stats.items():
                self.mistake_tracker.total_attempts[letter] = data['attempts']
                self.mistake_tracker.success_counts[letter] = data['successes']
                self.mistake_tracker.mistake_counts[letter] = data['mistakes']
            
            print(f"[ITS] Loaded previous session")
        except FileNotFoundError:
            print("[ITS] No previous session found")


if __name__ == '__main__':
    # Test the ITS
    its = IntelligentTutorSystem()
    
    # Simulate user struggling with A and T
    print("Simulating user making mistakes on A and T...")
    its.record_attempt('A', False)
    its.record_attempt('A', False)
    its.record_attempt('A', False)
    its.record_attempt('T', False)
    its.record_attempt('T', False)
    its.record_attempt('C', True)
    its.record_attempt('C', True)
    
    print(f"\nHigh error letters: {its.mistake_tracker.get_high_error_letters()}")
    print(f"Should activate word mode? {its.mistake_tracker.should_activate_word_mode()}")
    
    if its.mistake_tracker.should_activate_word_mode():
        its.activate_word_mode()
        print(f"Next letters will be: {[its.get_next_letter() for _ in range(5)]}")

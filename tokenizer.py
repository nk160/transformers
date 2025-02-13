import torch
from collections import Counter
from typing import List, Dict, Optional, Set, Tuple
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns

class CaptionTokenizer:
    """
    Tokenizer for image captions.
    Handles conversion between text and token ids.
    """
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.PAD_token = 0
        self.START_token = 1
        self.END_token = 2
        self.UNK_token = 3
        
        # Special token strings
        self.special_tokens = {
            self.PAD_token: "[PAD]",
            self.START_token: "[START]",
            self.END_token: "[END]",
            self.UNK_token: "[UNK]"
        }
        
        # Initialize vocabularies
        self.word2idx: Dict[str, int] = {v: k for k, v in self.special_tokens.items()}
        self.idx2word: Dict[int, str] = self.special_tokens
        self.word_freq: Counter = Counter()
        
    def build_vocab(self, captions: List[str]):
        """
        Build vocabulary from list of captions.
        
        Args:
            captions: List of caption strings
        """
        # Count words in all captions
        for caption in captions:
            words = caption.lower().split()
            self.word_freq.update(words)
        
        # Sort by frequency and take top vocab_size words
        most_common = self.word_freq.most_common(self.vocab_size - len(self.special_tokens))
        
        # Add to vocabulary
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word 

    def encode(self, caption: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Convert a caption string to token ids.
        
        Args:
            caption: Caption string to encode
            max_length: Optional maximum length for padding/truncation
        
        Returns:
            Tensor of token ids with START and END tokens added
        """
        # Tokenize and convert to indices
        words = caption.lower().split()
        tokens = [self.START_token]  # Add START token
        
        # Add word tokens
        for word in words:
            tokens.append(self.word2idx.get(word, self.UNK_token))
        
        tokens.append(self.END_token)  # Add END token
        
        # Handle max_length
        if max_length is not None:
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.END_token]
            # Pad if too short
            else:
                tokens.extend([self.PAD_token] * (max_length - len(tokens)))
        
        return torch.tensor(tokens)
    
    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Convert token ids back to a caption string.
        
        Args:
            tokens: Tensor of token ids
            skip_special_tokens: Whether to remove special tokens from output
        
        Returns:
            Caption string
        """
        words = []
        for token in tokens:
            idx = token.item() if torch.is_tensor(token) else token
            
            # Skip special tokens if requested
            if skip_special_tokens and idx in self.special_tokens:
                continue
                
            words.append(self.idx2word.get(idx, self.special_tokens[self.UNK_token]))
        
        return ' '.join(words)
    
    def save_vocab(self, path: str):
        """
        Save vocabulary to a JSON file.
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},  # Convert keys to strings for JSON
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f)
    
    @classmethod
    def from_file(cls, path: str) -> 'CaptionTokenizer':
        """
        Load vocabulary from a JSON file.
        """
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(vocab_size=vocab_data['vocab_size'])
        tokenizer.word2idx = vocab_data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}  # Convert keys back to integers
        
        return tokenizer 

    def encode_batch(self, captions: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Convert a batch of captions to token ids.
        
        Args:
            captions: List of caption strings
            max_length: Optional maximum length for padding/truncation
                       If None, will pad to longest sequence in batch
        
        Returns:
            Tensor of token ids with shape (batch_size, seq_length)
        """
        # If max_length not provided, find longest sequence after tokenization
        if max_length is None:
            max_length = max(
                len(caption.split()) + 2  # +2 for START and END tokens
                for caption in captions
            )
        
        # Encode all captions
        encoded = [
            self.encode(caption, max_length=max_length)
            for caption in captions
        ]
        
        # Stack into single tensor
        return torch.stack(encoded)
    
    def decode_batch(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Convert a batch of token ids back to caption strings.
        
        Args:
            tokens: Tensor of token ids with shape (batch_size, seq_length)
            skip_special_tokens: Whether to remove special tokens from output
        
        Returns:
            List of caption strings
        """
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in tokens
        ]
    
    def get_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            tokens: Tensor of token ids with shape (batch_size, seq_length)
        
        Returns:
            Boolean mask where True indicates non-pad tokens
        """
        return tokens != self.PAD_token 

    def clean_caption(self, caption: str) -> str:
        """
        Clean and normalize caption text.
        """
        # Convert to lowercase
        caption = caption.lower()
        
        # Basic punctuation handling
        punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
        caption = caption.translate(str.maketrans(punctuation, ' ' * len(punctuation)))
        
        # Remove extra whitespace
        caption = ' '.join(caption.split())
        
        return caption
    
    def get_vocab_size(self) -> int:
        """
        Get current vocabulary size including special tokens.
        """
        return len(self.word2idx)
    
    def get_most_common_words(self, n: int = 10) -> List[tuple]:
        """
        Get n most common words and their frequencies.
        """
        return self.word_freq.most_common(n)
    
    def token_to_id(self, token: str) -> int:
        """
        Convert a single token to its ID.
        """
        return self.word2idx.get(token.lower(), self.UNK_token)
    
    def id_to_token(self, idx: int) -> str:
        """
        Convert a single ID to its token.
        """
        return self.idx2word.get(idx, self.special_tokens[self.UNK_token])
    
    def preprocess_caption(self, caption: str, remove_numbers: bool = False, 
                         min_word_length: int = 2, max_word_length: int = 20) -> str:
        """
        Advanced caption preprocessing.
        
        Args:
            caption: Input caption string
            remove_numbers: Whether to remove numerical tokens
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        # Basic cleaning
        caption = self.clean_caption(caption)
        
        # Split into words
        words = caption.split()
        
        # Filter words
        filtered_words = []
        for word in words:
            # Skip numbers if requested
            if remove_numbers and word.isdigit():
                continue
                
            # Check word length
            if min_word_length <= len(word) <= max_word_length:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def analyze_vocabulary(self) -> Dict:
        """
        Analyze vocabulary statistics.
        """
        stats = {
            'total_vocab_size': self.get_vocab_size(),
            'num_special_tokens': len(self.special_tokens),
            'total_words': sum(self.word_freq.values()),
            'unique_words': len(self.word_freq),
            'most_common': self.get_most_common_words(10),
            'avg_word_length': sum(len(word) * freq 
                                 for word, freq in self.word_freq.items()) / 
                             sum(self.word_freq.values()),
            'word_length_dist': self._get_word_length_distribution(),
            'coverage': self._calculate_vocabulary_coverage()
        }
        return stats
    
    def _get_word_length_distribution(self) -> Dict[int, int]:
        """
        Get distribution of word lengths in vocabulary.
        """
        length_dist = {}
        for word, freq in self.word_freq.items():
            length = len(word)
            length_dist[length] = length_dist.get(length, 0) + freq
        return length_dist
    
    def _calculate_vocabulary_coverage(self) -> float:
        """
        Calculate what percentage of total words are covered by vocabulary.
        """
        vocab_words = set(self.word2idx.keys()) - set(self.special_tokens.values())
        total_freq = sum(self.word_freq.values())
        covered_freq = sum(self.word_freq[word] for word in vocab_words 
                         if word in self.word_freq)
        return covered_freq / total_freq if total_freq > 0 else 0.0
    
    def get_rare_words(self, threshold: int = 5) -> Set[str]:
        """
        Get set of words that appear less than threshold times.
        """
        return {word for word, freq in self.word_freq.items() 
                if freq < threshold}
    
    def get_vocabulary_overlap(self, other_tokenizer: 'CaptionTokenizer') -> Tuple[Set[str], float]:
        """
        Compare vocabulary with another tokenizer.
        
        Returns:
            Set of common words and overlap percentage
        """
        vocab1 = set(self.word2idx.keys()) - set(self.special_tokens.values())
        vocab2 = set(other_tokenizer.word2idx.keys()) - set(other_tokenizer.special_tokens.values())
        
        common_words = vocab1.intersection(vocab2)
        overlap_percentage = len(common_words) / len(vocab1) if vocab1 else 0.0
        
        return common_words, overlap_percentage 

    def plot_word_frequency_distribution(self, top_n: int = 50):
        """
        Plot distribution of word frequencies.
        """
        plt.figure(figsize=(15, 5))
        words, freqs = zip(*self.get_most_common_words(top_n))
        sns.barplot(x=list(range(len(words))), y=freqs)
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.title(f'Top {top_n} Most Frequent Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    
    def plot_word_length_distribution(self):
        """
        Plot distribution of word lengths.
        """
        dist = self._get_word_length_distribution()
        plt.figure(figsize=(10, 5))
        lengths = sorted(dist.keys())
        freqs = [dist[l] for l in lengths]
        sns.barplot(x=lengths, y=freqs)
        plt.title('Word Length Distribution')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show() 
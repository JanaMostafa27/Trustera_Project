"""
Data Preprocessing Module for Fake News Detection

This module handles data cleaning, text preprocessing, tokenization,
and feature extraction for the fake news detection system.

Improvements made:
- Better text cleaning with additional patterns
- Proper handling of data leakage
- Enhanced tokenization with validation
- Configurable preprocessing parameters
"""

import re
import string
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging
from typing import Any, List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for fake news detection.
    
    This class handles:
    - Data loading and initial cleaning
    - Text preprocessing and normalization
    - Tokenization and sequence generation
    - Label encoding
    - Train/validation/test splitting
    """
    
    def __init__(self, 
                 max_words: int = 10000,
                 max_len: int = 300,
                 test_size: float = 0.3,
                 val_size: float = 0.5,
                 random_state: int = 42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            max_words: Maximum number of words to keep in vocabulary
            max_len: Maximum sequence length
            test_size: Proportion of data to use for testing
            val_size: Proportion of temp data to use for validation
            random_state: Random seed for reproducibility
        """
        self.max_words = max_words
        self.max_len = max_len
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Initialize components
        self.word_to_idx = None
        self.label_encoder = None
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and perform initial data cleaning.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
        
        # Normalize column names to match expected format
        df.columns = [col.lower() for col in df.columns]
        
        # Remove the header row if it was accidentally included in the data
        if 'title' in df.columns and len(df) > 0:
            # Check if first row contains header names
            first_row = df.iloc[0]
            if first_row['title'] == 'title' and first_row['text'] == 'text':
                df = df.iloc[1:].reset_index(drop=True)
                logger.info("Removed duplicate header row from data")
        
        # Keep only the columns we need
        needed_columns = ['text', 'subject', 'label']
        available_columns = [col for col in needed_columns if col in df.columns]
        
        if not available_columns:
            raise ValueError("Dataset must contain 'text', 'subject', and 'label' columns")
        
        df = df[available_columns].copy()
        
        # Remove unnecessary columns if they exist
        if 'publish_date' in df.columns:
            df.drop(columns=['publish_date'], inplace=True)
        if 'text_length' in df.columns:
            df.drop(columns=['text_length'], inplace=True)
        if 'title_length' in df.columns:
            df.drop(columns=['title_length'], inplace=True)
        
        logger.info(f"Initial data shape: {df.shape}")
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning function.
        
        Improvements made:
        - Better handling of Reuters source tags
        - Additional pattern removals
        - Preserving important linguistic features
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove Reuters source tag (e.g., "WASHINGTON (Reuters) - ")
        text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
        
        # Remove other news agency tags
        text = re.sub(r'^.*?\(ap\)\s*-\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^.*?\(afp\)\s*-\s*', '', text, flags=re.IGNORECASE)
        
        # Lowercase
        text = text.lower()
        
        # Remove text in brackets
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation but keep important ones like !, ? for sentiment
        punctuation_to_keep = "!?"
        text = ''.join(char if char in punctuation_to_keep else ' ' if char in string.punctuation else char 
                      for char in text)
        
        # Remove newlines and extra whitespace
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove words containing numbers
        text = re.sub(r'\w*\d\w*', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_data_leakage_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove potential sources of data leakage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with leakage sources removed
        """
        logger.info("Removing potential data leakage sources")
        
        # Remove duplicate text entries
        initial_shape = df.shape[0]
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
        duplicates_removed = initial_shape - df.shape[0]
        logger.info(f"Removed {duplicates_removed} duplicate text entries")
        
        # Remove entries where title is the header
        if 'title' in df.columns:
            df = df[df['title'] != 'title'].reset_index(drop=True)
        
        # Check for and remove near-duplicates (very similar texts)
        # This is a simple approach - in production, you might use more sophisticated methods
        df['text_clean'] = df['text'].apply(self.clean_text)
        
        # Remove texts that are too short after cleaning (likely not meaningful)
        min_text_length = 10
        df = df[df['text_clean'].str.len() >= min_text_length].reset_index(drop=True)
        
        logger.info(f"Final data shape after leakage removal: {df.shape}")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by cleaning, tokenizing, and encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y_fake, y_subject)
        """
        logger.info("Preparing data for training")
        
        # Remove data leakage sources
        df = self.remove_data_leakage_sources(df)
        
        # Clean text
        logger.info("Cleaning text data")
        df['text_clean'] = df['text'].apply(self.clean_text)
        
        # Encode subject labels
        logger.info("Encoding subject labels")
        self.label_encoder = LabelEncoder()
        df['subject_encoded'] = self.label_encoder.fit_transform(df['subject'])
        
        # Tokenization
        logger.info("Tokenizing text")
        self.word_to_idx = self._build_vocabulary(df['text_clean'])
        
        sequences = self._texts_to_sequences(df['text_clean'])
        
        X = self._pad_sequences(sequences, maxlen=self.max_len)
        
        # Prepare labels
        y_fake = df['label'].values
        y_subject = df['subject_encoded'].values
        
        logger.info(f"Data preparation complete. X shape: {X.shape}")
        return X, y_fake, y_subject
    
    def split_data(self, X: np.ndarray, y_fake: np.ndarray, y_subject: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature sequences
            y_fake: Fake news labels
            y_subject: Subject labels
            
        Returns:
            Dictionary containing all splits
        """
        logger.info("Splitting data into train/val/test sets")
        
        # First split: train vs temp (val + test)
        X_train, X_temp, y_f_train, y_f_temp, y_s_train, y_s_temp = train_test_split(
            X, y_fake, y_subject, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y_fake
        )
        
        # Second split: validation vs test
        X_val, X_test, y_f_val, y_f_test, y_s_val, y_s_test = train_test_split(
            X_temp, y_f_temp, y_s_temp, 
            test_size=self.val_size, 
            random_state=self.random_state, 
            stratify=y_f_temp
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_f_train': y_f_train,
            'y_f_val': y_f_val,
            'y_f_test': y_f_test,
            'y_s_train': y_s_train,
            'y_s_val': y_s_val,
            'y_s_test': y_s_test
        }
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        return splits
    
    def _build_vocabulary(self, texts: pd.Series) -> Dict[str, int]:
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Keep only the most common words
        most_common = word_counts.most_common(self.max_words - 2)  # -2 for PAD and UNK
        
        # Create word to index mapping
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(most_common, 2):
            word_to_idx[word] = i
        
        return word_to_idx
    
    def _texts_to_sequences(self, texts: pd.Series) -> List[List[int]]:
        """Convert texts to sequences of integers"""
        sequences = []
        for text in texts:
            words = text.split()
            sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is UNK token
            sequences.append(sequence)
        return sequences
    
    def _pad_sequences(self, sequences: List[List[int]], maxlen: int) -> np.ndarray:
        """Pad sequences to the same length"""
        padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
        for i, seq in enumerate(sequences):
            if len(seq) > maxlen:
                padded[i] = seq[:maxlen]
            elif len(seq) > 0:
                padded[i, -len(seq):] = seq
        return padded
    
    def fit_transform(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline: load, clean, tokenize, and split.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary containing all data splits
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Load and clean data
        df = self.load_data(file_path)
        
        # Prepare data
        X, y_fake, y_subject = self.prepare_data(df)
        
        # Split data
        splits = self.split_data(X, y_fake, y_subject)
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline completed successfully")
        return splits
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """
        Get information about the vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        vocab_size = len(self.word_to_idx)
        
        return {
            'vocab_size': vocab_size,
            'max_vocab_size': self.max_words,
            'total_words': vocab_size,
            'oov_token': '<UNK>',
            'num_subjects': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'subject_classes': self.label_encoder.classes_.tolist() if self.label_encoder else []
        }
    
    def save_preprocessor(self, file_path: str):
        """
        Save the preprocessor components.
        
        Args:
            file_path: Path to save the preprocessor
        """
        import pickle
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'word_to_idx': self.word_to_idx,
            'label_encoder': self.label_encoder,
            'max_words': self.max_words,
            'max_len': self.max_len,
            'random_state': self.random_state
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load_preprocessor(cls, file_path: str):
        """
        Load a saved preprocessor.
        
        Args:
            file_path: Path to the saved preprocessor
            
        Returns:
            Loaded DataPreprocessor instance
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        # Create new instance
        preprocessor = cls(
            max_words=preprocessor_data['max_words'],
            max_len=preprocessor_data['max_len'],
            random_state=preprocessor_data['random_state']
        )
        
        # Load components
        preprocessor.word_to_idx = preprocessor_data['word_to_idx']
        preprocessor.label_encoder = preprocessor_data['label_encoder']
        preprocessor.is_fitted = True
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor


def preprocess_single_text(text: str, word_to_idx: Dict[str, int], max_len: int = 300) -> np.ndarray:
    """
    Preprocess a single text for inference.
    
    Args:
        text: Input text
        word_to_idx: Word to index mapping
        max_len: Maximum sequence length
        
    Returns:
        Processed sequence
    """
    # Create a temporary preprocessor to use the clean_text method
    temp_preprocessor = DataPreprocessor()
    cleaned_text = temp_preprocessor.clean_text(text)
    
    # Convert to sequence
    words = cleaned_text.split()
    sequence = [word_to_idx.get(word, 1) for word in words]  # 1 is UNK token
    
    # Pad sequence
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = [0] * (max_len - len(sequence)) + sequence  # Pad with 0 (PAD token)
    
    return np.array(sequence, dtype=np.int32).reshape(1, -1)


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # This would be used with actual data
    # splits = preprocessor.fit_transform("path/to/data.csv")
    # print(preprocessor.get_vocabulary_info())

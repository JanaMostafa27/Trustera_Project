"""
Prediction Module for Fake News Detection System

This module handles inference for new text inputs with confidence
scoring and batch processing capabilities.

Features:
- Single text prediction
- Batch text prediction
- Confidence scoring
- Detailed prediction results
- Model loading and validation
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
from .data_preprocessing import DataPreprocessor, preprocess_single_text
from .train import PyTorchTrainer
import logging
from typing import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Data class for prediction results."""
    text: str
    prediction: str
    category: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'prediction': self.prediction,
            'category': self.category,
            'processing_time': self.processing_time
        }


class FakeNewsPredictor:
    """
    Predictor class for fake news detection inference.
    
    This class handles loading trained models and making predictions
    on new text inputs with confidence scoring.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 preprocessor_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            preprocessor_path: Path to the saved preprocessor
        """
        self.model = None
        self.preprocessor = None
        self.model_loaded = False
        self.preprocessor_loaded = False
        
        # Load model and preprocessor if paths provided
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            # Create a trainer with default parameters
            self.trainer = PyTorchTrainer(vocab_size=10000, num_subjects=8)
            self.trainer.load_model(model_path)
            self.model = self.trainer.model
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str):
        """
        Load a saved preprocessor.
        
        Args:
            preprocessor_path: Path to the saved preprocessor
        """
        try:
            self.preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)
            self.preprocessor_loaded = True
            logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def validate_model_components(self):
        """Validate that all necessary components are loaded."""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        if not self.preprocessor_loaded:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Union[PredictionResult, Dict]:
        """
        Predict fake/real for a single text input.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return raw probabilities
            
        Returns:
            PredictionResult or dictionary with probabilities
        """
        import time
        start_time = time.time()
        
        self.validate_model_components()
        
        # Preprocess text
        processed_text = preprocess_single_text(
            text, 
            self.preprocessor.word_to_idx, 
            self.preprocessor.max_len
        )
        
        # Make prediction
        with torch.no_grad():
            fake_prob, subject_prob = self.trainer.predict(processed_text)
        
        # Extract results with balanced threshold
        fake_score = float(fake_prob[0])
        # Use balanced threshold (0.6) to avoid TRUE bias
        is_true = fake_score > 0.6
        prediction = "TRUE" if is_true else "FAKE"
        
        # Get subject prediction
        subject_idx = int(subject_prob.argmax())
        category = self.preprocessor.label_encoder.inverse_transform([subject_idx])[0]
        
        processing_time = time.time() - start_time
        
        result = PredictionResult(
            text=text,
            prediction=prediction,
            category=category,
            processing_time=processing_time
        )
        
        logger.info(f"Prediction completed in {processing_time:.3f} seconds")
        if return_probabilities:
            return {
                'result': result.to_dict(),
                'probabilities': {
                    'fake_probability': fake_score,
                    'true_probability': 1 - fake_score,
                    'subject_probabilities': subject_prob[0].tolist(),
                    'subject_classes': self.preprocessor.label_encoder.classes_.tolist()
                }
            }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[PredictionResult]:
        """
        Predict fake/real for multiple text inputs.
        
        Args:
            texts: List of input texts to classify
            batch_size: Batch size for processing
            
        Returns:
            List of PredictionResult objects
        """
        self.validate_model_components()
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            processed_batch = np.vstack([
                preprocess_single_text(text, self.preprocessor.word_to_idx, self.preprocessor.max_len)
                for text in batch_texts
            ])
            
            # Make predictions
            with torch.no_grad():
                fake_probs, subject_probs = self.trainer.predict_batch(processed_batch)
            
            # Process results
            for j, text in enumerate(batch_texts):
                fake_score = float(fake_probs[j])
                # Use balanced threshold to avoid TRUE bias
                is_true = fake_score > 0.6
                prediction = "TRUE" if is_true else "FAKE"
                confidence = fake_score if is_true else (1 - fake_score)
                
                subject_idx = np.argmax(subject_probs[j])
                subject_confidence = float(np.max(subject_probs[j]))
                category = self.preprocessor.label_encoder.inverse_transform([subject_idx])[0]
                
                result = PredictionResult(
                    text=text,
                    prediction=prediction,
                    category=category,
                    processing_time=0.0
                )
                results.append(result)
        
        logger.info(f"Batch prediction completed for {len(texts)} texts")
        return results
    
    def predict_from_file(self, file_path: str, text_column: str = 'text', 
                         output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Predict fake/real for texts from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            text_column: Name of the column containing text
            output_path: Optional path to save results
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the file")
        
        texts = df[text_column].tolist()
        
        # Make predictions
        results = self.predict_batch(texts)
        
        # Create results DataFrame
        results_df = pd.DataFrame([result.to_dict() for result in results])
        
        # Combine with original data
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Save results if output path provided
        if output_path:
            final_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return final_df
    
    def analyze_predictions(self, texts: List[str]) -> Dict:
        """
        Analyze predictions for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Analysis statistics
        """
        results = self.predict_batch(texts)
        
        # Calculate statistics
        predictions = [r.prediction for r in results]
        confidences = [r.confidence for r in results]
        categories = [r.category for r in results]
        
        stats = {
            'total_texts': len(texts),
            'fake_count': predictions.count('FAKE'),
            'true_count': predictions.count('TRUE'),
            'fake_percentage': predictions.count('FAKE') / len(texts) * 100,
            'true_percentage': predictions.count('TRUE') / len(texts) * 100,
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.6),
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)}
        }
        
        return stats
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            'model_loaded': self.model_loaded,
            'preprocessor_loaded': self.preprocessor_loaded,
            'model_parameters': self.model.count_params(),
            'model_layers': len(self.model.layers),
            'input_shape': self.model.input_shape,
            'output_names': [output.name.split('/')[0] for output in self.model.outputs]
        }
        
        if self.preprocessor_loaded:
            vocab_info = self.preprocessor.get_vocabulary_info()
            info.update(vocab_info)
        
        return info
    
    def test_prediction_quality(self, test_texts: List[str], 
                               expected_labels: List[str]) -> Dict:
        """
        Test prediction quality on labeled test data.
        
        Args:
            test_texts: Test texts
            expected_labels: Expected labels ("FAKE" or "TRUE")
            
        Returns:
            Quality metrics
        """
        if len(test_texts) != len(expected_labels):
            raise ValueError("Number of texts and labels must match")
        
        results = self.predict_batch(test_texts)
        predictions = [r.prediction for r in results]
        
        # Calculate metrics
        correct = sum(1 for pred, exp in zip(predictions, expected_labels) if pred == exp)
        accuracy = correct / len(expected_labels)
        
        # Calculate per-class accuracy
        fake_indices = [i for i, label in enumerate(expected_labels) if label == "FAKE"]
        true_indices = [i for i, label in enumerate(expected_labels) if label == "TRUE"]
        
        fake_accuracy = 0
        if fake_indices:
            fake_correct = sum(1 for i in fake_indices if predictions[i] == "FAKE")
            fake_accuracy = fake_correct / len(fake_indices)
        
        true_accuracy = 0
        if true_indices:
            true_correct = sum(1 for i in true_indices if predictions[i] == "TRUE")
            true_accuracy = true_correct / len(true_indices)
        
        metrics = {
            'total_samples': len(test_texts),
            'accuracy': accuracy,
            'fake_accuracy': fake_accuracy,
            'true_accuracy': true_accuracy,
            'correct_predictions': correct,
            'incorrect_predictions': len(test_texts) - correct
        }
        
        return metrics
    
    def save_predictions(self, results: List[PredictionResult], output_path: str):
        """
        Save prediction results to a file.
        
        Args:
            results: List of prediction results
            output_path: Path to save the results
        """
        results_df = pd.DataFrame([result.to_dict() for result in results])
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")


# Convenience functions for quick usage
def quick_predict(text: str, model_path: str, preprocessor_path: str) -> PredictionResult:
    """
    Quick prediction function for single text.
    
    Args:
        text: Input text
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
        
    Returns:
        Prediction result
    """
    predictor = FakeNewsPredictor(model_path, preprocessor_path)
    return predictor.predict_single(text)


def batch_predict(texts: List[str], model_path: str, preprocessor_path: str, 
                  batch_size: int = 32) -> List[PredictionResult]:
    """
    Quick batch prediction function.
    
    Args:
        texts: List of input texts
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
        batch_size: Batch size for processing
        
    Returns:
        List of prediction results
    """
    predictor = FakeNewsPredictor(model_path, preprocessor_path)
    return predictor.predict_batch(texts, batch_size)


if __name__ == "__main__":
    # Example usage
    predictor = FakeNewsPredictor()
    
    # This would be used with actual model and preprocessor
    # result = predictor.predict_single("This is a sample news article")
    # print(result.to_dict())
    
    print("Prediction module ready!")

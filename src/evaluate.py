"""
Evaluation Module for Fake News Detection System

This module provides comprehensive evaluation metrics and analysis
for the fake news detection model.

Features:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- ROC Curve and AUC
- Per-category performance analysis
- Bias and fairness analysis
- Detailed classification reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluator for fake news detection models.
    """
    
    def __init__(self, label_encoder=None):
        """
        Initialize the evaluator.
        
        Args:
            label_encoder: Fitted label encoder for subject categories
        """
        self.label_encoder = label_encoder
        self.results = {}
        
    def evaluate_model(self,
                      model: tf.keras.Model,
                      X_test: np.ndarray,
                      y_f_test: np.ndarray,
                      y_s_test: np.ndarray,
                      save_plots: bool = True,
                      plot_dir: str = "plots") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained Keras model
            X_test: Test sequences
            y_f_test: Test fake news labels
            y_s_test: Test subject labels
            save_plots: Whether to save evaluation plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        
        # Get predictions
        y_pred_f_prob, y_pred_s_prob = model.predict(X_test)
        y_pred_f = (y_pred_f_prob > 0.5).astype(int).flatten()
        y_pred_s = np.argmax(y_pred_s_prob, axis=1)
        
        # Evaluate fake news detection
        fake_results = self._evaluate_fake_news(
            y_f_test, y_pred_f, y_pred_f_prob, save_plots, plot_dir
        )
        
        # Evaluate subject classification
        subject_results = self._evaluate_subject_classification(
            y_s_test, y_pred_s, y_pred_s_prob, save_plots, plot_dir
        )
        
        # Combined metrics
        combined_results = self._evaluate_combined_performance(
            y_f_test, y_pred_f, y_s_test, y_pred_s, save_plots, plot_dir
        )
        
        # Bias and fairness analysis
        fairness_results = self._analyze_fairness(
            y_f_test, y_pred_f, y_s_test, save_plots, plot_dir
        )
        
        # Model complexity analysis
        complexity_results = self._analyze_model_complexity(model)
        
        # Compile all results
        self.results = {
            'fake_news_detection': fake_results,
            'subject_classification': subject_results,
            'combined_performance': combined_results,
            'fairness_analysis': fairness_results,
            'model_complexity': complexity_results,
            'summary': self._create_summary(fake_results, subject_results)
        }
        
        logger.info("Model evaluation completed successfully")
        return self.results
    
    def _evaluate_fake_news(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_pred_prob: np.ndarray,
                           save_plots: bool,
                           plot_dir: str) -> Dict[str, Any]:
        """Evaluate fake news detection performance."""
        logger.info("Evaluating fake news detection performance")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Detailed classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['Fake', 'True'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = average_precision_score(y_true, y_pred_prob)
        
        # Create plots if requested
        if save_plots:
            self._plot_confusion_matrix(cm, ['Fake', 'True'], plot_dir, 'fake_news_cm.png')
            self._plot_roc_curve(fpr, tpr, roc_auc, plot_dir, 'fake_news_roc.png')
            self._plot_precision_recall_curve(precision_curve, recall_curve, pr_auc, plot_dir, 'fake_news_pr.png')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr_curve': {'precision': precision_curve.tolist(), 'recall': recall_curve.tolist()}
        }
        
        logger.info(f"Fake News Detection - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        return results
    
    def _evaluate_subject_classification(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_pred_prob: np.ndarray,
                                       save_plots: bool,
                                       plot_dir: str) -> Dict[str, Any]:
        """Evaluate subject classification performance."""
        logger.info("Evaluating subject classification performance")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Class names
        if self.label_encoder:
            class_names = self.label_encoder.classes_.tolist()
        else:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        
        # Detailed classification report
        class_report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(class_report) - 3:  # Exclude macro avg, weighted avg, accuracy
                per_class_metrics[class_name] = {
                    'precision': class_report[class_name]['precision'],
                    'recall': class_report[class_name]['recall'],
                    'f1_score': class_report[class_name]['f1-score'],
                    'support': class_report[class_name]['support']
                }
        
        # Create plots if requested
        if save_plots:
            self._plot_confusion_matrix(cm, class_names, plot_dir, 'subject_cm.png', figsize=(12, 10))
            self._plot_per_class_metrics(per_class_metrics, plot_dir, 'subject_metrics.png')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'per_class_metrics': per_class_metrics,
            'class_names': class_names
        }
        
        logger.info(f"Subject Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results
    
    def _evaluate_combined_performance(self,
                                     y_f_true: np.ndarray,
                                     y_f_pred: np.ndarray,
                                     y_s_true: np.ndarray,
                                     y_s_pred: np.ndarray,
                                     save_plots: bool,
                                     plot_dir: str) -> Dict[str, Any]:
        """Evaluate combined performance across both tasks."""
        logger.info("Evaluating combined performance")
        
        # Overall accuracy (both tasks correct)
        both_correct = (y_f_true == y_f_pred) & (y_s_true == y_s_pred)
        combined_accuracy = np.mean(both_correct)
        
        # Task-specific accuracy
        fake_accuracy = np.mean(y_f_true == y_f_pred)
        subject_accuracy = np.mean(y_s_true == y_s_pred)
        
        # Error analysis
        fake_errors = y_f_true != y_f_pred
        subject_errors = y_s_true != y_s_pred
        both_errors = fake_errors & subject_errors
        
        error_analysis = {
            'fake_news_errors': np.sum(fake_errors),
            'subject_errors': np.sum(subject_errors),
            'both_tasks_errors': np.sum(both_errors),
            'only_fake_errors': np.sum(fake_errors & ~subject_errors),
            'only_subject_errors': np.sum(subject_errors & ~fake_errors),
            'total_samples': len(y_f_true)
        }
        
        # Create error analysis plot if requested
        if save_plots:
            self._plot_error_analysis(error_analysis, plot_dir, 'error_analysis.png')
        
        results = {
            'combined_accuracy': combined_accuracy,
            'fake_news_accuracy': fake_accuracy,
            'subject_accuracy': subject_accuracy,
            'error_analysis': error_analysis
        }
        
        logger.info(f"Combined Performance - Overall Accuracy: {combined_accuracy:.4f}")
        return results
    
    def _analyze_fairness(self,
                         y_f_true: np.ndarray,
                         y_f_pred: np.ndarray,
                         y_s_true: np.ndarray,
                         save_plots: bool,
                         plot_dir: str) -> Dict[str, Any]:
        """Analyze bias and fairness across different categories."""
        logger.info("Analyzing model fairness")
        
        if not self.label_encoder:
            logger.warning("No label encoder provided for fairness analysis")
            return {}
        
        # Get class names
        class_names = self.label_encoder.classes_.tolist()
        
        # Calculate accuracy per subject category
        category_accuracy = {}
        category_precision = {}
        category_recall = {}
        category_f1 = {}
        
        for i, class_name in enumerate(class_names):
            mask = y_s_true == i
            if np.sum(mask) > 0:
                y_true_cat = y_f_true[mask]
                y_pred_cat = y_f_pred[mask]
                
                category_accuracy[class_name] = accuracy_score(y_true_cat, y_pred_cat)
                
                if len(np.unique(y_true_cat)) > 1:  # Only if we have both classes
                    category_precision[class_name] = precision_score(y_true_cat, y_pred_cat, average='binary', zero_division=0)
                    category_recall[class_name] = recall_score(y_true_cat, y_pred_cat, average='binary', zero_division=0)
                    category_f1[class_name] = f1_score(y_true_cat, y_pred_cat, average='binary', zero_division=0)
                else:
                    category_precision[class_name] = 0.0
                    category_recall[class_name] = 0.0
                    category_f1[class_name] = 0.0
        
        # Calculate fairness metrics
        accuracy_values = list(category_accuracy.values())
        fairness_metrics = {
            'accuracy_variance': np.var(accuracy_values),
            'accuracy_range': max(accuracy_values) - min(accuracy_values),
            'accuracy_mean': np.mean(accuracy_values),
            'accuracy_std': np.std(accuracy_values)
        }
        
        # Create fairness plots if requested
        if save_plots:
            self._plot_fairness_analysis(category_accuracy, category_precision, category_recall, plot_dir, 'fairness_analysis.png')
        
        results = {
            'per_category_accuracy': category_accuracy,
            'per_category_precision': category_precision,
            'per_category_recall': category_recall,
            'per_category_f1': category_f1,
            'fairness_metrics': fairness_metrics,
            'category_names': class_names
        }
        
        logger.info(f"Fairness Analysis - Accuracy Range: {fairness_metrics['accuracy_range']:.4f}")
        return results
    
    def _analyze_model_complexity(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Analyze model complexity."""
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Calculate model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        results = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'model_size_mb': model_size_mb,
            'num_layers': len(model.layers)
        }
        
        logger.info(f"Model Complexity - Total Params: {total_params:,}, Size: {model_size_mb:.2f} MB")
        return results
    
    def _create_summary(self, fake_results: Dict, subject_results: Dict) -> Dict[str, Any]:
        """Create a summary of key metrics."""
        return {
            'fake_news_detection': {
                'accuracy': fake_results['accuracy'],
                'f1_score': fake_results['f1_score'],
                'roc_auc': fake_results['roc_auc']
            },
            'subject_classification': {
                'accuracy': subject_results['accuracy'],
                'f1_score': subject_results['f1_score']
            }
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], plot_dir: str, filename: str, figsize=(8, 6)):
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, plot_dir: str, filename: str):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray, pr_auc: float, plot_dir: str, filename: str):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, metrics: Dict[str, Dict], plot_dir: str, filename: str):
        """Plot per-class metrics."""
        classes = list(metrics.keys())
        precision = [metrics[c]['precision'] for c in classes]
        recall = [metrics[c]['recall'] for c in classes]
        f1 = [metrics[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Categories')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self, error_analysis: Dict, plot_dir: str, filename: str):
        """Plot error analysis."""
        labels = ['Fake News Errors', 'Subject Errors', 'Both Tasks Errors', 
                  'Only Fake Errors', 'Only Subject Errors']
        values = [error_analysis['fake_news_errors'], error_analysis['subject_errors'],
                  error_analysis['both_tasks_errors'], error_analysis['only_fake_errors'],
                  error_analysis['only_subject_errors']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, alpha=0.7)
        plt.xlabel('Error Type')
        plt.ylabel('Number of Errors')
        plt.title('Error Analysis')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fairness_analysis(self, accuracy: Dict, precision: Dict, recall: Dict, plot_dir: str, filename: str):
        """Plot fairness analysis."""
        categories = list(accuracy.keys())
        acc_values = list(accuracy.values())
        prec_values = list(precision.values())
        rec_values = list(recall.values())
        
        x = np.arange(len(categories))
        width = 0.25
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - width, acc_values, width, label='Accuracy', alpha=0.8)
        plt.bar(x, prec_values, width, label='Precision', alpha=0.8)
        plt.bar(x + width, rec_values, width, label='Recall', alpha=0.8)
        
        plt.xlabel('News Categories')
        plt.ylabel('Score')
        plt.title('Fairness Analysis: Performance Across Categories')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, save_path: str = "evaluation_report.txt"):
        """Generate a comprehensive evaluation report."""
        if not self.results:
            logger.warning("No evaluation results available. Run evaluate_model first.")
            return
        
        with open(save_path, 'w') as f:
            f.write("FAKE NEWS DETECTION MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = self.results['summary']
            f.write(f"Fake News Detection:\n")
            f.write(f"  Accuracy: {summary['fake_news_detection']['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {summary['fake_news_detection']['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {summary['fake_news_detection']['roc_auc']:.4f}\n\n")
            
            f.write(f"Subject Classification:\n")
            f.write(f"  Accuracy: {summary['subject_classification']['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {summary['subject_classification']['f1_score']:.4f}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            
            # Fake news detection details
            fake_results = self.results['fake_news_detection']
            f.write(f"Fake News Detection Details:\n")
            f.write(f"  Precision: {fake_results['precision']:.4f}\n")
            f.write(f"  Recall: {fake_results['recall']:.4f}\n")
            f.write(f"  PR-AUC: {fake_results['pr_auc']:.4f}\n\n")
            
            # Subject classification details
            subject_results = self.results['subject_classification']
            f.write(f"Subject Classification Details:\n")
            f.write(f"  Precision: {subject_results['precision']:.4f}\n")
            f.write(f"  Recall: {subject_results['recall']:.4f}\n\n")
            
            # Combined performance
            combined = self.results['combined_performance']
            f.write(f"Combined Performance:\n")
            f.write(f"  Both Tasks Correct: {combined['combined_accuracy']:.4f}\n\n")
            
            # Fairness analysis
            if self.results.get('fairness_analysis'):
                fairness = self.results['fairness_analysis']
                f.write(f"Fairness Analysis:\n")
                f.write(f"  Accuracy Range: {fairness['fairness_metrics']['accuracy_range']:.4f}\n")
                f.write(f"  Accuracy Std: {fairness['fairness_metrics']['accuracy_std']:.4f}\n\n")
            
            # Model complexity
            complexity = self.results['model_complexity']
            f.write(f"Model Complexity:\n")
            f.write(f"  Total Parameters: {complexity['total_parameters']:,}\n")
            f.write(f"  Model Size: {complexity['model_size_mb']:.2f} MB\n")
        
        logger.info(f"Evaluation report saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    print("Evaluation module ready!")

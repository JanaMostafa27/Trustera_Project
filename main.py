"""
Main Entry Point for Fake News Detection System

This script provides a command-line interface for training, evaluating,
and running inference with the fake news detection model.

Usage:
    python main.py train --data data/dataset.csv --model models/fake_news_model.h5
    python main.py evaluate --model models/fake_news_model.h5 --data data/test.csv
    python main.py predict --model models/fake_news_model.h5 --text "Your news text here"
    python main.py api --model models/fake_news_model.h5 --port 8000
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_preprocessing import DataPreprocessor
from src.train import PyTorchTrainer
from src.evaluate import ModelEvaluator
from src.predict import FakeNewsPredictor


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ["data", "models", "plots", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def train_model(args):
    """Train the fake news detection model."""
    print("🚀 Starting model training...")
    
    # Setup directories
    setup_directories()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        max_words=args.max_words,
        max_len=args.max_len,
        random_state=args.random_state
    )
    
    # Load and preprocess data
    print(f"📊 Loading data from {args.data}")
    splits = preprocessor.fit_transform(args.data)
    
    # Get vocabulary info
    vocab_info = preprocessor.get_vocabulary_info()
    print(f"📚 Vocabulary info: {vocab_info}")
    
    # Save preprocessor
    preprocessor_path = args.model.replace('.pt', '_preprocessor.pkl')
    preprocessor.save_preprocessor(preprocessor_path)
    
    # Initialize trainer
    trainer = PyTorchTrainer(
        vocab_size=vocab_info['vocab_size'],
        num_subjects=vocab_info['num_subjects'],
        embedding_dim=args.embedding_dim,
        random_state=args.random_state
    )
    
    # Hyperparameters
    hyperparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate
    }
    
    # Train model
    print("🎯 Training model...")
    metrics = trainer.train(
        X_train=splits['X_train'],
        y_f_train=splits['y_f_train'],
        y_s_train=splits['y_s_train'],
        X_val=splits['X_val'],
        y_f_val=splits['y_f_val'],
        y_s_val=splits['y_s_val'],
        model_save_path=args.model,
        use_class_weights=args.use_class_weights,
        hyperparams=hyperparams
    )
    
    print("✅ Training completed successfully!")
    print(f"📈 Final fake news accuracy: {metrics['val_fake_acc']:.4f}")
    print(f"📈 Final subject accuracy: {metrics['val_subject_acc']:.4f}")
    print(f" Model saved to: {args.model}")
    print(f" Preprocessor saved to: {preprocessor_path}")
    
    # Print final metrics
    print("\n Final Metrics:")
    print(f"  Best Validation Fake News Accuracy: {metrics['best_val_fake_acc']:.4f}")
    print(f"  Best Validation Subject Accuracy: {metrics['best_val_subject_acc']:.4f}")


def evaluate_model(args):
    """Evaluate the trained model."""
    print("🔍 Starting model evaluation...")
    
    # Setup directories
    setup_directories()
    
    # Load preprocessor
    preprocessor_path = args.model.replace('.pt', '_preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        print(f"❌ Preprocessor not found at {preprocessor_path}")
        return
    
    preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)
    
    # Load and preprocess test data
    print(f"📊 Loading test data from {args.data}")
    splits = preprocessor.fit_transform(args.data)
    
    # Load model
    trainer = PyTorchTrainer(vocab_size=10000, num_subjects=8)  # Default values, will be updated when loading
    trainer.load_model(args.model)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(label_encoder=preprocessor.label_encoder)
    
    # Evaluate model
    print("🎯 Evaluating model...")
    results = evaluator.evaluate_model(
        model=trainer.model,
        X_test=splits['X_test'],
        y_f_test=splits['y_f_test'],
        y_s_test=splits['y_s_test'],
        save_plots=True,
        plot_dir="plots"
    )
    
    # Generate report
    report_path = "evaluation_report.txt"
    evaluator.generate_report(report_path)
    
    print(f"✅ Model evaluation completed!")
    print(f"📁 Plots saved to: plots/")
    print(f"📁 Report saved to: {report_path}")
    
    # Print summary
    summary = results['summary']
    print("\n📈 Evaluation Summary:")
    print(f"  Fake News Detection:")
    print(f"    Accuracy: {summary['fake_news_detection']['accuracy']:.4f}")
    print(f"    F1-Score: {summary['fake_news_detection']['f1_score']:.4f}")
    print(f"    ROC-AUC: {summary['fake_news_detection']['roc_auc']:.4f}")
    print(f"  Subject Classification:")
    print(f"    Accuracy: {summary['subject_classification']['accuracy']:.4f}")
    print(f"    F1-Score: {summary['subject_classification']['f1_score']:.4f}")


def predict_text(args):
    """Make prediction on a single text."""
    print("🔮 Making prediction...")
    
    # Load model and preprocessor
    preprocessor_path = args.model.replace('.pt', '_preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        print(f"❌ Preprocessor not found at {preprocessor_path}")
        return
    
    predictor = FakeNewsPredictor(args.model, preprocessor_path)
    
    # Make prediction
    result = predictor.predict_single(args.text, return_probabilities=True)
    
    if isinstance(result, dict):
        # Detailed results
        prediction_result = result['result']
        print(f"Prediction: {prediction_result['prediction']}")
        print(f"Category: {prediction_result['category']}")
    else:
        # Simple result
        print(f"Prediction: {result.prediction}")
        print(f"Category: {result.category}")
    
    

def predict_file(args):
    """Make predictions on texts from a file."""
    print(" Processing file predictions...")
    
    # Load model and preprocessor
    preprocessor_path = args.model.replace('.pt', '_preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        print(f"❌ Preprocessor not found at {preprocessor_path}")
        return
    
    predictor = FakeNewsPredictor(args.model, preprocessor_path)
    
    # Make predictions
    output_path = args.output or "predictions.csv"
    results_df = predictor.predict_from_file(
        file_path=args.data,
        text_column=args.text_column,
        output_path=output_path
    )
    
    print(f"✅ File processing completed!")
    print(f"📁 Results saved to: {output_path}")
    
    # Print summary statistics
    predictions = results_df['prediction'].value_counts()
    print(f"📊 Summary:")
    print(f"  Total texts processed: {len(results_df)}")
    print(f"  Fake news: {len(results_df[results_df['prediction'] == 'FAKE'])}")
    print(f"  True news: {len(results_df[results_df['prediction'] == 'TRUE'])}")


def start_api(args):
    """Start the FastAPI server."""
    print("🌐 Starting API server...")
    
    # Import here to avoid dependency issues if not needed
    try:
        import uvicorn
        from api.main import app
    except ImportError:
        print("❌ FastAPI and uvicorn are required for API mode. Install with: pip install fastapi uvicorn")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found at {args.model}")
        return
    
    # Set model path environment variable for API
    os.environ['MODEL_PATH'] = args.model
    
    preprocessor_path = args.model.replace('.pt', '_preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        os.environ['PREPROCESSOR_PATH'] = preprocessor_path
    
    print(f"🚀 Starting server on http://localhost:{args.port}")
    print(f"📖 API documentation: http://localhost:{args.port}/docs")
    print(f"📊 Model: {args.model}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to training data CSV")
    train_parser.add_argument("--model", default="models/fake_news_model.pt", help="Path to save model")
    train_parser.add_argument("--max-words", type=int, default=10000, help="Maximum vocabulary size")
    train_parser.add_argument("--max-len", type=int, default=300, help="Maximum sequence length")
    train_parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate")
    train_parser.add_argument("--l2-reg", type=float, default=0.01, help="L2 regularization")
    train_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    train_parser.add_argument("--use-class-weights", action="store_true", help="Use class weights")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--data", required=True, help="Path to test data CSV")
    
    # Predict command (single text)
    predict_parser = subparsers.add_parser("predict", help="Predict on single text")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--text", required=True, help="Text to classify")
    
    # Predict file command
    file_parser = subparsers.add_parser("predict-file", help="Predict on file")
    file_parser.add_argument("--model", required=True, help="Path to trained model")
    file_parser.add_argument("--data", required=True, help="Path to input CSV file")
    file_parser.add_argument("--text-column", default="text", help="Column name containing text")
    file_parser.add_argument("--output", help="Output CSV file path")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--model", required=True, help="Path to trained model")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "predict":
        predict_text(args)
    elif args.command == "predict-file":
        predict_file(args)
    elif args.command == "api":
        start_api(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

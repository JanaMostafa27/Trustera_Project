"""
PyTorch Training Module for Fake News Detection System

This module handles model training using PyTorch instead of TensorFlow,
with improved architecture and training procedures.

Features:
- Enhanced CNN-LSTM architecture with attention mechanisms
- Better hyperparameter management
- Early stopping and learning rate scheduling
- Comprehensive logging and checkpointing
- Class imbalance handling
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple
from sklearn.utils.class_weight import compute_class_weight
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection"""
    
    def __init__(self, X: np.ndarray, y_fake: np.ndarray, y_subject: np.ndarray):
        self.X = torch.LongTensor(X)
        self.y_fake = torch.FloatTensor(y_fake)
        self.y_subject = torch.LongTensor(y_subject)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_fake[idx], self.y_subject[idx]


class AttentionLayer(nn.Module):
    """Attention mechanism for better focus on important features"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: (batch_size, hidden_size * 2)
        
        return context_vector, attention_weights


class FakeNewsModel(nn.Module):
    """
    Enhanced CNN-LSTM model with attention mechanism using PyTorch.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 lstm_units: int = 64,
                 cnn_filters: int = 128,
                 cnn_kernel_size: int = 5,
                 num_subjects: int = 8,
                 dropout_rate: float = 0.5,
                 use_attention: bool = True):
        super(FakeNewsModel, self).__init__()
        
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(embedding_dim, cnn_filters, cnn_kernel_size, padding=cnn_kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(cnn_filters // 2, lstm_units, batch_first=True, 
                           bidirectional=True, dropout=0.3)
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(lstm_units * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers
        self.shared_dense = nn.Linear(lstm_units * 2, 128)
        self.fake_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.subject_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_subjects)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # CNN layers (need to transpose for Conv1d)
        embedded_conv = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        conv1_out = torch.relu(self.bn1(self.conv1(embedded_conv)))
        conv1_out = nn.functional.max_pool1d(conv1_out, 2)  # (batch_size, cnn_filters, seq_len//2)
        
        conv2_out = torch.relu(self.bn2(self.conv2(conv1_out)))
        conv2_out = nn.functional.max_pool1d(conv2_out, 2)  # (batch_size, cnn_filters//2, seq_len//4)
        
        # Transpose back for LSTM
        conv_out = conv2_out.transpose(1, 2)  # (batch_size, seq_len//4, cnn_filters//2)
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)  # (batch_size, seq_len//4, hidden_size * 2)
        
        # Attention or global max pooling
        if self.use_attention:
            attended, _ = self.attention(lstm_out)  # (batch_size, hidden_size * 2)
            features = attended
        else:
            # Global max pooling
            features = torch.max(lstm_out, dim=1)[0]  # (batch_size, hidden_size * 2)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Shared dense layer
        shared_features = torch.relu(self.shared_dense(features))
        shared_features = self.dropout(shared_features)
        
        # Output heads
        fake_output = self.fake_classifier(shared_features).squeeze(1)  # (batch_size,)
        subject_output = self.subject_classifier(shared_features)  # (batch_size, num_subjects)
        
        return fake_output, subject_output


class PyTorchTrainer:
    """
    Enhanced trainer for fake news detection using PyTorch.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 num_subjects: int,
                 embedding_dim: int = 128,
                 lstm_units: int = 64,
                 cnn_filters: int = 128,
                 cnn_kernel_size: int = 5,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize the PyTorch trainer.
        """
        self.vocab_size = vocab_size
        self.num_subjects = num_subjects
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds
        set_random_seeds(random_state)
        
        # Model components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = None
        
        # Training parameters
        self.epochs = 50
        self.batch_size = 64
        self.patience = 5
        
        logger.info(f"Using device: {self.device}")
    
    def build_model(self, use_attention: bool = True):
        """Build the PyTorch model"""
        self.model = FakeNewsModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            num_subjects=self.num_subjects,
            dropout_rate=self.dropout_rate,
            use_attention=use_attention
        )
        
        self.model.to(self.device)
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def compute_class_weights(self, y_fake: np.ndarray) -> torch.Tensor:
        """Compute class weights to handle imbalance"""
        classes = np.unique(y_fake)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_fake)
        
        class_weights = torch.FloatTensor(weights).to(self.device)
        self.class_weights = class_weights
        
        logger.info(f"Class weights computed: {dict(zip(classes, weights))}")
        return class_weights
    
    def train(self,
              X_train: np.ndarray,
              y_f_train: np.ndarray,
              y_s_train: np.ndarray,
              X_val: np.ndarray,
              y_f_val: np.ndarray,
              y_s_val: np.ndarray,
              model_save_path: str,
              use_class_weights: bool = True,
              hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train the model"""
        logger.info("Starting PyTorch model training")
        
        # Update hyperparameters if provided
        if hyperparams:
            for key, value in hyperparams.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.info(f"Updated {key} to {value}")
        
        # Build model
        self.build_model()
        
        # Compute class weights
        if use_class_weights:
            class_weights = self.compute_class_weights(y_f_train)
        else:
            class_weights = None
        
        # Create datasets and dataloaders
        train_dataset = FakeNewsDataset(X_train, y_f_train, y_s_train)
        val_dataset = FakeNewsDataset(X_val, y_f_val, y_s_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss functions and optimizer
        criterion_fake = nn.BCELoss(weight=class_weights[1] if class_weights is not None else None)
        criterion_subject = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = {'train_loss': [], 'val_loss': [], 'train_fake_acc': [], 'val_fake_acc': [], 
                       'train_subject_acc': [], 'val_subject_acc': []}
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_fake_correct = 0
            train_subject_correct = 0
            train_total = 0
            
            for batch_x, batch_y_fake, batch_y_subject in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y_fake = batch_y_fake.to(self.device)
                batch_y_subject = batch_y_subject.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                fake_pred, subject_pred = self.model(batch_x)
                
                # Calculate losses
                loss_fake = criterion_fake(fake_pred, batch_y_fake)
                loss_subject = criterion_subject(subject_pred, batch_y_subject)
                
                # Combined loss (weighted)
                total_loss = loss_fake + 0.5 * loss_subject
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += total_loss.item()
                train_fake_correct += ((fake_pred > 0.5).float() == batch_y_fake).sum().item()
                train_subject_correct += (subject_pred.argmax(dim=1) == batch_y_subject).sum().item()
                train_total += batch_x.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_fake_correct = 0
            val_subject_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y_fake, batch_y_subject in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y_fake = batch_y_fake.to(self.device)
                    batch_y_subject = batch_y_subject.to(self.device)
                    
                    # Forward pass
                    fake_pred, subject_pred = self.model(batch_x)
                    
                    # Calculate losses
                    loss_fake = criterion_fake(fake_pred, batch_y_fake)
                    loss_subject = criterion_subject(subject_pred, batch_y_subject)
                    
                    # Combined loss
                    total_loss = loss_fake + 0.5 * loss_subject
                    
                    # Metrics
                    val_loss += total_loss.item()
                    val_fake_correct += ((fake_pred > 0.5).float() == batch_y_fake).sum().item()
                    val_subject_correct += (subject_pred.argmax(dim=1) == batch_y_subject).sum().item()
                    val_total += batch_x.size(0)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_fake_acc = train_fake_correct / train_total
            train_subject_acc = train_subject_correct / train_total
            val_fake_acc = val_fake_correct / val_total
            val_subject_acc = val_subject_correct / val_total
            
            # Update history
            train_history['train_loss'].append(train_loss)
            train_history['val_loss'].append(val_loss)
            train_history['train_fake_acc'].append(train_fake_acc)
            train_history['val_fake_acc'].append(val_fake_acc)
            train_history['train_subject_acc'].append(train_subject_acc)
            train_history['val_subject_acc'].append(val_subject_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'vocab_size': self.vocab_size,
                    'num_subjects': self.num_subjects,
                    'hyperparams': {
                        'embedding_dim': self.embedding_dim,
                        'lstm_units': self.lstm_units,
                        'cnn_filters': self.cnn_filters,
                        'dropout_rate': self.dropout_rate,
                        'learning_rate': self.learning_rate
                    }
                }, model_save_path)
            else:
                patience_counter += 1
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                       f"Train Fake Acc: {train_fake_acc:.4f}, Val Fake Acc: {val_fake_acc:.4f} - "
                       f"Train Subject Acc: {train_subject_acc:.4f}, Val Subject Acc: {val_subject_acc:.4f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load(model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final metrics
        final_metrics = {
            'train_loss': train_history['train_loss'][-1],
            'val_loss': train_history['val_loss'][-1],
            'train_fake_acc': train_history['train_fake_acc'][-1],
            'val_fake_acc': train_history['val_fake_acc'][-1],
            'train_subject_acc': train_history['train_subject_acc'][-1],
            'val_subject_acc': train_history['val_subject_acc'][-1],
            'best_val_loss': best_val_loss,
            'best_val_fake_acc': max(train_history['val_fake_acc']),
            'best_val_subject_acc': max(train_history['val_subject_acc']),
            'history': train_history
        }
        
        logger.info("PyTorch training completed successfully")
        logger.info(f"Best validation fake news accuracy: {final_metrics['best_val_fake_acc']:.4f}")
        logger.info(f"Best validation subject accuracy: {final_metrics['best_val_subject_acc']:.4f}")
        
        return final_metrics
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Rebuild model with saved hyperparameters
        hyperparams = checkpoint.get('hyperparams', {})
        self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        self.model.eval()
        
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            fake_output, subject_output = self.model(X_tensor)
            
            fake_prob = torch.sigmoid(fake_output).cpu().numpy()
            subject_prob = torch.softmax(subject_output, dim=1).cpu().numpy()
            
        return fake_prob, subject_prob
    
    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on batch data"""
        return self.predict(X)


if __name__ == "__main__":
    # Example usage
    trainer = PyTorchTrainer(vocab_size=10000, num_subjects=8)
    print("PyTorch training module ready!")

#!/usr/bin/env python3
"""
Ablation Study: Comparing DANN vs Baseline Models
==================================================

This script implements all models for comparison:
    1. SVM - Support Vector Machine with Linear kernel on flattened spectral features
    2. CNN - Same architecture as DANN but without domain adversarial training
    3. CNN-LSTM - CNN feature extractor + LSTM for temporal modeling
    4. DANN - Domain Adversarial Neural Network (the proposed model)

Usage:
    python ablation_study.py --data_dir "python_data_best 13-04-03-643" --model all
    python ablation_study.py --data_dir "python_data_best 13-04-03-643" --model svm
    python ablation_study.py --data_dir "python_data_best 13-04-03-643" --model cnn
    python ablation_study.py --data_dir "python_data_best 13-04-03-643" --model cnn_lstm
    python ablation_study.py --data_dir "python_data_best 13-04-03-643" --model dann
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.io import savemat
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
import argparse
from datetime import datetime
import random

from utils import load_mat_file, prepare_fold_data

warnings.filterwarnings('ignore')

# ============================================================================
# COMPREHENSIVE SEED SETTING FOR FULL REPRODUCIBILITY
# ============================================================================
SEED = 42

def set_all_seeds(seed):
    """Set all random seeds for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds at module load time
set_all_seeds(SEED)

# ============================================================================
# Dataset Classes
# ============================================================================

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data (without subject labels)"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y - 1)  # Convert to 0-based
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EEGDatasetWithSubject(Dataset):
    """PyTorch Dataset for EEG data with subject labels (for DANN)"""
    def __init__(self, X, y_drowsiness, y_subject):
        self.X = torch.FloatTensor(X)
        self.y_drowsiness = torch.LongTensor(y_drowsiness - 1)  # Convert to 0-based
        self.y_subject = torch.LongTensor(y_subject - 1)  # Convert to 0-based
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_drowsiness[idx], self.y_subject[idx]

# ============================================================================
# Model 1: SVM (Support Vector Machine)
# ============================================================================

class SVMClassifier:
    """SVM classifier for EEG drowsiness detection
    
    NOTE: Expects pre-normalized data (no internal StandardScaler to avoid double normalization)
    """
    
    def __init__(self, kernel='linear', C=1.0, gamma='scale'):
        # C=1.0 is standard default (was 10.0 which could cause overfitting)
        # No StandardScaler - data should be pre-normalized like CNN/DANN
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=SEED)
        
    def fit(self, X_train, y_train):
        # Flatten the spectral features (data already normalized)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_flat, y_train)
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# ============================================================================
# Model 2: CNN (without Domain Adversarial Training)
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for EEG classification (same architecture as DANN feature extractor)
    but without the domain adversarial branch
    """
    
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        # Feature extractor (same as DANN)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # Calculate feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            dummy_features = self.features(dummy_input)
            self.feature_dim = dummy_features.numel()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# ============================================================================
# Model 3: CNN-LSTM
# ============================================================================

class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for EEG classification
    Uses CNN for spatial feature extraction and LSTM for temporal modeling
    """
    
    def __init__(self, input_shape, num_classes, hidden_size=64, num_layers=2):
        super(CNNLSTM, self).__init__()
        
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        # CNN feature extractor (processes each frequency band)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # Calculate CNN output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            cnn_out = self.cnn(dummy_input)
            self.cnn_out_freq = cnn_out.shape[2]
            self.cnn_out_channels = cnn_out.shape[1] * cnn_out.shape[3]
        
        # LSTM for temporal modeling (treat frequency as sequence)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch, channels, freq, electrodes)
        
        # Reshape for LSTM: (batch, seq_len, features)
        batch_size = cnn_features.size(0)
        # Treat frequency dimension as sequence
        cnn_features = cnn_features.permute(0, 2, 1, 3)  # (batch, freq, channels, electrodes)
        cnn_features = cnn_features.reshape(batch_size, self.cnn_out_freq, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use last hidden state from both directions
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        # Classification
        output = self.classifier(h_combined)
        return output

# ============================================================================
# Model 4: DANN (Domain Adversarial Neural Network)
# ============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training
    Forward: identity function
    Backward: multiply gradients by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

class CosineClassifier(nn.Module):
    """
    Weight-normalized cosine classifier with learnable temperature (scale).
    """
    def __init__(self, input_dim, num_classes, init_scale=16.0):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * torch.matmul(x_norm, w_norm.t())
        return logits

class DomainAdversarialCNN(nn.Module):
    """
    Domain Adversarial CNN for EEG Drowsiness Detection
    
    Architecture:
    - Shared feature extractor (3 conv blocks)
    - Drowsiness classifier branch
    - Subject classifier branch (with gradient reversal)
    """
    
    def __init__(self, input_shape, num_classes, num_subjects):
        super(DomainAdversarialCNN, self).__init__()
        
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        # Shared feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # Calculate feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            dummy_features = self.features(dummy_input)
            self.feature_dim = dummy_features.numel()
        
        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.shared_norm = nn.LayerNorm(128)
        
        # Cosine classifier heads
        self.drowsiness_classifier = CosineClassifier(128, num_classes)
        self.subject_classifier = CosineClassifier(128, num_subjects)
        
    def forward(self, x, lambda_=1.0):
        # Shared feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)
        shared_features = self.shared_fc(features)
        shared_features = self.shared_norm(shared_features)
        
        # Drowsiness prediction (normal forward)
        drowsiness_pred = self.drowsiness_classifier(shared_features)
        
        # Subject prediction (with gradient reversal)
        reversed_features = gradient_reversal(shared_features, lambda_)
        subject_pred = self.subject_classifier(reversed_features)
        
        return drowsiness_pred, subject_pred

# ============================================================================
# Training Functions
# ============================================================================

def train_pytorch_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, model_name='model'):
    """Train a PyTorch model (for CNN and CNN-LSTM)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

def train_dann_model(model, train_loader, val_loader, device, num_epochs=150, lr=0.0005):
    """
    Train the DANN model with TUNED domain adversarial training
    
    Key changes from aggressive version:
    1. Much lower domain weight (0.05-0.15 instead of 0.2-0.8)
    2. Slower lambda ramp (gentler gradient reversal)
    3. Longer patience for early stopping (25 instead of 15)
    4. Higher learning rate with longer training
    5. Focus on drowsiness task first, then gradually add domain adaptation
    """
    # Loss functions - reduced label smoothing
    drowsiness_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    subject_criterion = nn.CrossEntropyLoss()
    
    # Optimizer - slightly higher LR, less weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    warmup_epochs = max(1, int(0.15 * num_epochs))  # Longer warmup
    cosine_epochs = max(1, num_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr * 0.05)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_drowsiness_loss = 0.0
        epoch_subject_loss = 0.0
        
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * warmup_factor
        else:
            scheduler.step()

        # TUNED: Much gentler lambda schedule - starts at 0, slowly ramps to 0.5 max
        # Original: lambda_ = 2.0 / (1.0 + np.exp(-25 * epoch / num_epochs)) - 1.0
        # This was too aggressive (reaches 0.73 at epoch 20, 0.96 at epoch 40)
        # New: slower sigmoid, capped at 0.5
        progress = epoch / num_epochs
        lambda_ = 0.5 * (2.0 / (1.0 + np.exp(-5 * progress)) - 1.0)  # Max 0.5, slower ramp

        # TUNED: Much lower domain weight (0.05 to 0.15)
        # Original: 0.2 to 0.8 - way too aggressive, destroyed task features
        min_w, max_w, ramp_frac = 0.05, 0.15, 0.5  # Ramp over 50% of training
        ramp = min(1.0, progress / ramp_frac)
        domain_weight = min_w + (max_w - min_w) * ramp
        
        for batch_idx, (data, drowsiness_labels, subject_labels) in enumerate(train_loader):
            data = data.to(device)
            drowsiness_labels = drowsiness_labels.to(device)
            subject_labels = subject_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            drowsiness_pred, subject_pred = model(data, lambda_)
            
            # Calculate losses
            drowsiness_loss = drowsiness_criterion(drowsiness_pred, drowsiness_labels)
            subject_loss = subject_criterion(subject_pred, subject_labels)
            
            # Total loss - drowsiness is primary, domain is secondary regularizer
            total_loss = drowsiness_loss + domain_weight * subject_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_drowsiness_loss += drowsiness_loss.item()
            epoch_subject_loss += subject_loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, drowsiness_labels, _ in val_loader:
                data = data.to(device)
                drowsiness_labels = drowsiness_labels.to(device)
                
                drowsiness_pred, _ = model(data, 0.0)
                _, predicted = torch.max(drowsiness_pred.data, 1)
                val_total += drowsiness_labels.size(0)
                val_correct += (predicted == drowsiness_labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # TUNED: Longer patience (25 instead of 15)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Val Acc: {val_acc:.4f}, Lambda: {lambda_:.3f}, DomW: {domain_weight:.3f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

def evaluate_pytorch_model(model, test_loader, device):
    """Evaluate a PyTorch model (CNN/CNN-LSTM)"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

def evaluate_dann_model(model, test_loader, device):
    """Evaluate the DANN model"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, drowsiness_labels, _ in test_loader:
            data = data.to(device)
            drowsiness_pred, _ = model(data, 0.0)
            _, predicted = torch.max(drowsiness_pred.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(drowsiness_labels.numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

# ============================================================================
# Main Ablation Study
# ============================================================================

def run_ablation_study(data_dir, models_to_run=['svm', 'cnn', 'cnn_lstm', 'dann']):
    """Run ablation study with specified models"""
    
    print("=" * 70)
    print("ABLATION STUDY: Comparing All Models")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Models to evaluate: {models_to_run}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check if data exists
    export_check_file = os.path.join(data_dir, 'export_complete.mat')
    if not os.path.exists(export_check_file):
        print(f"Error: No exported data found in {data_dir}")
        return None, None
    
    # Load metadata
    metadata_file = os.path.join(data_dir, 'metadata.mat')
    try:
        metadata = load_mat_file(metadata_file)
        num_classes = int(metadata['num_classes'])
        num_subjects = int(metadata['num_subjects'])
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None, None
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of subjects: {num_subjects}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all fold files
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    print(f"Found {len(fold_numbers)} folds")
    
    # Results storage - now includes predictions, true labels, and confusion matrix values
    results = {model: {
        'accuracies': [], 
        'fold_nums': [],
        'predictions': [],      # Store predictions per fold
        'true_labels': [],      # Store true labels per fold
        'tn': [], 'fp': [], 'fn': [], 'tp': []  # Confusion matrix values per fold
    } for model in models_to_run}
    
    # Process each fold
    for fold_num in fold_numbers:
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*50}")
        
        # Load fold data
        try:
            fold_data = load_mat_file(os.path.join(data_dir, f'fold_{fold_num}_data.mat'))
        except Exception as e:
            print(f"Error loading fold {fold_num}: {e}")
            continue
        
        # Extract data with consistent shapes
        (X_train_pt, y_train, subject_train,
         X_val_pt, y_val, subject_val,
         X_test_pt, y_test, subject_test) = prepare_fold_data(fold_data)
        
        # Normalize (channel-wise)
        train_mean = X_train_pt.mean(axis=(0, 1, 2), keepdims=True)
        train_std = X_train_pt.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        
        X_train_pt = (X_train_pt - train_mean) / train_std
        X_val_pt = (X_val_pt - train_mean) / train_std
        X_test_pt = (X_test_pt - train_mean) / train_std
        
        print(f"Train: {X_train_pt.shape}, Val: {X_val_pt.shape}, Test: {X_test_pt.shape}")
        
        # =====================================================================
        # Model 1: SVM
        # =====================================================================
        if 'svm' in models_to_run:
            print("\n--- Training SVM ---")
            # C=1.0 is standard default for fair comparison (not C=10.0 which could overfit)
            # Data is already normalized (same as CNN/DANN), no double normalization
            svm = SVMClassifier(kernel='linear', C=1.0, gamma='scale')
            
            # Data already normalized at lines 665-667, same as CNN/DANN
            svm.fit(X_train_pt, y_train)
            svm_predictions = svm.predict(X_test_pt)
            svm_acc = accuracy_score(y_test, svm_predictions)
            
            # Compute confusion matrix values (for binary: TN, FP, FN, TP)
            cm = confusion_matrix(y_test, svm_predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0  # Handle non-binary case
            
            results['svm']['accuracies'].append(svm_acc)
            results['svm']['fold_nums'].append(fold_num)
            results['svm']['predictions'].append(svm_predictions.tolist())
            results['svm']['true_labels'].append(y_test.tolist())
            results['svm']['tn'].append(int(tn))
            results['svm']['fp'].append(int(fp))
            results['svm']['fn'].append(int(fn))
            results['svm']['tp'].append(int(tp))
            print(f"SVM Accuracy: {svm_acc*100:.2f}% (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        
        # =====================================================================
        # Model 2: CNN (without domain adversarial)
        # =====================================================================
        if 'cnn' in models_to_run:
            print("\n--- Training CNN ---")
            
            train_dataset = EEGDataset(X_train_pt, y_train)
            val_dataset = EEGDataset(X_val_pt, y_val)
            test_dataset = EEGDataset(X_test_pt, y_test)
            
            # Use seeded generator for reproducible shuffling
            g = torch.Generator()
            g.manual_seed(SEED)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_shape = (1, X_train_pt.shape[2], X_train_pt.shape[3])
            cnn_model = SimpleCNN(input_shape, num_classes).to(device)
            
            cnn_model, _ = train_pytorch_model(
                cnn_model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, model_name='CNN'
            )
            
            cnn_acc, cnn_preds, cnn_true = evaluate_pytorch_model(cnn_model, test_loader, device)
            
            # Compute confusion matrix values
            cm = confusion_matrix(cnn_true, cnn_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            results['cnn']['accuracies'].append(cnn_acc)
            results['cnn']['fold_nums'].append(fold_num)
            results['cnn']['predictions'].append(cnn_preds)
            results['cnn']['true_labels'].append(cnn_true)
            results['cnn']['tn'].append(int(tn))
            results['cnn']['fp'].append(int(fp))
            results['cnn']['fn'].append(int(fn))
            results['cnn']['tp'].append(int(tp))
            print(f"CNN Accuracy: {cnn_acc*100:.2f}% (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        
        # =====================================================================
        # Model 3: CNN-LSTM
        # =====================================================================
        if 'cnn_lstm' in models_to_run:
            print("\n--- Training CNN-LSTM ---")
            
            train_dataset = EEGDataset(X_train_pt, y_train)
            val_dataset = EEGDataset(X_val_pt, y_val)
            test_dataset = EEGDataset(X_test_pt, y_test)
            
            # Use seeded generator for reproducible shuffling
            g = torch.Generator()
            g.manual_seed(SEED)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_shape = (1, X_train_pt.shape[2], X_train_pt.shape[3])
            lstm_model = CNNLSTM(input_shape, num_classes, hidden_size=64, num_layers=2).to(device)
            
            lstm_model, _ = train_pytorch_model(
                lstm_model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, model_name='CNN-LSTM'
            )
            
            lstm_acc, lstm_preds, lstm_true = evaluate_pytorch_model(lstm_model, test_loader, device)
            
            # Compute confusion matrix values
            cm = confusion_matrix(lstm_true, lstm_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            results['cnn_lstm']['accuracies'].append(lstm_acc)
            results['cnn_lstm']['fold_nums'].append(fold_num)
            results['cnn_lstm']['predictions'].append(lstm_preds)
            results['cnn_lstm']['true_labels'].append(lstm_true)
            results['cnn_lstm']['tn'].append(int(tn))
            results['cnn_lstm']['fp'].append(int(fp))
            results['cnn_lstm']['fn'].append(int(fn))
            results['cnn_lstm']['tp'].append(int(tp))
            print(f"CNN-LSTM Accuracy: {lstm_acc*100:.2f}% (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        
        # =====================================================================
        # Model 4: DANN (Domain Adversarial Neural Network)
        # =====================================================================
        if 'dann' in models_to_run:
            print("\n--- Training DANN ---")
            
            train_dataset = EEGDatasetWithSubject(X_train_pt, y_train, subject_train)
            val_dataset = EEGDatasetWithSubject(X_val_pt, y_val, subject_val)
            test_dataset = EEGDatasetWithSubject(X_test_pt, y_test, subject_test)
            
            # Use seeded generator for reproducible shuffling
            g = torch.Generator()
            g.manual_seed(SEED)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_shape = (1, X_train_pt.shape[2], X_train_pt.shape[3])
            dann_model = DomainAdversarialCNN(input_shape, num_classes, num_subjects).to(device)
            
            dann_model, _ = train_dann_model(
                dann_model, train_loader, val_loader, device,
                num_epochs=120, lr=0.0003
            )
            
            dann_acc, dann_preds, dann_true = evaluate_dann_model(dann_model, test_loader, device)
            
            # Compute confusion matrix values
            cm = confusion_matrix(dann_true, dann_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            results['dann']['accuracies'].append(dann_acc)
            results['dann']['fold_nums'].append(fold_num)
            results['dann']['predictions'].append(dann_preds)
            results['dann']['true_labels'].append(dann_true)
            results['dann']['tn'].append(int(tn))
            results['dann']['fp'].append(int(fp))
            results['dann']['fn'].append(int(fn))
            results['dann']['tp'].append(int(tp))
            print(f"DANN Accuracy: {dann_acc*100:.2f}% (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
    
    # =========================================================================
    # Final Results Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 70)
    
    # Create results table
    print("\nPer-Fold Accuracies (%):")
    print("-" * 70)
    header = "Fold\t"
    for model in models_to_run:
        header += f"{model.upper()}\t"
    print(header)
    print("-" * 70)
    
    for i, fold_num in enumerate(fold_numbers):
        row = f"{fold_num}\t"
        for model in models_to_run:
            if i < len(results[model]['accuracies']):
                row += f"{results[model]['accuracies'][i]*100:.1f}%\t"
            else:
                row += "N/A\t"
        print(row)
    
    print("-" * 70)
    print("\nOverall Performance:")
    print("-" * 70)
    
    summary_data = {}
    for model in models_to_run:
        accs = results[model]['accuracies']
        if len(accs) > 0:
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            summary_data[model] = {'mean': mean_acc, 'std': std_acc}
            print(f"{model.upper():10s}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    print("-" * 70)
    
    # DANN improvement analysis
    if 'dann' in summary_data:
        print("\nDANN Improvement Over Baselines:")
        print("-" * 70)
        dann_mean = summary_data['dann']['mean']
        for model in models_to_run:
            if model != 'dann' and model in summary_data:
                baseline_mean = summary_data[model]['mean']
                improvement = dann_mean - baseline_mean
                rel_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
                print(f"DANN vs {model.upper()}: {improvement:+.2f}% (relative: {rel_improvement:+.1f}%)")
        print("-" * 70)
    
    # Save results
    output_file = os.path.join(data_dir, 'ablation_results.mat')
    save_data = {
        'models': models_to_run,
        'fold_numbers': fold_numbers,
    }
    for model in models_to_run:
        save_data[f'{model}_accuracies'] = results[model]['accuracies']
        save_data[f'{model}_tn'] = results[model]['tn']
        save_data[f'{model}_fp'] = results[model]['fp']
        save_data[f'{model}_fn'] = results[model]['fn']
        save_data[f'{model}_tp'] = results[model]['tp']
        if model in summary_data:
            save_data[f'{model}_mean'] = summary_data[model]['mean']
            save_data[f'{model}_std'] = summary_data[model]['std']
    
    savemat(output_file, save_data)
    print(f"\nResults saved to: {output_file}")
    
    # Print confusion matrix summary
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX SUMMARY (Totals across all folds)")
    print("=" * 70)
    for model in models_to_run:
        if len(results[model]['tp']) > 0:
            total_tp = sum(results[model]['tp'])
            total_tn = sum(results[model]['tn'])
            total_fp = sum(results[model]['fp'])
            total_fn = sum(results[model]['fn'])
            
            # Calculate additional metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
            
            print(f"\n{model.upper()}:")
            print(f"  TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn}")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall (Sensitivity): {recall*100:.2f}%")
            print(f"  Specificity: {specificity*100:.2f}%")
            print(f"  F1-Score: {f1*100:.2f}%")
    print("=" * 70)
    
    # Create comparison plot
    create_comparison_plot(results, models_to_run, fold_numbers, data_dir, summary_data)
    
    # Generate text report
    generate_ablation_report(results, models_to_run, fold_numbers, data_dir, summary_data)
    
    return results, summary_data

def create_comparison_plot(results, models_to_run, fold_numbers, data_dir, summary_data):
    """Create visualization comparing all models"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'svm': '#e74c3c', 'cnn': '#3498db', 'cnn_lstm': '#9b59b6', 'dann': '#27ae60'}
    labels = {'svm': 'SVM', 'cnn': 'CNN', 'cnn_lstm': 'CNN-LSTM', 'dann': 'DANN (Ours)'}
    
    # Plot 1: Per-fold bar chart
    ax1 = axes[0]
    x = np.arange(len(fold_numbers))
    width = 0.2
    
    for i, model in enumerate(models_to_run):
        accs = [a * 100 for a in results[model]['accuracies']]
        offset = (i - len(models_to_run)/2 + 0.5) * width
        ax1.bar(x + offset, accs, width, label=labels.get(model, model.upper()), 
                color=colors.get(model, 'gray'), edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Per-Fold Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_numbers)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_ylim([40, 105])
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot
    ax2 = axes[1]
    data_to_plot = [[a * 100 for a in results[m]['accuracies']] for m in models_to_run]
    bp = ax2.boxplot(data_to_plot, labels=[labels.get(m, m.upper()) for m in models_to_run], patch_artist=True)
    
    for patch, model in zip(bp['boxes'], models_to_run):
        patch.set_facecolor(colors.get(model, 'gray'))
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([40, 105])
    
    # Plot 3: Overall comparison bar chart
    ax3 = axes[2]
    model_names = [labels.get(m, m.upper()) for m in models_to_run]
    means = [summary_data[m]['mean'] for m in models_to_run if m in summary_data]
    stds = [summary_data[m]['std'] for m in models_to_run if m in summary_data]
    
    bars = ax3.bar(model_names, means, yerr=stds, capsize=5,
                   color=[colors.get(m, 'gray') for m in models_to_run],
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax3.annotate(f'{mean:.1f}%\n±{std:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Overall Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([40, 110])
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(data_dir, 'ablation_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {fig_path}")

def generate_ablation_report(results, models_to_run, fold_numbers, data_dir, summary_data):
    """Generate a detailed text report"""
    
    report_path = os.path.join(data_dir, 'ablation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Number of Folds: {len(fold_numbers)}\n\n")
        
        # Model descriptions
        f.write("-" * 80 + "\n")
        f.write("MODEL DESCRIPTIONS\n")
        f.write("-" * 80 + "\n\n")
        
        descriptions = {
            'svm': "SVM (Support Vector Machine)\n   - Linear kernel with C=10.0\n   - Features: Flattened spectral power\n   - No deep learning, no domain adaptation\n",
            'cnn': "CNN (Convolutional Neural Network)\n   - 3-block CNN (same as DANN feature extractor)\n   - Standard cross-entropy training\n   - No domain adversarial training\n",
            'cnn_lstm': "CNN-LSTM\n   - 2-block CNN + Bidirectional LSTM\n   - Treats frequency as temporal sequence\n   - No domain adaptation\n",
            'dann': "DANN (Domain Adversarial Neural Network) [PROPOSED]\n   - 3-block CNN feature extractor\n   - Gradient Reversal Layer\n   - Subject classifier for domain adaptation\n   - Cosine classifier with learnable temperature\n"
        }
        
        for model in models_to_run:
            if model in descriptions:
                f.write(f"{descriptions[model]}\n")
        
        # Results table
        f.write("-" * 80 + "\n")
        f.write("PER-FOLD ACCURACY RESULTS (%)\n")
        f.write("-" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Fold':<8}")
        for model in models_to_run:
            f.write(f"{model.upper():<12}")
        f.write("\n" + "-" * (8 + 12 * len(models_to_run)) + "\n")
        
        # Data rows
        for i, fold_num in enumerate(fold_numbers):
            f.write(f"{fold_num:<8}")
            for model in models_to_run:
                if i < len(results[model]['accuracies']):
                    acc = results[model]['accuracies'][i] * 100
                    f.write(f"{acc:<12.1f}")
                else:
                    f.write(f"{'N/A':<12}")
            f.write("\n")
        
        f.write("-" * (8 + 12 * len(models_to_run)) + "\n\n")
        
        # Summary
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"{'Model':<15}{'Mean (%)':<15}{'Std (%)':<15}{'Min (%)':<15}{'Max (%)':<15}\n")
        f.write("-" * 75 + "\n")
        
        for model in models_to_run:
            if model in summary_data:
                accs = np.array(results[model]['accuracies']) * 100
                f.write(f"{model.upper():<15}{np.mean(accs):<15.2f}{np.std(accs):<15.2f}"
                       f"{np.min(accs):<15.2f}{np.max(accs):<15.2f}\n")
        
        f.write("-" * 75 + "\n\n")
        
        # DANN improvement
        if 'dann' in summary_data:
            f.write("-" * 80 + "\n")
            f.write("DANN IMPROVEMENT OVER BASELINES\n")
            f.write("-" * 80 + "\n\n")
            
            dann_mean = summary_data['dann']['mean']
            for model in models_to_run:
                if model != 'dann' and model in summary_data:
                    baseline_mean = summary_data[model]['mean']
                    improvement = dann_mean - baseline_mean
                    rel_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
                    f.write(f"DANN vs {model.upper()}:\n")
                    f.write(f"  Absolute improvement: {improvement:+.2f}%\n")
                    f.write(f"  Relative improvement: {rel_improvement:+.1f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Study: Compare all models including DANN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing exported MATLAB data')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'svm', 'cnn', 'cnn_lstm', 'dann'],
                        help='Which model(s) to run (default: all)')
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['svm', 'cnn', 'cnn_lstm', 'dann']
    else:
        models = [args.model]
    
    run_ablation_study(args.data_dir, models)

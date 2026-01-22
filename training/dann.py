#!/usr/bin/env python3
"""
Domain Adversarial Training for EEG Drowsiness Detection
========================================================

This script implements Domain Adversarial Neural Networks (DANN) for 
cross-subject EEG drowsiness detection using conservative hyperparameters
optimized for stable cross-subject generalization.

Requirements:
    pip install torch torchvision scipy matplotlib scikit-learn

Usage:
    python training/dann.py
    python training/dann.py --data_dir outputs
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
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from utils import load_mat_file, prepare_fold_data

# Set random seeds for reproducibility
torch.manual_seed(43)
np.random.seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(43)

# ============================================================================
# Model Components
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
        return grad_output.neg() * ctx.lambda_, None

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
        return self.scale * torch.matmul(x_norm, w_norm.t())

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
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            dummy_features = self.features(dummy_input)
            self.feature_dim = dummy_features.numel()
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.shared_norm = nn.LayerNorm(128)
        
        self.drowsiness_classifier = CosineClassifier(128, num_classes)
        self.subject_classifier = CosineClassifier(128, num_subjects)
        
    def forward(self, x, lambda_=1.0):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        shared_features = self.shared_fc(features)
        shared_features = self.shared_norm(shared_features)
        
        drowsiness_pred = self.drowsiness_classifier(shared_features)
        reversed_features = gradient_reversal(shared_features, lambda_)
        subject_pred = self.subject_classifier(reversed_features)
        
        return drowsiness_pred, subject_pred

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data with subject labels"""
    
    def __init__(self, X, y_drowsiness, y_subject):
        self.X = torch.FloatTensor(X)
        self.y_drowsiness = torch.LongTensor(y_drowsiness - 1)  # Convert to 0-based
        self.y_subject = torch.LongTensor(y_subject - 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_drowsiness[idx], self.y_subject[idx]

# ============================================================================
# Training Configuration
# ============================================================================

CONFIG = {
    'num_epochs': 150,
    'lr': 0.0005,
    'weight_decay': 5e-5,
    'lambda_max': 0.3,
    'lambda_speed': 3,
    'domain_weight_min': 0.02,
    'domain_weight_max': 0.1,
    'domain_ramp_frac': 0.6,
    'patience': 30,
    'label_smoothing': 0.05,
}

def train_domain_adversarial_model(train_loader, val_loader, model, device):
    """
    Train the domain adversarial model with conservative hyperparameters
    """
    cfg = CONFIG
    
    drowsiness_criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    subject_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    num_epochs = cfg['num_epochs']
    warmup_epochs = max(1, int(0.1 * num_epochs))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=cfg['lr'] * 0.05
    )
    
    train_history = {'drowsiness_loss': [], 'subject_loss': [], 'total_loss': []}
    val_history = {'accuracy': [], 'loss': []}
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_drowsiness_loss = 0.0
        epoch_subject_loss = 0.0
        epoch_total_loss = 0.0
        
        # Warmup LR
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg['lr'] * (epoch + 1) / warmup_epochs
        else:
            scheduler.step()
        
        progress = epoch / num_epochs
        
        # Conservative lambda schedule
        lambda_ = cfg['lambda_max'] * (2.0 / (1.0 + np.exp(-cfg['lambda_speed'] * progress)) - 1.0)
        
        # Conservative domain weight schedule
        ramp = min(1.0, progress / cfg['domain_ramp_frac'])
        domain_weight = cfg['domain_weight_min'] + (cfg['domain_weight_max'] - cfg['domain_weight_min']) * ramp
        
        for data, drowsiness_labels, subject_labels in train_loader:
            data = data.to(device)
            drowsiness_labels = drowsiness_labels.to(device)
            subject_labels = subject_labels.to(device)
            
            optimizer.zero_grad()
            drowsiness_pred, subject_pred = model(data, lambda_)
            
            drowsiness_loss = drowsiness_criterion(drowsiness_pred, drowsiness_labels)
            subject_loss = subject_criterion(subject_pred, subject_labels)
            total_loss = drowsiness_loss + domain_weight * subject_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_drowsiness_loss += drowsiness_loss.item()
            epoch_subject_loss += subject_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, drowsiness_labels, _ in val_loader:
                data = data.to(device)
                drowsiness_labels = drowsiness_labels.to(device)
                
                drowsiness_pred, _ = model(data, 0.0)
                val_loss += drowsiness_criterion(drowsiness_pred, drowsiness_labels).item()
                
                _, predicted = torch.max(drowsiness_pred.data, 1)
                val_total += drowsiness_labels.size(0)
                val_correct += (predicted == drowsiness_labels).sum().item()
        
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        
        # Save history
        train_history['drowsiness_loss'].append(epoch_drowsiness_loss / len(train_loader))
        train_history['subject_loss'].append(epoch_subject_loss / len(train_loader))
        train_history['total_loss'].append(epoch_total_loss / len(train_loader))
        val_history['accuracy'].append(val_acc)
        val_history['loss'].append(val_loss)
        
        # Early stopping with longer patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f'  Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 25 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Val Acc: {val_acc:.4f}, Lambda: {lambda_:.3f}, '
                  f'DomW: {domain_weight:.3f}, LR: {current_lr:.6f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_history, val_history

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
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
    
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return accuracy, predictions, true_labels

def main(data_dir='outputs'):
    """Main training loop for all folds"""
    
    print("=" * 70)
    print("Domain Adversarial Training for EEG Drowsiness Detection")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Configuration: Conservative (matching tune_dann.py)")
    print("=" * 70)
    
    # Check if data exists
    export_check_file = os.path.join(data_dir, 'export_complete.mat')
    if not os.path.exists(export_check_file):
        print(f"Error: No exported data found in {data_dir}")
        print("Run MATLAB preprocessing first (main_dl_pipeline.m)")
        return
    
    # Load metadata
    metadata_file = os.path.join(data_dir, 'metadata.mat')
    try:
        metadata = load_mat_file(metadata_file)
        num_classes = int(metadata['num_classes'])
        num_subjects = int(metadata['num_subjects'])
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of subjects: {num_subjects}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all fold files
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    print(f"Found {len(fold_numbers)} folds to process\n")
    
    # Results storage
    fold_accuracies = []
    all_predictions = {}
    all_true_labels = {}
    
    # Process each fold
    for fold_num in fold_numbers:
        print(f"--- Fold {fold_num} ---")
        
        try:
            fold_data = load_mat_file(os.path.join(data_dir, f'fold_{fold_num}_data.mat'))
        except Exception as e:
            print(f"Error loading fold {fold_num}: {e}")
            continue
        
        # Prepare data with consistent shapes
        (X_train, y_train, subject_train,
         X_val, y_val, subject_val,
         X_test, y_test, subject_test) = prepare_fold_data(fold_data)
        
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Normalize
        train_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        train_std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        
        # Create datasets and loaders
        train_dataset = EEGDataset(X_train, y_train, subject_train)
        val_dataset = EEGDataset(X_val, y_val, subject_val)
        test_dataset = EEGDataset(X_test, y_test, subject_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        input_shape = (1, X_train.shape[2], X_train.shape[3])
        model = DomainAdversarialCNN(input_shape, num_classes, num_subjects).to(device)
        
        # Train model
        model, train_history, val_history = train_domain_adversarial_model(
            train_loader, val_loader, model, device
        )
        
        # Evaluate on test set
        test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)
        
        print(f"  Test Accuracy: {test_accuracy*100:.2f}%\n")
        
        # Store results
        fold_accuracies.append((fold_num, test_accuracy))
        all_predictions[fold_num] = predictions
        all_true_labels[fold_num] = true_labels
        
        # Save fold results
        fold_results = {
            'fold_num': fold_num,
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'train_history': train_history,
            'val_history': val_history
        }
        savemat(os.path.join(data_dir, f'fold_{fold_num}_results.mat'), fold_results)
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Fold {fold_num}: Accuracy {test_accuracy*100:.2f}%')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_save_path = os.path.join(data_dir, f'confusion_matrix_fold_{fold_num}.png')
        plt.savefig(cm_save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Final results
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    accuracies_only = [acc for _, acc in fold_accuracies]
    
    print(f"\nProcessed {len(fold_accuracies)} folds")
    print(f"Mean CV Accuracy: {np.mean(accuracies_only)*100:.2f}% ± {np.std(accuracies_only)*100:.2f}%")
    
    print("\nPer-fold accuracies:")
    for fold_num, acc in fold_accuracies:
        print(f"  Fold {fold_num}: {acc*100:.2f}%")
    
    # Save final results
    final_results = {
        'fold_accuracies': accuracies_only,
        'fold_numbers': [f for f, _ in fold_accuracies],
        'mean_accuracy': np.mean(accuracies_only),
        'std_accuracy': np.std(accuracies_only),
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'config': CONFIG
    }
    final_results_path = os.path.join(data_dir, 'final_results.mat')
    savemat(final_results_path, final_results)
    
    print(f"\nResults saved to {final_results_path}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Domain Adversarial Training for EEG Drowsiness Detection')
    parser.add_argument('--data_dir', type=str, default='outputs/python_data',
                        help='Directory containing exported MATLAB data (default: outputs/python_data)')
    args = parser.parse_args()
    
    main(data_dir=args.data_dir)

#!/usr/bin/env python3
"""
DANN Hyperparameter Grid Search
================================

Systematic grid search over DANN hyperparameters to find optimal configuration.
Builds on the configurations from tune_dann.py.

Grid Search Ranges (from report):
    Optimizer:              {Adam, AdamW, SGD}           -> Selected: AdamW
    Learning Rate:          {1e-4, 3e-4, 5e-4, 1e-3}     -> Selected: 5e-4
    Weight Decay:           {0, 1e-5, 1e-4, 1e-3}        -> Selected: 1e-4
    Batch Size:             {16, 32, 64}                 -> Selected: 32
    Max Epochs:             {80, 100, 120, 150}          -> Selected: 120
    Early Stopping Patience:{15, 30, 45}                 -> Selected: 30
    Gradient Clip Norm:     {0.5, 1.0, 2.0}              -> Selected: 1.0
    Label Smoothing (ε):    {0, 0.05, 0.1, 0.2}          -> Selected: 0.05

Usage:
    python grid_search_dann.py --data_dir diagnostics/python_data
    python grid_search_dann.py --data_dir diagnostics/python_data --quick
    python grid_search_dann.py --data_dir diagnostics/python_data --full   # Include DANN-specific params
    python grid_search_dann.py --data_dir diagnostics/python_data --resume results.csv
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.io import savemat
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
from datetime import datetime
from itertools import product
import csv
import json

from utils import load_mat_file, prepare_fold_data

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================================
# Model Components (same as tune_dann.py)
# ============================================================================

class EEGDatasetWithSubject(Dataset):
    def __init__(self, X, y_drowsiness, y_subject):
        self.X = torch.FloatTensor(X)
        self.y_drowsiness = torch.LongTensor(y_drowsiness - 1)
        self.y_subject = torch.LongTensor(y_subject - 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_drowsiness[idx], self.y_subject[idx]

class GradientReversalLayer(torch.autograd.Function):
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

# ============================================================================
# Hyperparameter Grid Definition
# ============================================================================

# Full grid search space (matches report Table)
PARAM_GRID = {
    'optimizer': ['Adam', 'AdamW', 'SGD'],
    'lr': [1e-4, 3e-4, 5e-4, 1e-3],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'num_epochs': [80, 100, 120, 150],
    'patience': [15, 30, 45],
    'grad_clip_norm': [0.5, 1.0, 2.0],
    'label_smoothing': [0, 0.05, 0.1, 0.2],
}

# DANN-specific parameters (domain adversarial training)
PARAM_GRID_DANN = {
    'lambda_max': [0.1, 0.3, 0.5],
    'lambda_speed': [2, 3, 5],
    'domain_weight_min': [0.01, 0.02, 0.05],
    'domain_weight_max': [0.05, 0.1, 0.2],
    'domain_ramp_frac': [0.4, 0.6, 0.8],
}

# Quick grid for faster iteration
PARAM_GRID_QUICK = {
    'optimizer': ['AdamW'],
    'lr': [3e-4, 5e-4],
    'weight_decay': [1e-5, 1e-4],
    'batch_size': [32],
    'num_epochs': [100, 120],
    'patience': [30],
    'grad_clip_norm': [1.0],
    'label_smoothing': [0, 0.05],
}

# Combined grid (general + DANN-specific)
PARAM_GRID_FULL = {**PARAM_GRID, **PARAM_GRID_DANN}

def get_default_config():
    """Return default/selected values for parameters not in grid (from report)"""
    return {
        # General hyperparameters (Selected Values from report)
        'optimizer': 'AdamW',
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'num_epochs': 120,
        'patience': 30,
        'grad_clip_norm': 1.0,
        'label_smoothing': 0.05,
        # DANN-specific defaults
        'lambda_max': 0.3,
        'lambda_speed': 3,
        'domain_weight_min': 0.02,
        'domain_weight_max': 0.1,
        'domain_ramp_frac': 0.6,
    }

def generate_configs(param_grid):
    """Generate all combinations of hyperparameters"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        # Fill in defaults for missing params
        defaults = get_default_config()
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
        configs.append(config)

    return configs


def create_optimizer(model, config):
    """Create optimizer based on config"""
    optimizer_name = config.get('optimizer', 'AdamW')
    lr = config['lr']
    weight_decay = config['weight_decay']

    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# ============================================================================
# Training Function
# ============================================================================

def train_dann_with_config(model, train_loader, val_loader, device, config, verbose=False):
    """Train DANN with specified configuration"""

    drowsiness_criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    subject_criterion = nn.CrossEntropyLoss()

    # Create optimizer based on config
    optimizer = create_optimizer(model, config)
    num_epochs = config['num_epochs']
    warmup_epochs = max(1, int(0.1 * num_epochs))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=config['lr'] * 0.05
    )

    # Get gradient clip norm from config
    grad_clip_norm = config.get('grad_clip_norm', 1.0)

    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()

        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr'] * (epoch + 1) / warmup_epochs
        else:
            scheduler.step()

        progress = epoch / num_epochs

        # Lambda schedule
        lambda_ = config['lambda_max'] * (2.0 / (1.0 + np.exp(-config['lambda_speed'] * progress)) - 1.0)

        # Domain weight schedule
        ramp = min(1.0, progress / config['domain_ramp_frac'])
        domain_weight = config['domain_weight_min'] + (config['domain_weight_max'] - config['domain_weight_min']) * ramp

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_acc

def evaluate_model(model, test_loader, device):
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

    acc = accuracy_score(true_labels, predictions)
    return acc, predictions, true_labels

# ============================================================================
# Grid Search Runner
# ============================================================================

def run_single_config(data_dir, config, device, num_classes, num_subjects, fold_numbers, verbose=False):
    """Run a single configuration across all folds and return mean/std accuracy"""

    accuracies = []

    for fold_num in fold_numbers:
        set_seed(SEED)  # Reset seed for each fold for reproducibility

        fold_data = load_mat_file(os.path.join(data_dir, f'fold_{fold_num}_data.mat'))

        (X_train, y_train, subject_train,
         X_val, y_val, subject_val,
         X_test, y_test, subject_test) = prepare_fold_data(fold_data)

        # Normalize
        train_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        train_std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        # Datasets
        train_dataset = EEGDatasetWithSubject(X_train, y_train, subject_train)
        val_dataset = EEGDatasetWithSubject(X_val, y_val, subject_val)
        test_dataset = EEGDatasetWithSubject(X_test, y_test, subject_test)

        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        # Model
        input_shape = (1, X_train.shape[2], X_train.shape[3])
        model = DomainAdversarialCNN(input_shape, num_classes, num_subjects).to(device)

        # Train
        model, _ = train_dann_with_config(model, train_loader, val_loader, device, config, verbose=verbose)

        # Evaluate
        acc, _, _ = evaluate_model(model, test_loader, device)
        accuracies.append(acc)

        if verbose:
            print(f"    Fold {fold_num}: {acc*100:.2f}%")

    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100

    return mean_acc, std_acc, accuracies

def run_grid_search(data_dir, param_grid, resume_file=None, max_configs=None, verbose=True):
    """Run grid search over all configurations"""

    print("=" * 70)
    print("DANN HYPERPARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load metadata
    metadata = load_mat_file(os.path.join(data_dir, 'metadata.mat'))
    num_classes = int(metadata['num_classes'])
    num_subjects = int(metadata['num_subjects'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find folds
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    print(f"Found {len(fold_numbers)} folds")

    # Generate configs
    configs = generate_configs(param_grid)
    total_configs = len(configs)

    if max_configs:
        configs = configs[:max_configs]

    print(f"Total configurations to test: {len(configs)} (of {total_configs})")
    print(f"Parameters being searched: {list(param_grid.keys())}")
    print("=" * 70)

    # Load previous results if resuming
    completed_configs = set()
    if resume_file and os.path.exists(resume_file):
        print(f"Resuming from {resume_file}")
        with open(resume_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config_str = row['config']
                completed_configs.add(config_str)
        print(f"Found {len(completed_configs)} previously completed configurations")

    # Results storage
    results = []
    best_mean = 0
    best_std = 100
    best_config = None

    # Output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(data_dir, f'grid_search_results_{timestamp}.csv')

    # Write header
    fieldnames = ['config_id', 'mean_acc', 'std_acc', 'fold_accs', 'config'] + list(param_grid.keys())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Run grid search
    for i, config in enumerate(configs):
        config_str = json.dumps(config, sort_keys=True)

        # Skip if already completed
        if config_str in completed_configs:
            print(f"[{i+1}/{len(configs)}] Skipping (already completed)")
            continue

        print(f"\n[{i+1}/{len(configs)}] Testing configuration:")
        for k in param_grid.keys():
            print(f"  {k}: {config[k]}")

        try:
            mean_acc, std_acc, fold_accs = run_single_config(
                data_dir, config, device, num_classes, num_subjects, fold_numbers, verbose=verbose
            )

            print(f"  Result: {mean_acc:.2f}% +/- {std_acc:.2f}%")

            # Track best
            if mean_acc > best_mean or (mean_acc == best_mean and std_acc < best_std):
                best_mean = mean_acc
                best_std = std_acc
                best_config = config.copy()
                print(f"  *** NEW BEST ***")

            # Save result
            result_row = {
                'config_id': i,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'fold_accs': json.dumps([round(a*100, 2) for a in fold_accs]),
                'config': config_str,
            }
            for k in param_grid.keys():
                result_row[k] = config[k]

            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result_row)

            results.append({
                'config': config,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'fold_accs': fold_accs
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)

    if results:
        # Sort by mean accuracy (descending), then by std (ascending)
        results.sort(key=lambda x: (-x['mean_acc'], x['std_acc']))

        print(f"\nTop 5 configurations:")
        print("-" * 70)
        for i, r in enumerate(results[:5]):
            print(f"\n{i+1}. Mean: {r['mean_acc']:.2f}% +/- {r['std_acc']:.2f}%")
            for k in param_grid.keys():
                print(f"   {k}: {r['config'][k]}")

        print("\n" + "-" * 70)
        print(f"Best configuration:")
        print(f"  Mean Accuracy: {best_mean:.2f}% +/- {best_std:.2f}%")
        for k, v in best_config.items():
            if k in param_grid.keys():
                print(f"  {k}: {v}")

        # Save best config
        best_config_path = os.path.join(data_dir, f'best_config_{timestamp}.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'config': best_config,
                'mean_acc': best_mean,
                'std_acc': best_std,
            }, f, indent=2)
        print(f"\nBest config saved to: {best_config_path}")

    print(f"Full results saved to: {output_csv}")
    print("=" * 70)

    # Save summary to .mat file
    if results:
        mat_results = {
            'best_mean': best_mean,
            'best_std': best_std,
            'best_config': json.dumps(best_config),
            'num_configs_tested': len(results),
            'all_means': [r['mean_acc'] for r in results],
            'all_stds': [r['std_acc'] for r in results],
        }
        savemat(os.path.join(data_dir, f'grid_search_summary_{timestamp}.mat'), mat_results)

    return results, best_config

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DANN Hyperparameter Grid Search')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing exported MATLAB data')
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller grid for quick testing')
    parser.add_argument('--full', action='store_true',
                        help='Include DANN-specific params (lambda, domain weights) in grid')
    parser.add_argument('--dann_only', action='store_true',
                        help='Only search DANN-specific params (use default general params)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from previous results CSV file')
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Maximum number of configurations to test')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-fold results')
    args = parser.parse_args()

    # Select grid based on arguments
    if args.quick:
        print("Using QUICK grid (reduced search space)")
        param_grid = PARAM_GRID_QUICK
    elif args.full:
        print("Using FULL grid (general + DANN-specific params)")
        print(f"  WARNING: This will test {np.prod([len(v) for v in PARAM_GRID_FULL.values()])} configurations!")
        param_grid = PARAM_GRID_FULL
    elif args.dann_only:
        print("Using DANN-specific params only (domain adaptation tuning)")
        param_grid = PARAM_GRID_DANN
    else:
        print("Using STANDARD grid (general hyperparameters from report)")
        param_grid = PARAM_GRID

    # Print grid summary
    print("\nGrid search space:")
    total_configs = 1
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
        total_configs *= len(v)
    print(f"\nTotal configurations: {total_configs}")

    run_grid_search(
        args.data_dir,
        param_grid,
        resume_file=args.resume,
        max_configs=args.max_configs,
        verbose=args.verbose
    )

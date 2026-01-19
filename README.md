# EEG Drowsiness Detection with Domain Adversarial Neural Networks

A hybrid MATLAB-Python deep learning pipeline for subject-independent frontal EEG-based drowsiness detection using Domain Adversarial Neural Networks (DANN).

## Overview

This pipeline implements Leave-One-Subject-Out (LOSO) cross-validation with domain adversarial training to achieve robust, subject-independent drowsiness detection from frontal EEG signals. The system processes raw EEG data through frequency-domain transformation and trains CNNs with gradient reversal to learn subject-invariant features.

## Pipeline Architecture

```
Raw EEG Data (.cnt files)
        │
        ▼
┌──────────────────────────────────────┐
│  MATLAB Preprocessing                │
│  ├── Bandpass filtering (0.5-50 Hz)  │
│  ├── Downsampling (250 Hz)           │
│  ├── Re-referencing (linked mastoids)│
│  ├── Regression-based blink removal  │
│  ├── ICA + ICLabel artifact removal  │
│  ├── FFT spectral transformation     │
│  └── Subject-wise normalization      │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Python Deep Learning (DANN)         │
│  ├── 3-block CNN feature extractor   │
│  ├── Gradient reversal layer         │
│  ├── Drowsiness classifier           │
│  └── Subject classifier (adversarial)│
└──────────────────────────────────────┘
        │
        ▼
    Results & Analysis
```

## Requirements

### MATLAB
- MATLAB R2020b or later
- EEGLAB (tested with 2025.0.0)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

### Python
```bash
pip install torch>=2.0.0 torchvision numpy scipy matplotlib scikit-learn h5py
```

## Repository Structure

### MATLAB Scripts

| Script | Description |
|--------|-------------|
| `main_dl_pipeline.m` | Main orchestrator - runs the complete preprocessing pipeline |
| `step1_preprocess_data.m` | Channel selection, filtering, downsampling, re-referencing, blink removal |
| `step2_run_ica_and_iclabel.m` | ICA decomposition and automatic artifact rejection via ICLabel |
| `step3_prepare_sequence_data.m` | Windowing (5s) and FFT transformation to frequency-domain images |
| `step4_export_for_python.m` | Export LOSO cross-validation splits for Python training |
| `apply_subject_normalization.m` | Per-subject z-score normalization helper |
| `analyze_subject_characteristics.m` | Per-subject analysis to explain fold performance variation |
| `create_effect_size_figure.m` | Publication-quality figures for spectral separability analysis |
| `visualize_eeg_timeseries.m` | Multi-channel EEG visualization for preprocessing validation |

### Python Scripts

| Script | Description |
|--------|-------------|
| `dann.py` | Domain Adversarial Neural Network training with LOSO cross-validation |
| `utils.py` | Data loading utilities for MATLAB v5/v7.3 files and data preparation |
| `visualize_eeg_samples.py` | Frequency-domain heatmaps, per-channel spectra, and band power analysis |

## Usage

### 1. Prepare Data

Organize EEG data in the following structure:
```
/path/to/data/
├── Subject1/
│   ├── Normal state.cnt
│   └── Fatigue state.cnt
├── Subject2/
│   ├── Normal state.cnt
│   └── Fatigue state.cnt
└── ...
```

### 2. Run MATLAB Preprocessing

```matlab
% Update the data path in main_dl_pipeline.m
data_root_path = '/path/to/your/EEG/data';

% Initialize EEGLAB and run pipeline
eeglab nogui
main_dl_pipeline
```

This will:
- Process all subjects through preprocessing steps
- Apply subject-wise spectral normalization
- Create LOSO cross-validation splits
- Export data to `diagnostics/python_data/`

### 3. Train DANN Model

```bash
python dann.py --data_dir diagnostics/python_data
```

### 4. Analyze Results

```matlab
% Generate per-subject analysis
analyze_subject_characteristics('diagnostics/python_data')

% Create publication figures
create_effect_size_figure('diagnostics/python_data')
```

```bash
# Generate spectral comparison figures (alert vs drowsy heatmaps, per-channel spectra)
python visualize_eeg_samples.py
```

## Preprocessing Details

### Step 1: Basic Preprocessing
- **Channels**: Frontal EEG (FP1, FP2, F7, F3, FZ, F4, F8) + mastoid references (A1, A2)
- **Filtering**: 0.5-50 Hz bandpass (FIR filter)
- **Sampling**: Downsampled to 250 Hz
- **Reference**: Linked mastoids, then removed
- **Blink Removal**: Regression-based correction using vEOG (FP1-F7)

### Step 2: ICA Artifact Removal
- **Method**: Extended Infomax ICA
- **Classification**: ICLabel automatic component classification
- **Rejection**: Components where P(Brain) < P(Eye) or P(Brain) < P(Muscle)

### Step 3: Spectral Features
- **Windowing**: 5-second windows with 1-second stride
- **FFT**: 1024-point FFT → 128 frequency bins (0-31.25 Hz)
- **Format**: Frequency × Channels × 1 images for CNN input

### Step 4: Normalization
- **Method**: Subject-wise z-score normalization
- **Training**: Statistics computed per training subject
- **Test**: Each held-out subject normalized by their own statistics

## Model Architecture

The DANN model consists of:

1. **Shared Feature Extractor**: 3-block CNN (32→64→128 filters) with BatchNorm, ReLU, MaxPool
2. **Drowsiness Classifier**: Cosine classifier with learnable temperature
3. **Subject Classifier**: Cosine classifier with gradient reversal layer

The gradient reversal layer enables adversarial training to learn subject-invariant features.

## Output Files

After training, results are saved to `diagnostics/python_data/`:
- `fold_X_data.mat` - Preprocessed data for each LOSO fold
- `fold_X_results.mat` - Predictions and metrics for each fold
- `final_results.mat` - Aggregated cross-validation results
- `confusion_matrix_fold_X.png` - Per-fold confusion matrices

Visualization outputs (saved to working directory):
- `eeg_spectral_comparison.png/svg` - Alert vs drowsy frequency heatmaps and channel spectra
- `eeg_comparison_fold_X.png/svg` - Per-fold EEG comparison plots


## License

This project is provided for research and educational purposes.

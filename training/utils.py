"""
Shared utility functions for EEG Drowsiness Detection Pipeline
==============================================================

This module contains common data loading functions used across multiple Python scripts.
"""

import numpy as np
import h5py
from scipy.io import loadmat as scipy_loadmat


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_matlab_v73(filename):
    """
    Load MATLAB v7.3 files using h5py.
    
    MATLAB v7.3 files use HDF5 format, which requires h5py to read.
    This function handles the conversion from MATLAB to Python data types.
    
    Parameters
    ----------
    filename : str
        Path to the .mat file (v7.3 format)
    
    Returns
    -------
    dict
        Dictionary containing the loaded data with keys matching MATLAB variable names
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):  # Skip HDF5 metadata
                continue
            try:
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    if item.dtype.char == 'U':  # Unicode strings
                        data[key] = [''.join(chr(c[0]) for c in f[item[0][0]][:].T)]
                    elif len(item.shape) == 2 and item.shape[0] == 1:
                        data[key] = item[0, 0] if item.size == 1 else item[0, :]
                    else:
                        data[key] = item[:]
                        if len(data[key].shape) > 2:
                            data[key] = np.transpose(data[key])
            except:
                continue
    return data


def load_matlab_v5(filename):
    """
    Load MATLAB v5 files using scipy.io.loadmat.
    
    Parameters
    ----------
    filename : str
        Path to the .mat file (v5 format)
    
    Returns
    -------
    dict
        Dictionary containing the loaded data
    """
    data = scipy_loadmat(filename, squeeze_me=False, struct_as_record=False)
    return {k: v for k, v in data.items() if not k.startswith('_')}


def load_mat_file(filename):
    """
    Load .mat file, trying v7.3 (HDF5) first, then falling back to v5.
    
    This is the recommended function to use for loading MATLAB files
    as it handles both formats automatically.
    
    Parameters
    ----------
    filename : str
        Path to the .mat file
    
    Returns
    -------
    dict
        Dictionary containing the loaded data
    """
    try:
        return load_matlab_v73(filename)
    except (OSError, IOError):
        return load_matlab_v5(filename)


def ensure_4d(arr):
    """
    Ensure array is 4D (freq, channels, 1, samples) for MATLAB-exported data.
    
    Some MATLAB exports may squeeze out singleton dimensions. This function
    adds them back to ensure consistent 4D shape.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array (3D or 4D)
    
    Returns
    -------
    np.ndarray
        4D array with shape (freq, channels, 1, samples)
    """
    arr = np.array(arr)
    if arr.ndim == 3:
        arr = arr[:, :, np.newaxis, :]
    return arr


def prepare_fold_data(fold_data):
    """
    Prepare fold data with consistent shapes for PyTorch training.
    
    Handles the conversion from MATLAB format (freq, ch, 1, samples) 
    to PyTorch format (samples, 1, freq, ch).
    
    Parameters
    ----------
    fold_data : dict
        Dictionary containing fold data loaded from MATLAB .mat file
        Expected keys: XTrain, XValidation, XTest, YTrain_numeric, 
        YValidation_numeric, YTest_numeric, train_subject_nums,
        val_subject_nums, test_subject_nums
    
    Returns
    -------
    tuple
        (X_train, y_train, subject_train,
         X_val, y_val, subject_val,
         X_test, y_test, subject_test)
    """
    # Ensure 4D arrays
    XTrain = ensure_4d(fold_data['XTrain'])
    XValidation = ensure_4d(fold_data['XValidation'])
    XTest = ensure_4d(fold_data['XTest'])
    
    # Transpose from MATLAB (freq, ch, 1, samples) to PyTorch (samples, 1, freq, ch)
    X_train = np.transpose(XTrain, (3, 2, 0, 1))
    X_val = np.transpose(XValidation, (3, 2, 0, 1))
    X_test = np.transpose(XTest, (3, 2, 0, 1))
    
    # Flatten labels and subject indices
    y_train = np.array(fold_data['YTrain_numeric']).flatten()
    y_val = np.array(fold_data['YValidation_numeric']).flatten()
    y_test = np.array(fold_data['YTest_numeric']).flatten()
    
    subject_train = np.array(fold_data['train_subject_nums']).flatten()
    subject_val = np.array(fold_data['val_subject_nums']).flatten()
    subject_test = np.array(fold_data['test_subject_nums']).flatten()
    
    return (X_train, y_train, subject_train,
            X_val, y_val, subject_val,
            X_test, y_test, subject_test)

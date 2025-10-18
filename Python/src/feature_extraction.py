"""
feature_extraction.py

This module provides functions to extract time-domain features from EEG epochs.
It is designed to be robust, well-documented, and ready for integration into the
main project pipeline.

For Iteration 1, this module implements a complete set of 16 time-domain features.
"""

import numpy as np
from scipy import stats
import pandas as pd
import config # Assuming config.py is in the parent directory or PYTHONPATH

def _calculate_hjorth_parameters(epoch):
    """
    Calculates the three Hjorth parameters (Activity, Mobility, Complexity).

    This implementation includes a small epsilon to prevent division-by-zero
    errors on flat or near-flat signals.

    Args:
        epoch (np.ndarray): A 1D signal epoch.

    Returns:
        tuple: A tuple containing (activity, mobility, complexity).
    """
    # Ensure epoch is a numpy array
    epoch = np.asarray(epoch)
    
    # Calculate first and second derivatives
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)

    # --- Activity ---
    # The variance of the signal itself.
    activity = np.var(epoch)

    # --- Mobility ---
    # The standard deviation of the first derivative divided by the standard
    # deviation of the original signal. Represents the mean frequency.
    var_diff1 = np.var(diff1)
    # Add a small constant (epsilon) to the denominator to avoid division by zero
    mobility = np.sqrt(var_diff1 / (activity + 1e-8))

    # --- Complexity ---
    # The ratio of the mobility of the first derivative to the mobility of the
    # original signal. Represents the change in frequency.
    var_diff2 = np.var(diff2)
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-8))
    complexity = mobility_diff1 / (mobility + 1e-8)
    
    return activity, mobility, complexity

def extract_time_domain_features(epoch):
    """
    Extracts a comprehensive set of 16 time-domain features from a single epoch.
    This is the complete feature set required for Iteration 1.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        dict: A dictionary of feature names and their calculated values.
    """
    features = {}

    # --- 1. Basic Statistical Features (7 features) ---
    features['mean'] = np.mean(epoch)
    features['median'] = np.median(epoch)
    features['std'] = np.std(epoch)
    features['variance'] = np.var(epoch)
    features['rms'] = np.sqrt(np.mean(epoch**2))
    features['skewness'] = stats.skew(epoch)
    features['kurtosis'] = stats.kurtosis(epoch)

    # --- 2. Amplitude and Range Features (4 features) ---
    features['min'] = np.min(epoch)
    features['max'] = np.max(epoch)
    features['ptp_amplitude'] = np.ptp(epoch) # Peak-to-peak
    q75, q25 = np.percentile(epoch, [75, 25])
    features['iqr'] = q75 - q25 # Interquartile range

    # --- 3. Hjorth Parameters (3 features) ---
    activity, mobility, complexity = _calculate_hjorth_parameters(epoch)
    features['hjorth_activity'] = activity
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity
    
    # --- 4. Signal-based Features (2 features) ---
    features['waveform_length'] = np.sum(np.abs(np.diff(epoch)))
    features['zero_crossing_rate'] = len(np.where(np.diff(np.sign(epoch)))[0]) / len(epoch)

    return features

def extract_features(preprocessed_data, labels):
    """
    Orchestrates the feature extraction process for all epochs and channels.

    This function iterates through each epoch and each specified channel, applies
    the required feature extraction functions based on the current iteration
    defined in the config file, and returns a structured DataFrame.

    Args:
        preprocessed_data (dict): A dictionary where keys are channel names
                                  (e.g., 'EEG C3-A2') and values are numpy arrays
                                  of shape (n_epochs, n_samples).
        labels (np.ndarray): A 1D array of labels corresponding to each epoch.

    Returns:
        pd.DataFrame: A DataFrame where each row is an epoch and each column is a
                      unique feature (e.g., 'EEG C3-A2_mean').
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")

    channels = list(preprocessed_data.keys())
    if not channels:
        raise ValueError("No channels found in preprocessed data.")
    
    n_epochs = preprocessed_data[channels[0]].shape[0]
    all_features_list = []

    for i in range(n_epochs):
        epoch_features = {}
        
        # --- Iteration-specific Logic ---
        if config.CURRENT_ITERATION == 1:
            # For Iteration 1, only extract time-domain features from EEG channels
            for channel in channels:
                if 'EEG' in channel:
                    epoch_data = preprocessed_data[channel][i, :]
                    time_features = extract_time_domain_features(epoch_data)
                    for feature_name, value in time_features.items():
                        epoch_features[f"{channel}_{feature_name}"] = value

        # TODO: STUDENT IMPLEMENTATION for Iteration 2
        # elif config.CURRENT_ITERATION == 2:
        #     # Add frequency-domain features for EEG and EOG
        #     pass

        # TODO: STUDENT IMPLEMENTATION for Iteration 3
        # elif config.CURRENT_ITERATION == 3:
        #     # Add EMG features
        #     pass
            
        all_features_list.append(epoch_features)

    features_df = pd.DataFrame(all_features_list)
    
    print(f"Feature extraction complete. Final feature matrix shape: {features_df.shape}")
    
    return features_df


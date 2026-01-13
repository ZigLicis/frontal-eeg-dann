%% MAIN DROWSINESS ANALYSIS PIPELINE (DEEP LEARNING)
% This script implements an end-to-end deep learning pipeline to detect
% driver drowsiness from continuous EEG recordings.
%
% Pipeline Steps:
% 1. Data Traversal: Scans a root directory for subject folders and their
%    corresponding 'Normal' and 'Fatigued' state EEG files.
% 2. Preprocessing: For each file, it performs:
%    - Channel selection (frontal channels + reference)
%    - Bandpass filtering (0.5-50 Hz)
%    - Downsampling (250 Hz)
%    - Re-referencing to linked mastoids
%    - ICA artifact removal with ICLabel
% 3. Data Preparation (Step 3): The preprocessed data is segmented
%    into windows and converted to spectral features.
% 4. Data Export: The windowed data from all subjects is exported
%    for Python deep learning training.
%
% To Use:
% 1. Update the 'data_root_path' variable below.
% 2. Start EEGLAB.
% 3. Run this script from the MATLAB command window or editor.
%
% Requires: Deep Learning Toolbox, Signal Processing Toolbox,
%           Statistics and Machine Learning Toolbox.

% --- USER-DEFINED PARAMETERS ---
data_root_path = '/Users/ziglicis/Desktop/Research/ResearchDatasets/TheOriginalEEG'; 

% Preprocessing Parameters (passed to step1)
low_cutoff_freq  = 0.5;
high_cutoff_freq = 50;
downsample_rate  = 250;
frontal_channels = {'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8'};
ref_channels     = {'A1', 'A2'};

% Windowing Parameters (passed to step3)
window_length_sec = 5;
stride_seconds = 1; % 1 second overlap

% Visualization Parameters
enable_visualization = false; % Set to true to generate before/after plots
viz_subject_limit = 12; % Only visualize first N subjects (set to Inf for all)

% ICA Manual Review
manual_ica_review = true; % Set to true to manually select ICs to reject after ICLabel

% --- END OF PARAMETERS ---

% Start EEGLAB if not running
if ~exist('eeglab', 'file')
    eeglab_path = fullfile(fileparts(pwd), 'eeglab2025.0.0');
    if exist(eeglab_path, 'dir')
        fprintf('Adding EEGLAB to path: %s\n', eeglab_path);
        addpath(eeglab_path);
    else
        error('EEGLAB not found. Please add it to your MATLAB path or ensure eeglab2025.0.0 folder exists in parent directory.');
    end
end

% Properly initialize EEGLAB (this loads plugins)
fprintf('Initializing EEGLAB...\n');
eeglab nogui;

% Check if pop_loadcnt is available
if ~exist('pop_loadcnt', 'file')
    error('pop_loadcnt function not found. Make sure the Neuroscan plugin is installed and EEGLAB is properly initialized.');
end

% --- 1. Data Traversal and Preparation ---
fprintf('--- Phase 1: Data Traversal and Preparation ---\n');

ALL_X_data = {};
ALL_Y_data = {};
ALL_subject_ids = {};

subject_folders = dir(fullfile(data_root_path, '*'));
subject_folders = subject_folders([subject_folders.isdir] & ~ismember({subject_folders.name},{'.','..'}));

fprintf('Found %d subject folders:\n', length(subject_folders));
for i = 1:length(subject_folders)
    fprintf('  %s\n', subject_folders(i).name);
end
fprintf('\n');

% Loop through each subject folder
for i = 1:length(subject_folders)
    subject_id = subject_folders(i).name;
    subject_path = fullfile(data_root_path, subject_id);
    fprintf('\n--- Processing Subject: %s ---\n', subject_id);
    
    files_to_process = {
        struct('path', fullfile(subject_path, 'Normal state.cnt'), 'label', 'Normal'),...
        struct('path', fullfile(subject_path, 'Fatigue state.cnt'), 'label', 'Fatigued')
    };
    
    for j = 1:length(files_to_process)
        file_info = files_to_process{j};
        fprintf('  Checking file: %s\n', file_info.path);
        if ~exist(file_info.path, 'file')
            fprintf('  File not found, skipping: %s\n', file_info.path);
            continue; 
        end
        
        fprintf('> Loading and processing file: %s\n', file_info.path);
        
        % Try to load the .cnt file using different methods
        try
            EEG = pop_loadcnt(file_info.path, 'dataformat', 'auto');
        catch ME1
            fprintf('  pop_loadcnt failed, trying pop_biosig...\n');
            try
                EEG = pop_biosig(file_info.path);
            catch ME2
                fprintf('  Both loading methods failed. Skipping file.\n');
                fprintf('  Error 1: %s\n', ME1.message);
                fprintf('  Error 2: %s\n', ME2.message);
                continue;
            end
        end
        
        EEG = pop_chanedit(EEG, 'lookup','Standard-10-20-Cap81.ced');
        
        % Determine if visualization should be enabled for this subject
        should_visualize = enable_visualization && (i <= viz_subject_limit);
        
        % Step 1: Basic Preprocessing
        EEG = step1_preprocess_data(EEG, low_cutoff_freq, high_cutoff_freq, downsample_rate, frontal_channels, ref_channels, should_visualize);
        
        % Step 2: ICA and ICLabel artifact removal
        subject_file_id = sprintf('%s_%s', subject_id, file_info.label);
        EEG = step2_run_ica_and_iclabel(EEG, should_visualize, subject_file_id, manual_ica_review);
        
        % Step 3: Prepare sequence data (windowing + spectral features)
        [X, Y] = step3_prepare_sequence_data(EEG, file_info.label, window_length_sec, stride_seconds);
        fprintf('  Generated %d windows from this file\n', length(Y));
        
        % Create subject ID array for this file's windows
        subject_ids_for_file = repmat({subject_id}, length(Y), 1);
        
        % Accumulate results
        ALL_X_data = [ALL_X_data; X];
        ALL_Y_data = [ALL_Y_data; Y];
        ALL_subject_ids = [ALL_subject_ids; subject_ids_for_file];
    end
end

fprintf('\n\n--- All subjects processed. Total windows collected: %d ---\n', length(ALL_Y_data));

% Check if any data was collected
if isempty(ALL_Y_data) || length(ALL_Y_data) == 0
    error('No data windows were collected! Please check:\n1. Data path is correct\n2. Files "Normal state.cnt" and "Fatigue state.cnt" exist in subject folders\n3. Files can be loaded by EEGLAB');
end

% Convert labels to categorical for analysis
if iscell(ALL_Y_data)
    ALL_Y_data = string(ALL_Y_data);
end

% --- 2. Data Summary ---
fprintf('--- Phase 2: Data Summary ---\n');

unique_subjects = unique(ALL_subject_ids);
fprintf('Total subjects found: %d\n', length(unique_subjects));
fprintf('Subjects: %s\n', strjoin(unique_subjects, ', '));

% Verify preprocessing
fprintf('\n--- Preprocessing Verification ---\n');
normal_count = sum(strcmp(ALL_Y_data, 'Normal'));
fatigued_count = sum(strcmp(ALL_Y_data, 'Fatigued'));
fprintf('Label distribution: Normal=%d, Fatigued=%d (%.1f%% / %.1f%%)\n', ...
    normal_count, fatigued_count, ...
    100*normal_count/length(ALL_Y_data), 100*fatigued_count/length(ALL_Y_data));

% Check data distribution per subject
fprintf('\nData distribution per subject:\n');
for i = 1:length(unique_subjects)
    subj_indices = strcmp(ALL_subject_ids, unique_subjects{i});
    subj_labels = ALL_Y_data(subj_indices);
    subj_normal = sum(strcmp(subj_labels, 'Normal'));
    subj_fatigued = sum(strcmp(subj_labels, 'Fatigued'));
    fprintf('  %s: %d windows (Normal=%d, Fatigued=%d)\n', ...
        unique_subjects{i}, length(subj_labels), subj_normal, subj_fatigued);
end

% --- 3. Leave-One-Subject-Out (LOSO) Cross-Validation ---
fprintf('\n--- Phase 3: Leave-One-Subject-Out (LOSO) Cross-Validation ---\n');

k_folds = length(unique_subjects);
rng('default');
rng(42, 'twister');
shuffled_subjects = unique_subjects(randperm(k_folds));

fprintf('Performing LOSO cross-validation across %d subjects...\n', k_folds);

cv_accuracies = [];

for fold = 1:k_folds
    fprintf('\n--- Fold %d/%d (Test Subject: %s) ---\n', fold, k_folds, shuffled_subjects{fold});
    
    test_subjects = shuffled_subjects(fold);
    train_subjects = setdiff(shuffled_subjects, test_subjects);
    
    fprintf('Test subjects: %s\n', strjoin(test_subjects, ', '));
    fprintf('Train subjects: %s\n', strjoin(train_subjects, ', '));
    
    % Create train/test splits based on subjects
    test_mask = ismember(ALL_subject_ids, test_subjects);
    train_mask = ~test_mask;
    
    % Further split training data into train/validation (80/20)
    train_indices = find(train_mask);
    val_split = cvpartition(length(train_indices), 'HoldOut', 0.2);
    
    final_train_indices = train_indices(val_split.training);
    val_indices = train_indices(val_split.test);
    test_indices = find(test_mask);
    
    % Extract data for this fold
    XTrain = ALL_X_data(final_train_indices);
    YTrain = categorical(ALL_Y_data(final_train_indices));
    XValidation = ALL_X_data(val_indices);
    YValidation = categorical(ALL_Y_data(val_indices));
    XTest = ALL_X_data(test_indices);
    YTest = categorical(ALL_Y_data(test_indices));

    %% --- SUBJECT-WISE SPECTRAL NORMALIZATION ---
    train_subject_stats = containers.Map();
    for subj_idx = 1:length(unique_subjects)
        subj_id = unique_subjects{subj_idx};
        if ismember(subj_id, train_subjects)
            subj_train_mask = ismember(ALL_subject_ids(final_train_indices), subj_id);
            if sum(subj_train_mask) > 0
                subj_windows = XTrain(subj_train_mask);
                subj_spectra = cat(4, subj_windows{:});
                subj_spectra = reshape(subj_spectra, [], size(subj_spectra,4));
                subj_mean = mean(subj_spectra, 2);
                subj_std = std(subj_spectra, 0, 2) + eps;
                train_subject_stats(subj_id) = struct('mean', subj_mean, 'std', subj_std);
            end
        end
    end
    
    XTrain = apply_subject_normalization(XTrain, ALL_subject_ids(final_train_indices), train_subject_stats);
    XValidation = apply_subject_normalization(XValidation, ALL_subject_ids(val_indices), train_subject_stats);
    
    % For test subject, compute stats from their own data
    test_subj_id = test_subjects{1};
    test_windows = XTest;
    test_spectra = cat(4, test_windows{:});
    test_spectra = reshape(test_spectra, [], size(test_spectra,4));
    test_mean = mean(test_spectra, 2);
    test_std = std(test_spectra, 0, 2) + eps;
    test_stats = containers.Map();
    test_stats(test_subj_id) = struct('mean', test_mean, 'std', test_std);
    XTest = apply_subject_normalization(XTest, ALL_subject_ids(test_indices), test_stats);

    % Manual z-score for sequence data (legacy path)
    if ndims(XTrain{1}) == 2
        concatTrain = cat(2, XTrain{:});
        mu  = mean(concatTrain, 2);
        sig = std(concatTrain, 0, 2) + eps;
        normFn = @(x) (x - mu) ./ sig;
        XTrain      = cellfun(normFn, XTrain,      'UniformOutput', false);
        XValidation = cellfun(normFn, XValidation, 'UniformOutput', false);
        XTest       = cellfun(normFn, XTest,       'UniformOutput', false);
    end
    
    fprintf('  Training samples: %d\n', length(XTrain));
    fprintf('  Validation samples: %d\n', length(XValidation));
    fprintf('  Test samples: %d\n', length(XTest));
    
    %% --- EXPORT DATA FOR PYTHON ---
    fprintf('\nExporting data for fold %d...\n', fold);
    try
        step4_export_for_python(XTrain, YTrain, XValidation, YValidation, XTest, YTest, fold, ...
            final_train_indices, val_indices, test_indices, ALL_subject_ids);
        fold_accuracy = NaN;
    catch ME
        fprintf('Error in fold %d: %s\n', fold, ME.message);
        fold_accuracy = NaN;
    end
    
    cv_accuracies = [cv_accuracies, fold_accuracy];
end

%% --- FINAL RESULTS ---
fprintf('\n=== MATLAB PREPROCESSING COMPLETE ===\n');
fprintf('Data exported for %d folds to diagnostics/python_data/ directory\n', k_folds);
fprintf('\nNext steps:\n');
fprintf('Run: python dann.py\n');

% Save export completion flag
save('diagnostics/python_data/export_complete.mat', 'k_folds', 'unique_subjects');

fprintf('\nMATLAB preprocessing completed successfully!\n');
fprintf('\n*** Pipeline Complete! ***\n');

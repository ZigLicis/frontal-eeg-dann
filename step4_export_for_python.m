function step4_export_for_python(XTrain, YTrain, XValidation, YValidation, XTest, YTest, fold_num, ...
    final_train_indices, val_indices, test_indices, ALL_subject_ids)
% step4_export_for_python() - Export processed data for Python deep learning
%
% This function exports the preprocessed EEG data and labels to files that
% Python can easily load for domain adversarial training.
%
% Inputs:
%   XTrain, YTrain - Training data and labels (cell arrays)
%   XValidation, YValidation - Validation data and labels
%   XTest, YTest - Test data and labels
%   fold_num - Current fold number
%   final_train_indices, val_indices, test_indices - Data indices
%   ALL_subject_ids - Subject IDs for all data points
%

fprintf('Exporting fold %d data for Python...\n', fold_num);

% Create export directory
export_dir = fullfile(pwd, 'diagnostics/python_data');
if ~exist(export_dir, 'dir')
    mkdir(export_dir);
end

% Convert cell arrays to 4-D numeric arrays if needed
if iscell(XTrain)
    XTrain = cat(4, XTrain{:});
    XValidation = cat(4, XValidation{:});
    XTest = cat(4, XTest{:});
end

% Convert categorical labels to numeric
YTrain_numeric = double(YTrain);
YValidation_numeric = double(YValidation);
YTest_numeric = double(YTest);

% Create subject labels
unique_subjects = unique(ALL_subject_ids);
subject_map = containers.Map();
for i = 1:length(unique_subjects)
    subject_map(unique_subjects{i}) = i;
end

% Get subject IDs for each set
train_subjects = cell(length(final_train_indices), 1);
val_subjects = cell(length(val_indices), 1);
test_subjects = cell(length(test_indices), 1);

for i = 1:length(final_train_indices)
    train_subjects{i} = ALL_subject_ids{final_train_indices(i)};
end

for i = 1:length(val_indices)
    val_subjects{i} = ALL_subject_ids{val_indices(i)};
end

for i = 1:length(test_indices)
    test_subjects{i} = ALL_subject_ids{test_indices(i)};
end

% Convert subject IDs to numeric
train_subject_nums = zeros(length(train_subjects), 1);
val_subject_nums = zeros(length(val_subjects), 1);
test_subject_nums = zeros(length(test_subjects), 1);

for i = 1:length(train_subjects)
    train_subject_nums(i) = subject_map(train_subjects{i});
end

for i = 1:length(val_subjects)
    val_subject_nums(i) = subject_map(val_subjects{i});
end

for i = 1:length(test_subjects)
    test_subject_nums(i) = subject_map(test_subjects{i});
end

% Save data for this fold
fold_file = fullfile(export_dir, sprintf('fold_%d_data.mat', fold_num));

save(fold_file, ...
    'XTrain', 'YTrain_numeric', 'train_subject_nums', ...
    'XValidation', 'YValidation_numeric', 'val_subject_nums', ...
    'XTest', 'YTest_numeric', 'test_subject_nums', ...
    'unique_subjects', 'fold_num', '-v7.3');

fprintf('Exported fold %d data to: %s\n', fold_num, fold_file);

% Save metadata on first fold
if fold_num == 1
    metadata_file = fullfile(export_dir, 'metadata.mat');
    num_classes = length(unique(YTrain_numeric));
    num_subjects = length(unique_subjects);
    data_shape = size(XTrain);
    
    save(metadata_file, 'num_classes', 'num_subjects', 'data_shape', ...
         'unique_subjects', '-v7.3');
    
    fprintf('Exported metadata to: %s\n', metadata_file);
end

end 
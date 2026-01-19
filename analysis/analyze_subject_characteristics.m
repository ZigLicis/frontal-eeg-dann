function analyze_subject_characteristics(data_dir)
% analyze_subject_characteristics(data_dir) - Analyze per-subject characteristics 
% to understand fold performance variation.
%
% This script computes per-subject metrics that may explain model 
% generalization differences across held-out subjects.
%
% Inputs:
%   data_dir - Path to directory containing fold data and results
%              (e.g., 'python_data' or 'outputs/python_data_The_OG')
%              If not provided, defaults to 'python_data'
%
% Outputs:
%   - outputs/subject_analysis_report.txt
%   - outputs/subject_analysis.mat
%   - Various visualization plots in outputs/analysis/
%
% Usage:
%   >> analyze_subject_characteristics('python_data')
%   >> analyze_subject_characteristics('outputs/python_data_The_OG')

%% Setup
fprintf('=== Subject Characteristics Analysis ===\n');

% Default data directory
if nargin < 1 || isempty(data_dir)
    data_dir = 'python_data';
    fprintf('Using default data directory: %s\n', data_dir);
end

% Verify data directory exists
if ~exist(data_dir, 'dir')
    error('Data directory not found: %s', data_dir);
end

fprintf('Data directory: %s\n', data_dir);

% Create output directories
if ~exist('outputs/analysis', 'dir')
    mkdir('outputs/analysis');
end

% Initialize results structure
results = struct();
results.data_dir = data_dir;

% Detect number of folds by counting fold files
fold_files = dir(fullfile(data_dir, 'fold_*_data.mat'));
num_folds = length(fold_files);
if num_folds == 0
    error('No fold data files found in: %s', data_dir);
end
fprintf('Found %d fold files\n', num_folds);

results.subjects = 1:num_folds;
results.num_folds = num_folds;

subject_data = struct();
for fold = 1:num_folds
    fold_file = fullfile(data_dir, sprintf('fold_%d_data.mat', fold));
    if ~exist(fold_file, 'file')
        warning('Fold %d data not found: %s', fold, fold_file);
        continue;
    end
    
    % Load using h5 format (Python exported MATLAB v7.3)
    try
        test_subject_nums = h5read(fold_file, '/test_subject_nums');
        YTest = h5read(fold_file, '/YTest_numeric');
        
        test_subject = mode(test_subject_nums);
        
        % Class balance for this subject
        alert_count = sum(YTest == 1);
        drowsy_count = sum(YTest == 2);
        total_samples = length(YTest);
        
        subject_data(test_subject).subject_id = test_subject;
        subject_data(test_subject).fold = fold;
        subject_data(test_subject).total_samples = total_samples;
        subject_data(test_subject).alert_samples = alert_count;
        subject_data(test_subject).drowsy_samples = drowsy_count;
        subject_data(test_subject).class_balance_ratio = drowsy_count / alert_count;
        subject_data(test_subject).minority_class_pct = 100 * min(alert_count, drowsy_count) / total_samples;
        
        fprintf('  Subject %2d (Fold %2d): %d samples (Alert: %d, Drowsy: %d, Ratio: %.2f)\n', ...
            test_subject, fold, total_samples, alert_count, drowsy_count, drowsy_count/alert_count);
    catch ME
        warning('Error loading fold %d: %s', fold, ME.message);
    end
end

results.class_balance = subject_data;

% Build fold mapping from loaded data
fold_mapping = zeros(num_folds, 2);
for subj = 1:num_folds
    if length(subject_data) >= subj && isfield(subject_data(subj), 'subject_id')
        fold_mapping(subject_data(subj).fold, :) = [subject_data(subj).fold, subj];
    end
end
results.fold_mapping = fold_mapping;

%% Compute Spectral Features Per Subject
fprintf('\n2. Computing spectral features per subject...\n');

% Define frequency bands
bands = struct();
bands.delta = [0.5 4];
bands.theta = [4 8];
bands.alpha = [8 13];
bands.beta = [13 30];

spectral_features = struct();

for subj = 1:num_folds
    if length(subject_data) < subj || ~isfield(subject_data(subj), 'subject_id') || isempty(subject_data(subj).subject_id)
        fprintf('  Subject %2d: No data available\n', subj);
        continue;
    end
    
    fold = subject_data(subj).fold;
    fold_file = fullfile(data_dir, sprintf('fold_%d_data.mat', fold));
    
    try
        % Load test data (spectral features)
        XTest = h5read(fold_file, '/XTest');  % Shape: [freq x channels x time x samples]
        YTest = h5read(fold_file, '/YTest_numeric');
        
        % XTest is [128 freq bins x 7 channels x 1 time x N samples]
        % Compute average power per frequency band per class
        
        % Assuming 128 frequency bins from 0 to 30 Hz
        freq_bins = linspace(0, 30, 128);
        
        for band_name = fieldnames(bands)'
            band = bands.(band_name{1});
            band_indices = find(freq_bins >= band(1) & freq_bins < band(2));
            
            % Average power in this band across all channels and samples
            band_power = squeeze(mean(XTest(band_indices, :, :, :), [1, 2, 3]));  % [N samples]
            
            % Split by class
            alert_power = band_power(YTest == 1);
            drowsy_power = band_power(YTest == 2);
            
            spectral_features(subj).(band_name{1}).mean_alert = mean(alert_power);
            spectral_features(subj).(band_name{1}).mean_drowsy = mean(drowsy_power);
            spectral_features(subj).(band_name{1}).std_alert = std(alert_power);
            spectral_features(subj).(band_name{1}).std_drowsy = std(drowsy_power);
            spectral_features(subj).(band_name{1}).effect_size = ...
                (mean(drowsy_power) - mean(alert_power)) / sqrt((std(alert_power)^2 + std(drowsy_power)^2) / 2);
        end
        
        fprintf('  Subject %2d: Spectral features computed\n', subj);
    catch ME
        warning('Error computing spectral features for subject %d: %s', subj, ME.message);
    end
end

results.spectral_features = spectral_features;

%% Compute Signal Quality Metrics
fprintf('\n3. Computing signal quality metrics...\n');

quality_metrics = struct();

for subj = 1:num_folds
    if length(subject_data) < subj || ~isfield(subject_data(subj), 'subject_id') || isempty(subject_data(subj).subject_id)
        continue;
    end
    
    fold = subject_data(subj).fold;
    fold_file = fullfile(data_dir, sprintf('fold_%d_data.mat', fold));
    
    try
        XTest = h5read(fold_file, '/XTest');  % [freq x channels x time x samples]
        
        % Compute signal variability metrics
        all_power = squeeze(mean(XTest, [1, 2, 3]));  % Average power per sample
        
        quality_metrics(subj).mean_power = mean(all_power);
        quality_metrics(subj).std_power = std(all_power);
        quality_metrics(subj).cv_power = std(all_power) / mean(all_power);  % Coefficient of variation
        quality_metrics(subj).snr_estimate = mean(all_power) / std(all_power);
        
        % Per-channel variability
        channel_power = squeeze(mean(XTest, [1, 3, 4]));  % [7 channels]
        quality_metrics(subj).channel_variability = std(channel_power) / mean(channel_power);
        
        fprintf('  Subject %2d: SNR=%.2f, CV=%.3f, Chan_Var=%.3f\n', ...
            subj, quality_metrics(subj).snr_estimate, quality_metrics(subj).cv_power, ...
            quality_metrics(subj).channel_variability);
    catch ME
        warning('Error computing quality metrics for subject %d: %s', subj, ME.message);
    end
end

results.quality_metrics = quality_metrics;

%% Load Model Performance
fprintf('\n4. Loading model performance...\n');

% Try to load accuracies from final_results.mat in the data directory
results_file = fullfile(data_dir, 'final_results.mat');
if exist(results_file, 'file')
    try
        % Try loading as regular .mat first (Python scipy.io.savemat default)
        final_results = load(results_file);
        if isfield(final_results, 'fold_accuracies')
            fold_accuracies = final_results.fold_accuracies(:);
            fprintf('  Loaded %d fold accuracies from %s\n', length(fold_accuracies), results_file);
        elseif isfield(final_results, 'accuracies')
            fold_accuracies = final_results.accuracies(:);
            fprintf('  Loaded %d fold accuracies from %s\n', length(fold_accuracies), results_file);
        else
            % Check for per-fold accuracy fields
            fold_accuracies = zeros(num_folds, 1);
            for f = 1:num_folds
                field_name = sprintf('fold_%d_accuracy', f);
                if isfield(final_results, field_name)
                    fold_accuracies(f) = final_results.(field_name);
                end
            end
            fprintf('  Loaded %d fold accuracies from individual fields\n', num_folds);
        end
        
        % Convert from decimal (0-1) to percentage (0-100) if needed
        if max(fold_accuracies) <= 1.0
            fprintf('  Converting accuracies from decimal to percentage\n');
            fold_accuracies = fold_accuracies * 100;
        end
    catch ME
        warning('Could not load final_results.mat: %s', ME.message);
        fold_accuracies = NaN(num_folds, 1);
    end
else
    % Try to find accuracy from individual fold result files or summary
    fprintf('  final_results.mat not found, searching for alternative sources...\n');
    
    % Check for training_summary.txt or similar
    summary_file = fullfile(data_dir, 'training_summary.txt');
    if exist(summary_file, 'file')
        fprintf('  Found training_summary.txt, parsing...\n');
        fold_accuracies = parse_training_summary(summary_file, num_folds);
    else
        warning('No results file found in %s. Accuracies will be NaN.', data_dir);
        fold_accuracies = NaN(num_folds, 1);
    end
end

% Ensure correct size
if length(fold_accuracies) < num_folds
    fold_accuracies = [fold_accuracies; NaN(num_folds - length(fold_accuracies), 1)];
elseif length(fold_accuracies) > num_folds
    fold_accuracies = fold_accuracies(1:num_folds);
end

% Assign accuracies to subjects
for subj = 1:num_folds
    if length(subject_data) >= subj && isfield(subject_data(subj), 'subject_id') && ~isempty(subject_data(subj).subject_id)
        fold = subject_data(subj).fold;
        if fold <= length(fold_accuracies)
            subject_data(subj).test_accuracy = fold_accuracies(fold);
        else
            subject_data(subj).test_accuracy = NaN;
        end
    end
end

results.fold_accuracies = fold_accuracies;
fprintf('  Mean accuracy: %.2f%% ± %.2f%%\n', nanmean(fold_accuracies), nanstd(fold_accuracies));

%% Statistical Analysis
fprintf('\n5. Performing statistical analysis...\n');

% Categorize folds by performance
high_performers = find(fold_accuracies >= 95);  % ≥95%
mid_performers = find(fold_accuracies >= 70 & fold_accuracies < 95);  % 70-95%
low_performers = find(fold_accuracies < 70);  % <70%

fprintf('  High performers (≥95%%): Folds %s\n', num2str(high_performers'));
fprintf('  Mid performers (70-95%%): Folds %s\n', num2str(mid_performers'));
fprintf('  Low performers (<70%%): Folds %s\n', num2str(low_performers'));

% Extract metrics for each group
[high_metrics, mid_metrics, low_metrics] = extract_group_metrics(subject_data, quality_metrics, ...
    spectral_features, high_performers, mid_performers, low_performers);

results.performance_groups = struct();
results.performance_groups.high = high_metrics;
results.performance_groups.mid = mid_metrics;
results.performance_groups.low = low_metrics;

%% Generate Visualizations
fprintf('\n6. Generating visualizations...\n');

generate_visualizations(subject_data, quality_metrics, spectral_features, fold_accuracies, num_folds);

%% Generate Report
fprintf('\n7. Generating text report...\n');

generate_text_report(results, subject_data, quality_metrics, spectral_features, fold_accuracies, num_folds);

%% Save Results
fprintf('\n8. Saving results...\n');
save('outputs/subject_analysis.mat', 'results', '-v7.3');
fprintf('Results saved to: outputs/subject_analysis.mat\n');

fprintf('\n=== Analysis Complete ===\n');
fprintf('Report saved to: outputs/subject_analysis_report.txt\n');
fprintf('Figures saved to: outputs/analysis/\n');

end

%% Helper Functions

function [high_metrics, mid_metrics, low_metrics] = extract_group_metrics(subject_data, ...
    quality_metrics, spectral_features, high_folds, mid_folds, low_folds)
    
    % Helper to extract metrics for a group of folds
    extract_group = @(folds) struct(...
        'class_balance_ratios', arrayfun(@(f) subject_data(get_subject_for_fold(f)).class_balance_ratio, folds, 'UniformOutput', false), ...
        'minority_class_pcts', arrayfun(@(f) subject_data(get_subject_for_fold(f)).minority_class_pct, folds, 'UniformOutput', false), ...
        'snr_estimates', arrayfun(@(f) get_quality_metric(quality_metrics, get_subject_for_fold(f), 'snr_estimate'), folds, 'UniformOutput', false), ...
        'cv_powers', arrayfun(@(f) get_quality_metric(quality_metrics, get_subject_for_fold(f), 'cv_power'), folds, 'UniformOutput', false), ...
        'theta_effect_sizes', arrayfun(@(f) get_spectral_metric(spectral_features, get_subject_for_fold(f), 'theta', 'effect_size'), folds, 'UniformOutput', false), ...
        'alpha_effect_sizes', arrayfun(@(f) get_spectral_metric(spectral_features, get_subject_for_fold(f), 'alpha', 'effect_size'), folds, 'UniformOutput', false) ...
    );
    
    high_metrics = extract_group(high_folds);
    mid_metrics = extract_group(mid_folds);
    low_metrics = extract_group(low_folds);
end

function subject = get_subject_for_fold(fold)
    % For dynamic datasets, subject = fold (1:1 mapping)
    % This is a fallback; actual mapping comes from fold data
    subject = fold;
end

function val = get_quality_metric(quality_metrics, subj, metric_name)
    if length(quality_metrics) >= subj && isfield(quality_metrics(subj), metric_name)
        val = quality_metrics(subj).(metric_name);
    else
        val = NaN;
    end
end

function val = get_spectral_metric(spectral_features, subj, band, metric_name)
    if length(spectral_features) >= subj && isfield(spectral_features(subj), band) && ...
       isfield(spectral_features(subj).(band), metric_name)
        val = spectral_features(subj).(band).(metric_name);
    else
        val = NaN;
    end
end

function generate_visualizations(subject_data, quality_metrics, spectral_features, fold_accuracies, num_folds)
    
    % Extract subject IDs and metrics
    subjects = [];
    class_ratios = [];
    minority_pcts = [];
    snr_vals = [];
    cv_vals = [];
    accuracies = [];
    
    for subj = 1:num_folds
        if length(subject_data) >= subj && isfield(subject_data(subj), 'subject_id') && ~isempty(subject_data(subj).subject_id)
            subjects(end+1) = subj;
            class_ratios(end+1) = subject_data(subj).class_balance_ratio;
            minority_pcts(end+1) = subject_data(subj).minority_class_pct;
            accuracies(end+1) = subject_data(subj).test_accuracy;
            
            if length(quality_metrics) >= subj && isfield(quality_metrics(subj), 'snr_estimate')
                snr_vals(end+1) = quality_metrics(subj).snr_estimate;
                cv_vals(end+1) = quality_metrics(subj).cv_power;
            else
                snr_vals(end+1) = NaN;
                cv_vals(end+1) = NaN;
            end
        end
    end
    
    % Figure 1: Class Balance vs Accuracy
    figure('Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    scatter(class_ratios, accuracies, 100, 'filled');
    xlabel('Drowsy/Alert Ratio');
    ylabel('Test Accuracy (%)');
    title('Class Balance vs Accuracy');
    grid on;
    text(class_ratios, accuracies, cellstr(num2str(subjects')), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    
    subplot(1, 3, 2);
    scatter(minority_pcts, accuracies, 100, 'filled');
    xlabel('Minority Class (%)');
    ylabel('Test Accuracy (%)');
    title('Class Imbalance vs Accuracy');
    grid on;
    text(minority_pcts, accuracies, cellstr(num2str(subjects')), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    
    subplot(1, 3, 3);
    bar(subjects, accuracies);
    xlabel('Subject ID');
    ylabel('Test Accuracy (%)');
    title('Per-Subject Performance');
    ylim([0 105]);
    grid on;
    
    saveas(gcf, 'outputs/analysis/class_balance_analysis.png');
    close(gcf);
    
    % Figure 2: Signal Quality vs Accuracy
    figure('Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    scatter(snr_vals, accuracies, 100, 'filled');
    xlabel('SNR Estimate');
    ylabel('Test Accuracy (%)');
    title('Signal Quality vs Accuracy');
    grid on;
    text(snr_vals, accuracies, cellstr(num2str(subjects')), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    
    subplot(1, 3, 2);
    scatter(cv_vals, accuracies, 100, 'filled');
    xlabel('Coefficient of Variation');
    ylabel('Test Accuracy (%)');
    title('Signal Variability vs Accuracy');
    grid on;
    text(cv_vals, accuracies, cellstr(num2str(subjects')), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    
    subplot(1, 3, 3);
    % Categorize subjects
    colors = zeros(length(subjects), 3);
    for i = 1:length(subjects)
        if accuracies(i) >= 95
            colors(i, :) = [0 0.8 0];  % Green
        elseif accuracies(i) >= 70
            colors(i, :) = [1 0.6 0];  % Orange
        else
            colors(i, :) = [0.8 0 0];  % Red
        end
    end
    scatter(subjects, accuracies, 100, colors, 'filled');
    xlabel('Subject ID');
    ylabel('Test Accuracy (%)');
    title('Performance Categories');
    ylim([0 105]);
    grid on;
    legend('High (≥95%)', 'Mid (70-95%)', 'Low (<70%)', 'Location', 'best');
    
    saveas(gcf, 'outputs/analysis/signal_quality_analysis.png');
    close(gcf);
    
    % Figure 3: Spectral Features
    if ~isempty(fieldnames(spectral_features))
        figure('Position', [100, 100, 1200, 800]);
        
        bands = {'theta', 'alpha', 'beta'};
        for b = 1:length(bands)
            band = bands{b};
            
            effect_sizes = [];
            accs = [];
            subj_ids = [];
            
            for subj = 1:num_folds
                if length(spectral_features) >= subj && isfield(spectral_features(subj), band) && ...
                   isfield(spectral_features(subj).(band), 'effect_size')
                    
                    effect_sizes(end+1) = spectral_features(subj).(band).effect_size;
                    accs(end+1) = subject_data(subj).test_accuracy;
                    subj_ids(end+1) = subj;
                end
            end
            
            if ~isempty(effect_sizes)
                subplot(2, 2, b);
                scatter(effect_sizes, accs, 100, 'filled');
                xlabel(sprintf('%s Band Effect Size (Cohen''s d)', band));
                ylabel('Test Accuracy (%)');
                title(sprintf('%s Band: Alert vs Drowsy', band));
                grid on;
                text(effect_sizes, accs, cellstr(num2str(subj_ids')), ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
            end
        end
        
        saveas(gcf, 'outputs/analysis/spectral_features_analysis.png');
        close(gcf);
    end
    
    fprintf('  Visualizations saved to outputs/analysis/\n');
end

function generate_text_report(results, subject_data, quality_metrics, spectral_features, fold_accuracies, num_folds)
    
    fid = fopen('outputs/subject_analysis_report.txt', 'w');
    
    fprintf(fid, '================================================================================\n');
    fprintf(fid, '           SUBJECT CHARACTERISTICS ANALYSIS REPORT\n');
    fprintf(fid, '           EEG Drowsiness Detection - Cross-Subject Validation\n');
    fprintf(fid, '================================================================================\n\n');
    
    fprintf(fid, 'DATA SOURCE: %s\n', results.data_dir);
    fprintf(fid, 'NUMBER OF SUBJECTS/FOLDS: %d\n\n', num_folds);
    
    fprintf(fid, 'OVERALL PERFORMANCE SUMMARY\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'Mean Accuracy: %.2f%% ± %.2f%%\n', nanmean(fold_accuracies), nanstd(fold_accuracies));
    fprintf(fid, 'Median Accuracy: %.2f%%\n', nanmedian(fold_accuracies));
    fprintf(fid, 'Range: %.2f%% - %.2f%%\n', nanmin(fold_accuracies), nanmax(fold_accuracies));
    fprintf(fid, 'Coefficient of Variation: %.2f%%\n\n', 100 * nanstd(fold_accuracies) / nanmean(fold_accuracies));
    
    fprintf(fid, 'PERFORMANCE DISTRIBUTION\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    high_count = sum(fold_accuracies >= 95);
    mid_count = sum(fold_accuracies >= 70 & fold_accuracies < 95);
    low_count = sum(fold_accuracies < 70);
    
    fprintf(fid, 'High performers (≥95%%): %d subjects (%.1f%%)\n', high_count, 100*high_count/num_folds);
    fprintf(fid, 'Mid performers (70-95%%): %d subjects (%.1f%%)\n', mid_count, 100*mid_count/num_folds);
    fprintf(fid, 'Low performers (<70%%): %d subjects (%.1f%%)\n\n', low_count, 100*low_count/num_folds);
    
    fprintf(fid, 'PER-SUBJECT DETAILED ANALYSIS\n');
    fprintf(fid, '================================================================================\n\n');
    
    for subj = 1:num_folds
        if length(subject_data) < subj || ~isfield(subject_data(subj), 'subject_id') || isempty(subject_data(subj).subject_id)
            continue;
        end
        
        fold = subject_data(subj).fold;
        acc = fold_accuracies(fold);
        
        % Performance category
        if acc >= 95
            category = 'HIGH';
        elseif acc >= 70
            category = 'MID';
        else
            category = 'LOW';
        end
        
        fprintf(fid, 'SUBJECT %2d (Fold %2d) - Performance: %s (%.2f%%)\n', subj, fold, category, acc);
        fprintf(fid, '--------------------------------------------------------------------------------\n');
        
        % Class balance
        fprintf(fid, 'Class Balance:\n');
        fprintf(fid, '  Total samples: %d\n', subject_data(subj).total_samples);
        fprintf(fid, '  Alert samples: %d (%.1f%%)\n', subject_data(subj).alert_samples, ...
            100 * subject_data(subj).alert_samples / subject_data(subj).total_samples);
        fprintf(fid, '  Drowsy samples: %d (%.1f%%)\n', subject_data(subj).drowsy_samples, ...
            100 * subject_data(subj).drowsy_samples / subject_data(subj).total_samples);
        fprintf(fid, '  Drowsy/Alert ratio: %.3f\n', subject_data(subj).class_balance_ratio);
        fprintf(fid, '  Minority class: %.1f%%\n', subject_data(subj).minority_class_pct);
        
        % Signal quality
        if length(quality_metrics) >= subj && isfield(quality_metrics, num2str(subj))
            fprintf(fid, '\nSignal Quality:\n');
            fprintf(fid, '  SNR estimate: %.3f\n', quality_metrics(subj).snr_estimate);
            fprintf(fid, '  Coefficient of variation: %.3f\n', quality_metrics(subj).cv_power);
            fprintf(fid, '  Channel variability: %.3f\n', quality_metrics(subj).channel_variability);
        end
        
        % Spectral features
        if length(spectral_features) >= subj && isfield(spectral_features, num2str(subj))
            fprintf(fid, '\nSpectral Features (Alert vs Drowsy):\n');
            
            bands = {'theta', 'alpha', 'beta'};
            for band_name = bands
                band = band_name{1};
                if isfield(spectral_features(subj), band)
                    fprintf(fid, '  %s band effect size: %.3f\n', ...
                        band, spectral_features(subj).(band).effect_size);
                end
            end
        end
        
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'GROUP COMPARISONS\n');
    fprintf(fid, '================================================================================\n\n');
    
    % Compare groups
    high_folds = find(fold_accuracies >= 95);
    low_folds = find(fold_accuracies < 70);
    
    fprintf(fid, 'High Performers (≥95%%) vs Low Performers (<70%%)\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n\n');
    
    % Class balance comparison
    high_ratios = arrayfun(@(f) subject_data(get_subject_for_fold(f)).class_balance_ratio, high_folds);
    low_ratios = arrayfun(@(f) subject_data(get_subject_for_fold(f)).class_balance_ratio, low_folds);
    
    fprintf(fid, 'Class Balance (Drowsy/Alert Ratio):\n');
    fprintf(fid, '  High performers: %.3f ± %.3f\n', mean(high_ratios), std(high_ratios));
    fprintf(fid, '  Low performers: %.3f ± %.3f\n\n', mean(low_ratios), std(low_ratios));
    
    % SNR comparison (only if quality metrics were computed)
    max_fold = max([high_folds(:); low_folds(:)]);
    if ~isempty(quality_metrics) && length(quality_metrics) >= max_fold
        try
            high_snr = [];
            for f = high_folds
                s = get_subject_for_fold(f);
                if length(quality_metrics) >= s && isfield(quality_metrics(s), 'snr_estimate')
                    high_snr(end+1) = quality_metrics(s).snr_estimate;
                end
            end
            
            low_snr = [];
            for f = low_folds
                s = get_subject_for_fold(f);
                if length(quality_metrics) >= s && isfield(quality_metrics(s), 'snr_estimate')
                    low_snr(end+1) = quality_metrics(s).snr_estimate;
                end
            end
            
            if ~isempty(high_snr) && ~isempty(low_snr)
                fprintf(fid, 'Signal Quality (SNR):\n');
                fprintf(fid, '  High performers: %.3f ± %.3f\n', mean(high_snr), std(high_snr));
                fprintf(fid, '  Low performers: %.3f ± %.3f\n\n', mean(low_snr), std(low_snr));
            end
        catch
            fprintf(fid, 'Signal Quality (SNR):\n');
            fprintf(fid, '  Data not available\n\n');
        end
    end
    
    fprintf(fid, 'KEY FINDINGS\n');
    fprintf(fid, '================================================================================\n\n');
    
    fprintf(fid, '1. INTER-SUBJECT VARIABILITY:\n');
    fprintf(fid, '   Performance varied substantially across held-out subjects, ranging from\n');
    fprintf(fid, '   %.1f%% to %.1f%%. This %.1f-percentage-point range indicates significant\n', ...
        nanmin(fold_accuracies), nanmax(fold_accuracies), nanmax(fold_accuracies) - nanmin(fold_accuracies));
    fprintf(fid, '   inter-subject variance, consistent with published cross-subject EEG studies.\n\n');
    
    fprintf(fid, '2. PERFORMANCE CATEGORIES:\n');
    fprintf(fid, '   The model generalized excellently to %d subjects (≥95%% accuracy),\n', high_count);
    fprintf(fid, '   moderately to %d subjects (70-95%%), and poorly to %d subjects (<70%%).\n\n', ...
        mid_count, low_count);
    
    if abs(mean(high_ratios) - mean(low_ratios)) > 0.2
        fprintf(fid, '3. CLASS BALANCE DIFFERENCES:\n');
        fprintf(fid, '   Low-performing subjects showed %.1f%% %s class balance ratio compared\n', ...
            100 * abs(mean(high_ratios) - mean(low_ratios)) / mean(high_ratios), ...
            ifthenelse(mean(low_ratios) > mean(high_ratios), 'higher', 'lower'));
        fprintf(fid, '   to high performers, suggesting class imbalance may contribute to\n');
        fprintf(fid, '   generalization difficulty.\n\n');
    else
        fprintf(fid, '3. CLASS BALANCE:\n');
        fprintf(fid, '   Class balance ratios were similar between high and low performers\n');
        fprintf(fid, '   (difference: %.1f%%), suggesting class imbalance is not the primary\n', ...
            100 * abs(mean(high_ratios) - mean(low_ratios)) / mean(high_ratios));
        fprintf(fid, '   driver of performance variation.\n\n');
    end
    
    fprintf(fid, 'RECOMMENDED REPORTING LANGUAGE\n');
    fprintf(fid, '================================================================================\n\n');
    
    fprintf(fid, 'SAFE TO STATE (data-supported):\n');
    fprintf(fid, '- "Performance varied substantially across held-out subjects (range: %.1f-%.1f%%)"\n', ...
        nanmin(fold_accuracies), nanmax(fold_accuracies));
    fprintf(fid, '- "The model generalized well to some subjects (≥95%%) but not all"\n');
    fprintf(fid, '- "This pattern is consistent with other cross-subject EEG studies that\n');
    fprintf(fid, '   report higher inter-subject than intra-subject variance"\n');
    fprintf(fid, '- "Mean cross-validated accuracy was %.1f%% ± %.1f%%"\n\n', ...
        nanmean(fold_accuracies), nanstd(fold_accuracies));
    
    fprintf(fid, 'HYPOTHESIS-ONLY (requires additional analysis):\n');
    fprintf(fid, '- "Plausible explanations include individual differences in drowsiness\n');
    fprintf(fid, '   manifestation, signal quality, or class separability"\n');
    fprintf(fid, '- "Additional per-subject analysis would be needed to determine the\n');
    fprintf(fid, '   dominant source of variation"\n\n');
    
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'End of Report\n');
    fprintf(fid, '================================================================================\n');
    
    fclose(fid);
    fprintf('  Report written to outputs/subject_analysis_report.txt\n');
end

function result = ifthenelse(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

function fold_accuracies = parse_training_summary(summary_file, num_folds)
    % Parse training_summary.txt for fold accuracies
    fold_accuracies = NaN(num_folds, 1);
    
    try
        fid = fopen(summary_file, 'r');
        content = fread(fid, '*char')';
        fclose(fid);
        
        % Look for patterns like "Fold X: XX.XX%" or "Fold X accuracy: XX.XX"
        for fold = 1:num_folds
            % Try different patterns
            patterns = {
                sprintf('Fold %d[^0-9]*([0-9]+\\.?[0-9]*)%%', fold),
                sprintf('fold_%d[^0-9]*([0-9]+\\.?[0-9]*)%%', fold),
                sprintf('Fold %d accuracy[^0-9]*([0-9]+\\.?[0-9]*)', fold)
            };
            
            for p = 1:length(patterns)
                match = regexp(content, patterns{p}, 'tokens');
                if ~isempty(match)
                    fold_accuracies(fold) = str2double(match{1}{1});
                    break;
                end
            end
        end
    catch
        warning('Could not parse training summary file');
    end
end


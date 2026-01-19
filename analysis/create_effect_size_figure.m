function create_effect_size_figure(data_dir)
% create_effect_size_figure(data_dir) - Create publication-quality figure showing
% relationship between spectral effect sizes and model performance
%
% This function generates a multi-panel figure demonstrating that subjects
% with larger spectral separability (effect sizes) between alert and drowsy
% states achieve higher model accuracy.
%
% Inputs:
%   data_dir - Path to directory containing fold data and results
%              (e.g., 'python_data' or 'outputs/python_data_The_OG')
%              If not provided, loads from 'outputs/subject_analysis.mat'
%
% Outputs:
%   - outputs/analysis/effect_size_vs_accuracy.png
%   - outputs/analysis/effect_size_vs_accuracy.pdf (vector)
%   - outputs/analysis/effect_size_vs_accuracy_combined.png
%
% Usage:
%   >> create_effect_size_figure()  % Uses existing subject_analysis.mat
%   >> create_effect_size_figure('python_data')  % Runs analysis first

fprintf('=== Creating Effect Size vs Accuracy Figure ===\n');

% Load or compute results
if nargin >= 1 && ~isempty(data_dir)
    fprintf('Running analysis on data directory: %s\n', data_dir);
    % Run the analysis first to generate subject_analysis.mat
    analyze_subject_characteristics(data_dir);
end

% Load results
if ~exist('outputs/subject_analysis.mat', 'file')
    error('subject_analysis.mat not found. Run analyze_subject_characteristics(data_dir) first.');
end
load('outputs/subject_analysis.mat');

% Get number of subjects from results
if isfield(results, 'num_folds')
    num_subjects = results.num_folds;
else
    num_subjects = length(results.fold_accuracies);
end
fprintf('Processing %d subjects\n', num_subjects);

% Create subject-to-accuracy mapping
% For dynamic datasets, we use direct fold->subject mapping (subject = fold index)
subject_accuracy = zeros(num_subjects, 1);
for fold = 1:num_subjects
    % Find which subject corresponds to this fold
    if isfield(results, 'fold_mapping') && size(results.fold_mapping, 1) >= fold
        subj = results.fold_mapping(fold, 2);
        if subj == 0
            subj = fold;  % Fallback: assume fold = subject
        end
    else
        subj = fold;  % Default: fold index = subject index
    end
    if subj > 0 && subj <= num_subjects
        subject_accuracy(subj) = results.fold_accuracies(fold);
    end
end

% Extract effect sizes for all subjects
subjects = 1:num_subjects;
theta_effects = zeros(num_subjects, 1);
alpha_effects = zeros(num_subjects, 1);
beta_effects = zeros(num_subjects, 1);

for subj = 1:num_subjects
    if length(results.spectral_features) >= subj && ...
       isfield(results.spectral_features(subj), 'theta') && ...
       isfield(results.spectral_features(subj).theta, 'effect_size')
        theta_effects(subj) = results.spectral_features(subj).theta.effect_size;
        alpha_effects(subj) = results.spectral_features(subj).alpha.effect_size;
        beta_effects(subj) = results.spectral_features(subj).beta.effect_size;
    else
        theta_effects(subj) = NaN;
        alpha_effects(subj) = NaN;
        beta_effects(subj) = NaN;
    end
end

% Remove subjects with NaN values for plotting
valid_subjects = ~isnan(subject_accuracy) & ~isnan(theta_effects);
if sum(valid_subjects) < num_subjects
    fprintf('  Note: %d subjects have missing data and will be excluded from plots\n', ...
        num_subjects - sum(valid_subjects));
end

% Categorize subjects (only valid ones)
high_perf = subject_accuracy >= 95 & valid_subjects;
mid_perf = subject_accuracy >= 70 & subject_accuracy < 95 & valid_subjects;
low_perf = subject_accuracy < 70 & valid_subjects;

%% Figure 1: Individual Band Analysis (3 panels)
fig1 = figure('Position', [100, 100, 1400, 450]);
set(fig1, 'Color', 'w');

bands = {'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)'};
effect_data = {theta_effects, alpha_effects, beta_effects};

for b = 1:3
    subplot(1, 3, b);
    hold on; grid on; box on;
    
    effects = effect_data{b};
    
    % Plot points with color coding (only valid subjects)
    if any(low_perf)
        scatter(effects(low_perf), subject_accuracy(low_perf), 120, [0.8 0.2 0.2], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end
    if any(mid_perf)
        scatter(effects(mid_perf), subject_accuracy(mid_perf), 120, [1.0 0.6 0.0], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end
    if any(high_perf)
        scatter(effects(high_perf), subject_accuracy(high_perf), 120, [0.2 0.7 0.2], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end
    
    % Add subject labels (only valid subjects)
    for s = find(valid_subjects)'
        text(effects(s), subject_accuracy(s), sprintf(' %d', s), ...
            'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
    end
    
    % Compute and plot correlation line (only valid data)
    valid_effects = effects(valid_subjects);
    valid_acc = subject_accuracy(valid_subjects);
    [r, p] = corrcoef(valid_effects, valid_acc);
    r_val = r(1,2);
    p_val = p(1,2);
    
    % Fit line
    if abs(r_val) > 0.3  % Only plot line if correlation is reasonable
        p_fit = polyfit(valid_effects, valid_acc, 1);
        x_fit = linspace(min(valid_effects), max(valid_effects), 100);
        y_fit = polyval(p_fit, x_fit);
        plot(x_fit, y_fit, 'k--', 'LineWidth', 1.5);
    end
    
    % Styling
    xlabel(sprintf('%s Effect Size (Cohen''s d)', bands{b}), 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Test Accuracy (%)', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('%s Band', bands{b}), 'FontSize', 15, 'FontWeight', 'bold');
    
    % Add correlation text
    text(0.05, 0.95, sprintf('r = %.3f\np = %.3f', r_val, p_val), ...
        'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', 'EdgeColor', 'k', 'VerticalAlignment', 'top');
    
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);
    ylim([45 105]);
    
    % Add legend to each panel
    legend({'Low (<70%)', 'Mid (70-95%)', 'High (≥95%)'}, ...
        'Location', 'southeast', 'FontSize', 12, 'FontWeight', 'bold');
end

% Add main title
sgtitle('Spectral Separability vs Model Performance', 'FontSize', 17, 'FontWeight', 'bold');

% Save
saveas(fig1, 'outputs/analysis/effect_size_vs_accuracy.png');
try
    saveas(fig1, 'outputs/analysis/effect_size_vs_accuracy.pdf');
catch
    fprintf('  Note: Could not save PDF (requires vector graphics support)\n');
end
fprintf('  Saved: outputs/analysis/effect_size_vs_accuracy.png\n');

%% Figure 2: Combined Effect Size Analysis
fig2 = figure('Position', [100, 100, 1200, 500]);
set(fig2, 'Color', 'w');

% Panel A: Average absolute effect size vs accuracy
subplot(1, 2, 1);
hold on; grid on; box on;

% Compute average absolute effect size
avg_effect_size = mean(abs([theta_effects, alpha_effects, beta_effects]), 2);

if any(low_perf)
    scatter(avg_effect_size(low_perf), subject_accuracy(low_perf), 150, [0.8 0.2 0.2], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end
if any(mid_perf)
    scatter(avg_effect_size(mid_perf), subject_accuracy(mid_perf), 150, [1.0 0.6 0.0], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end
if any(high_perf)
    scatter(avg_effect_size(high_perf), subject_accuracy(high_perf), 150, [0.2 0.7 0.2], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end

% Add subject labels (only valid subjects)
for s = find(valid_subjects)'
    text(avg_effect_size(s), subject_accuracy(s), sprintf(' %d', s), ...
        'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end

% Correlation (only valid data)
valid_avg = avg_effect_size(valid_subjects);
valid_acc = subject_accuracy(valid_subjects);
[r, p] = corrcoef(valid_avg, valid_acc);
r_val = r(1,2);
p_val = p(1,2);

% Fit line
p_fit = polyfit(valid_avg, valid_acc, 1);
x_fit = linspace(min(valid_avg), max(valid_avg), 100);
y_fit = polyval(p_fit, x_fit);
plot(x_fit, y_fit, 'k--', 'LineWidth', 2);

xlabel('Mean Absolute Effect Size (|d|)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Test Accuracy (%)', 'FontSize', 14, 'FontWeight', 'bold');
title('(A) Combined Band Analysis', 'FontSize', 15, 'FontWeight', 'bold');

text(0.05, 0.95, sprintf('r = %.3f\np = %.3f', r_val, p_val), ...
    'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'VerticalAlignment', 'top');

legend({'Low (<70%)', 'Mid (70-95%)', 'High (≥95%)'}, 'Location', 'southeast', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
ylim([45 105]);

% Panel B: Group comparison bar plot
subplot(1, 2, 2);
hold on; grid on; box on;

% Compute group means (handle empty groups)
groups = {'Theta', 'Alpha', 'Beta'};

if any(low_perf)
    low_means = [mean(abs(theta_effects(low_perf))), mean(abs(alpha_effects(low_perf))), mean(abs(beta_effects(low_perf)))];
    low_stds = [std(abs(theta_effects(low_perf))), std(abs(alpha_effects(low_perf))), std(abs(beta_effects(low_perf)))];
else
    low_means = [0, 0, 0];
    low_stds = [0, 0, 0];
end

if any(high_perf)
    high_means = [mean(abs(theta_effects(high_perf))), mean(abs(alpha_effects(high_perf))), mean(abs(beta_effects(high_perf)))];
    high_stds = [std(abs(theta_effects(high_perf))), std(abs(alpha_effects(high_perf))), std(abs(beta_effects(high_perf)))];
else
    high_means = [0, 0, 0];
    high_stds = [0, 0, 0];
end

x = 1:3;
width = 0.35;

% Plot bars
b1 = bar(x - width/2, low_means, width, 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
b2 = bar(x + width/2, high_means, width, 'FaceColor', [0.2 0.7 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);

% Add error bars
errorbar(x - width/2, low_means, low_stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
errorbar(x + width/2, high_means, high_stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);

% Styling
set(gca, 'XTick', x, 'XTickLabel', groups, 'FontSize', 12, 'LineWidth', 1.2);
ylabel('Mean Absolute Effect Size (|d|)', 'FontSize', 14, 'FontWeight', 'bold');
title('(B) Group Comparison', 'FontSize', 15, 'FontWeight', 'bold');
legend({'Low Performers (<70%)', 'High Performers (≥95%)'}, 'Location', 'northwest', 'FontSize', 12, 'FontWeight', 'bold');

% Add significance stars (if difference > 2*pooled_std, add star)
for i = 1:3
    pooled_std = sqrt((low_stds(i)^2 + high_stds(i)^2) / 2);
    if (high_means(i) - low_means(i)) > 2*pooled_std
        % Add star
        y_pos = max([low_means(i), high_means(i)]) + max([low_stds(i), high_stds(i)]) + 0.5;
        text(i, y_pos, '*', 'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

% Save
saveas(fig2, 'outputs/analysis/effect_size_vs_accuracy_combined.png');
try
    saveas(fig2, 'outputs/analysis/effect_size_vs_accuracy_combined.pdf');
catch
end
fprintf('  Saved: outputs/analysis/effect_size_vs_accuracy_combined.png\n');

%% Figure 3: Detailed per-subject breakdown
fig3 = figure('Position', [100, 100, 1400, 600]);
set(fig3, 'Color', 'w');

% Top panel: Effect sizes per subject
subplot(2, 1, 1);
hold on; grid on; box on;

x_subj = 1:num_subjects;
width = 0.25;

% Replace NaN with 0 for bar plotting
theta_plot = theta_effects; theta_plot(isnan(theta_plot)) = 0;
alpha_plot = alpha_effects; alpha_plot(isnan(alpha_plot)) = 0;
beta_plot = beta_effects; beta_plot(isnan(beta_plot)) = 0;

b1 = bar(x_subj - width, theta_plot, width, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'k', 'LineWidth', 1.2);
b2 = bar(x_subj, alpha_plot, width, 'FaceColor', [0.9 0.6 0.2], 'EdgeColor', 'k', 'LineWidth', 1.2);
b3 = bar(x_subj + width, beta_plot, width, 'FaceColor', [0.5 0.2 0.6], 'EdgeColor', 'k', 'LineWidth', 1.2);

% Add horizontal line at 0
plot([0 num_subjects+1], [0 0], 'k--', 'LineWidth', 1);

% Styling
set(gca, 'XTick', 1:num_subjects, 'XTickLabel', 1:num_subjects, 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Subject ID', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Effect Size (Cohen''s d)', 'FontSize', 14, 'FontWeight', 'bold');
title('(A) Spectral Effect Sizes by Subject', 'FontSize', 15, 'FontWeight', 'bold');
legend({'Theta', 'Alpha', 'Beta'}, 'Location', 'northwest', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0 num_subjects+1]);

% Bottom panel: Accuracy per subject
subplot(2, 1, 2);
hold on; grid on; box on;

% Color bars by performance
colors = zeros(num_subjects, 3);
for s = 1:num_subjects
    if isnan(subject_accuracy(s))
        colors(s, :) = [0.5 0.5 0.5];  % Gray for missing
    elseif subject_accuracy(s) >= 95
        colors(s, :) = [0.2 0.7 0.2];  % Green
    elseif subject_accuracy(s) >= 70
        colors(s, :) = [1.0 0.6 0.0];  % Orange
    else
        colors(s, :) = [0.8 0.2 0.2];  % Red
    end
end

acc_plot = subject_accuracy; acc_plot(isnan(acc_plot)) = 0;
for s = 1:num_subjects
    bar(s, acc_plot(s), 'FaceColor', colors(s, :), 'EdgeColor', 'k', 'LineWidth', 1.2);
end

% Add performance threshold lines
plot([0 num_subjects+1], [95 95], 'g--', 'LineWidth', 1.5);
plot([0 num_subjects+1], [70 70], 'r--', 'LineWidth', 1.5);

% Styling
set(gca, 'XTick', 1:num_subjects, 'XTickLabel', 1:num_subjects, 'FontSize', 12, 'LineWidth', 1.2);
xlabel('Subject ID', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Test Accuracy (%)', 'FontSize', 14, 'FontWeight', 'bold');
title('(B) Model Performance by Subject', 'FontSize', 15, 'FontWeight', 'bold');
xlim([0 num_subjects+1]);
ylim([0 105]);

% Save
saveas(fig3, 'outputs/analysis/effect_size_per_subject.png');
try
    saveas(fig3, 'outputs/analysis/effect_size_per_subject.pdf');
catch
end
fprintf('  Saved: outputs/analysis/effect_size_per_subject.png\n');

%% Print summary statistics
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Number of subjects: %d\n', num_subjects);
fprintf('Correlation between mean |effect size| and accuracy: r=%.3f, p=%.4f\n', r_val, p_val);

fprintf('\nPerformance distribution:\n');
fprintf('  High performers (≥95%%): %d subjects\n', sum(high_perf));
fprintf('  Mid performers (70-95%%): %d subjects\n', sum(mid_perf));
fprintf('  Low performers (<70%%): %d subjects\n', sum(low_perf));

if any(high_perf) && any(low_perf)
    fprintf('\nEffect Size Ratios (High/Low Performers):\n');
    fprintf('  Theta: %.1fx\n', mean(abs(theta_effects(high_perf))) / mean(abs(theta_effects(low_perf))));
    fprintf('  Alpha: %.1fx\n', mean(abs(alpha_effects(high_perf))) / mean(abs(alpha_effects(low_perf))));
    fprintf('  Beta: %.1fx\n', mean(abs(beta_effects(high_perf))) / mean(abs(beta_effects(low_perf))));
else
    fprintf('\nNote: Cannot compute effect size ratios (need both high and low performers)\n');
end

fprintf('\n=== Figure Generation Complete ===\n');
fprintf('Figures saved to: outputs/analysis/\n');

end


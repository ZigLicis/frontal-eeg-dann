function visualize_eeg_timeseries(EEG, title_str, start_time_sec, duration_sec, save_path)
% visualize_eeg_timeseries() - Plot EEG time series for validation
%
% This function creates a stacked multi-channel plot of EEG data to help
% visualize the effects of preprocessing and artifact removal.
%
% Usage:
%   >> visualize_eeg_timeseries(EEG, 'After Filtering');  % Shows entire recording
%   >> visualize_eeg_timeseries(EEG, 'After ICA', [], [], 'outputs/viz/after_ica.png');  % Entire recording, saved
%   >> visualize_eeg_timeseries(EEG, 'After ICA', 30, 10, 'outputs/viz/after_ica.png');  % 10 sec starting at 30 sec
%
% Inputs:
%   EEG             - EEGLAB dataset structure
%   title_str       - Title for the plot
%   start_time_sec  - Starting time in seconds (default: 0, use [] for default)
%   duration_sec    - Duration to plot in seconds (default: entire recording, use [] for default)
%   save_path       - Optional path to save figure (if not provided, displays only)
%

% Set defaults
if nargin < 3 || isempty(start_time_sec)
    start_time_sec = 0;
end
if nargin < 4 || isempty(duration_sec)
    % Default: show entire recording
    duration_sec = (EEG.pnts - 1) / EEG.srate;
end

% Convert time to sample indices
start_sample = max(1, round(start_time_sec * EEG.srate) + 1);
end_sample = min(EEG.pnts, round((start_time_sec + duration_sec) * EEG.srate));
sample_indices = start_sample:end_sample;

% Extract data segment
data_segment = EEG.data(:, sample_indices);
time_vec = (sample_indices - 1) / EEG.srate;

% Create figure
fig = figure('Position', [100, 100, 1200, 800]);

% Normalize and stack channels for visualization
num_channels = size(data_segment, 1);
channel_spacing = 5; % Spacing between channels in plot

% Compute per-channel normalization (z-score)
data_normalized = zeros(size(data_segment));
for ch = 1:num_channels
    ch_data = data_segment(ch, :);
    ch_mean = mean(ch_data);
    ch_std = std(ch_data);
    if ch_std > 0
        data_normalized(ch, :) = (ch_data - ch_mean) / ch_std;
    else
        data_normalized(ch, :) = ch_data - ch_mean;
    end
end

% Plot each channel with offset
hold on;
colors = lines(num_channels);
channel_labels = cell(num_channels, 1);

for ch = 1:num_channels
    offset = (num_channels - ch) * channel_spacing;
    plot(time_vec, data_normalized(ch, :) + offset, 'Color', colors(ch, :), 'LineWidth', 0.5);
    
    % Get channel label
    if ~isempty(EEG.chanlocs) && length(EEG.chanlocs) >= ch
        channel_labels{ch} = EEG.chanlocs(ch).labels;
    else
        channel_labels{ch} = sprintf('Ch%d', ch);
    end
end
hold off;

% Set y-axis ticks to channel names
ytick_positions = (0:(num_channels-1)) * channel_spacing;
set(gca, 'YTick', ytick_positions, 'YTickLabel', flipud(channel_labels));

% Labels and title
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('Channels', 'FontSize', 12);
title(sprintf('%s (%.1f-%.1f sec, %d Hz)', title_str, start_time_sec, start_time_sec + duration_sec, EEG.srate), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Grid
grid on;
set(gca, 'GridAlpha', 0.15);

% Add info text
info_str = sprintf('Channels: %d | Samples: %d | Rate: %.1f Hz', ...
    num_channels, length(sample_indices), EEG.srate);
text(0.02, 0.98, info_str, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

% Save if path provided
if nargin >= 5 && ~isempty(save_path)
    % Create directory if needed
    [save_dir, ~, ~] = fileparts(save_path);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % Save figure
    saveas(fig, save_path);
    fprintf('  Saved visualization to: %s\n', save_path);
    close(fig);
else
    % Display and wait for user
    fprintf('  Displaying visualization. Close figure to continue.\n');
end

end


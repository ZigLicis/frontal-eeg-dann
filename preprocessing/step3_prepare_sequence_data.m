function [X_data, Y_data] = step3_prepare_sequence_data(EEG, label, window_length_sec, stride_sec)
% step3_prepare_sequence_data() - Segments continuous data into windows
% of raw time-series data suitable for deep learning.
%
% Usage:
%   >> [X, Y] = step3_prepare_sequence_data(EEG, 'Normal', 5, 1);
%
% Inputs:
%   EEG               - Fully cleaned EEGLAB dataset structure.
%   label             - The session-level label ('Normal' or 'Fatigued').
%   window_length_sec - The duration of each window in seconds.
%   stride_sec        - The stride between window starts in seconds (overlap if < window_length_sec).
%
% Outputs:
%   X_data            - Cell array, where each cell is a [channels x timepoints] matrix.
%   Y_data            - Cell array of corresponding string labels.
%

X_data = {};
Y_data = {};

% Default stride to 1 second if not provided
if nargin < 4 || isempty(stride_sec)
    stride_sec = 1;
end

% --- Calculate window parameters in data points ---
window_length_pnts = floor(window_length_sec * EEG.srate);
% Overlapping windows controlled by stride_sec
step_size_pnts = max(1, floor(stride_sec * EEG.srate));

% --- Start the first window at the beginning of the data ---
current_pos = 1;
window_count = 0;

% --- Loop through the data, sliding the window ---
while (current_pos + window_length_pnts - 1) <= EEG.pnts
    window_count = window_count + 1;
    
    % Define the segment of data for the current window
    window_end = current_pos + window_length_pnts - 1;
    data_segment = EEG.data(:, current_pos:window_end);
    
    %% --- Convert to frequency-domain image (freq × channels × 1) ---
    % Ensure each segment has 1024 points (resample or zero-pad)
    targetLen = 1024;
    data_segment = double(data_segment); % ensure double precision for resample
    if size(data_segment,2) ~= targetLen
        data_segment = resample(data_segment', targetLen, size(data_segment,2))'; % channels × 1024
    end
    % FFT along time
    spec = abs(fft(data_segment, targetLen, 2));    % channels × 1024
    spec = spec(:,1:128);                            % keep up to 128 Hz (assuming 250 Hz fs)
    % Rearrange to freq × channels × 1
    img = permute(spec, [2 1]);                      % 128 × channels
    img = reshape(img, [size(img,1), size(img,2), 1]);
    
    % Store the image and its label
    X_data{window_count, 1} = img;
    Y_data{window_count, 1} = label;
    
    % Move the window start position forward
    current_pos = current_pos + step_size_pnts;
end

end 
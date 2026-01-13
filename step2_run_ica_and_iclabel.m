function EEG_out = step2_run_ica_and_iclabel(EEG_in, visualize, subject_id, manual_review)
% step2_run_ica_and_iclabel() - Runs ICA and uses ICLabel to remove artifacts.
%
% This function runs Independent Component Analysis (ICA) and then uses the
% ICLabel plugin to automatically classify and reject components that are
% not brain-related. Optionally allows manual review of components.
%
% Usage:
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in);
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, true, 'Subject01');
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, true, 'Subject01', true);  % Manual review
%
% Inputs:
%   EEG_in        - Input EEGLAB dataset structure (preprocessed).
%   visualize     - Optional boolean to enable visualization (default: false).
%   subject_id    - Optional subject identifier for plot filenames (default: 'unknown').
%   manual_review - Optional boolean to enable manual IC selection (default: false).
%                   When true, displays IC topomaps, time series, and ICLabel
%                   predictions, then prompts user to select which ICs to reject.
%
% Outputs:
%   EEG_out     - EEG dataset with artifactual ICs removed.
%
% Requires: ICLabel plugin for EEGLAB.

% Set defaults
if nargin < 2
    visualize = false;
end
if nargin < 3 || isempty(subject_id)
    subject_id = 'unknown';
end
if nargin < 4
    manual_review = false;
end

EEG = EEG_in;

% Visualize before ICA
if visualize
    viz_path = sprintf('diagnostics/viz/%s_05_before_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 5. Before ICA', subject_id), [], [], viz_path);
end

% --- 1. Run ICA ---
% Let ICA automatically determine the data rank to handle potential
% rank-deficiency issues after preprocessing, which is more robust than
% manually setting the PCA dimension.
fprintf('Running ICA (infomax)...\n');
try
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
catch ME
    warning('ICA failed: %s. Skipping ICA/ICLabel for this dataset.', ME.message);
    EEG_out = EEG;
    return;
end
EEG.comments = pop_comments(EEG.comments, '', 'Ran extended infomax ICA.', 1);

% --- 2. Run ICLabel ---
% This automatically classifies components into categories like brain, muscle, eye, etc.
fprintf('Running ICLabel for component classification...\n');
if ~exist('pop_iclabel', 'file')
    error('ICLabel plugin not found. Please install it through the EEGLAB extension manager.');
end
% Ensure fields some plugins expect exist
if ~isfield(EEG, 'dipfit')
    EEG.dipfit = [];
end
if ~isfield(EEG, 'icaact')
    EEG.icaact = [];
end
try
    EEG = pop_iclabel(EEG, 'default');
catch ME
    warning('ICLabel failed: %s. Proceeding without IC rejection.', ME.message);
    EEG_out = EEG;
    return;
end
EEG.comments = pop_comments(EEG.comments, '', 'Ran ICLabel for component classification.', 1);

% --- 3. Identify and Remove Artifactual Components ---
% After vEOG cleaning in Step 1, the remaining artifacts are more subtle.
% We use a more sensitive threshold here: flag a component for rejection if
% its 'Brain' probability is lower than its 'Eye' or 'Muscle' probability.

if ~isfield(EEG,'etc') || ~isfield(EEG.etc,'ic_classification') || ~isfield(EEG.etc.ic_classification,'ICLabel')
    EEG_out = EEG;
    return;
end

% Get ICLabel classifications
% Categories: 1=Brain, 2=Muscle, 3=Eye, 4=Heart, 5=Line Noise, 6=Channel Noise, 7=Other
classifications = EEG.etc.ic_classification.ICLabel.classifications;
class_labels = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Ch Noise', 'Other'};

brain_prob = classifications(:,1);
muscle_prob = classifications(:,2);
eye_prob = classifications(:,3);

% Auto-suggested artifacts (brain < eye OR brain < muscle)
auto_artifact_indices = find(brain_prob < eye_prob | brain_prob < muscle_prob);

if manual_review
    % === MANUAL REVIEW MODE ===
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('MANUAL IC REVIEW MODE - Subject: %s\n', subject_id);
    fprintf('%s\n', repmat('=', 1, 70));
    
    num_ics = size(EEG.icaweights, 1);
    
    % --- Display ICLabel classification summary table ---
    fprintf('\n--- ICLabel Classification Summary ---\n');
    fprintf('%-4s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s | %s\n', ...
        'IC', 'Brain', 'Muscle', 'Eye', 'Heart', 'Line', 'ChNoise', 'Other', 'Auto-Reject?');
    fprintf('%s\n', repmat('-', 1, 85));
    
    for ic = 1:num_ics
        is_auto_reject = ismember(ic, auto_artifact_indices);
        reject_str = '';
        if is_auto_reject
            reject_str = '<-- REJECT';
        end
        fprintf('%-4d | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %s\n', ...
            ic, classifications(ic,1)*100, classifications(ic,2)*100, ...
            classifications(ic,3)*100, classifications(ic,4)*100, ...
            classifications(ic,5)*100, classifications(ic,6)*100, ...
            classifications(ic,7)*100, reject_str);
    end
    fprintf('%s\n', repmat('-', 1, 85));
    fprintf('Auto-suggested rejections: [%s]\n', num2str(auto_artifact_indices'));
    
    % --- Open visualization windows ---
    fprintf('\nOpening visualization windows...\n');
    
    % 1. Topographical maps for all components
    fprintf('  - IC Scalp Topomaps (pop_topoplot)\n');
    pop_topoplot(EEG, 0, 1:num_ics, subject_id, [ceil(sqrt(num_ics)) ceil(num_ics/ceil(sqrt(num_ics)))], 0, 'electrodes', 'on');
    
    % 2. ICLabel properties viewer (shows topomap + spectrum + classification)
    fprintf('  - ICLabel Properties Viewer (pop_viewprops)\n');
    try
        pop_viewprops(EEG, 0, 1:min(num_ics, 35), {'freqrange', [1 45]});
    catch
        warning('pop_viewprops not available. Showing individual component properties instead.');
        for ic = 1:min(num_ics, 10)
            pop_prop(EEG, 0, ic, NaN, {'freqrange', [1 45]});
        end
    end
    
    % 3. IC activations time series (raw IC signals)
    fprintf('  - IC Activations Time Series (eegplot)\n');
    if isempty(EEG.icaact)
        EEG.icaact = eeg_getdatact(EEG, 'component', 1:num_ics);
    end
    % Create labels for each IC
    ic_labels = cell(1, num_ics);
    for ic = 1:num_ics
        [~, max_class] = max(classifications(ic,:));
        ic_labels{ic} = sprintf('IC%d (%s %.0f%%)', ic, class_labels{max_class}, classifications(ic, max_class)*100);
    end
    eegplot(EEG.icaact, 'srate', EEG.srate, 'title', sprintf('%s - IC Activations', subject_id), ...
        'eloc_file', ic_labels, 'events', EEG.event, 'winlength', 10);
    
    % --- Prompt user for selection ---
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('REVIEW THE WINDOWS ABOVE\n');
    fprintf('Look for:\n');
    fprintf('  - Eye blinks: Frontal topography, low-frequency bursts in time series\n');
    fprintf('  - Muscle: High-frequency noise, temporal/peripheral topography\n');
    fprintf('  - Heart: Regular periodic signal, dipolar topography\n');
    fprintf('  - Line noise: 50/60 Hz peak in spectrum\n');
    fprintf('%s\n', repmat('=', 1, 70));
    
    fprintf('\nEnter IC numbers to REJECT (comma or space separated)\n');
    fprintf('  Examples: "1,3,5" or "1 3 5"\n');
    fprintf('  Type "none" to reject nothing\n');
    fprintf('  Type "auto" or press Enter to use auto-suggested: [%s]\n', num2str(auto_artifact_indices'));
    fprintf('  Type "all" to review each IC individually\n');
    
    user_input = input('Your selection: ', 's');
    user_input = strtrim(lower(user_input));
    
    if isempty(user_input) || strcmp(user_input, 'auto')
        artifact_indices = auto_artifact_indices;
        fprintf('Using auto-suggested rejections.\n');
    elseif strcmp(user_input, 'none')
        artifact_indices = [];
        fprintf('No components will be rejected.\n');
    elseif strcmp(user_input, 'all')
        % Individual review mode
        artifact_indices = [];
        fprintf('\n--- Individual IC Review ---\n');
        for ic = 1:num_ics
            fprintf('\nIC %d: Brain=%.1f%%, Eye=%.1f%%, Muscle=%.1f%%\n', ...
                ic, brain_prob(ic)*100, eye_prob(ic)*100, muscle_prob(ic)*100);
            
            % Show individual component
            pop_prop(EEG, 0, ic, NaN, {'freqrange', [1 45]});
            
            % Also show just this IC's time series
            fig_ts = figure('Name', sprintf('IC %d Time Series', ic), 'NumberTitle', 'off');
            plot_duration = min(10, EEG.xmax); % 10 seconds or full data
            plot_samples = round(plot_duration * EEG.srate);
            t = (0:plot_samples-1) / EEG.srate;
            plot(t, EEG.icaact(ic, 1:plot_samples), 'b', 'LineWidth', 0.5);
            xlabel('Time (s)'); ylabel('Amplitude');
            title(sprintf('IC %d Time Series (first %.1f s)', ic, plot_duration));
            grid on;
            
            is_auto = ismember(ic, auto_artifact_indices);
            if is_auto
                fprintf('  [AUTO-SUGGESTED FOR REJECTION]\n');
            end
            
            response = input(sprintf('Reject IC %d? (y/n/q to quit review): ', ic), 's');
            response = strtrim(lower(response));
            
            close(fig_ts);  % Close the time series figure
            
            if strcmp(response, 'y')
                artifact_indices = [artifact_indices, ic];
                fprintf('  -> IC %d marked for rejection\n', ic);
            elseif strcmp(response, 'q')
                fprintf('Exiting individual review.\n');
                break;
            else
                fprintf('  -> IC %d kept\n', ic);
            end
        end
    else
        % Parse user input for specific IC numbers
        user_input = strrep(user_input, ',', ' ');
        ic_nums = str2num(user_input); %#ok<ST2NM>
        if isempty(ic_nums)
            warning('Could not parse input. Using auto-suggested rejections.');
            artifact_indices = auto_artifact_indices;
        else
            artifact_indices = ic_nums(ic_nums >= 1 & ic_nums <= num_ics);
            fprintf('Selected for rejection: [%s]\n', num2str(artifact_indices));
        end
    end
    
    % Close any remaining figures from the review
    fprintf('\nClose the visualization windows when ready, then press Enter to continue...\n');
    pause;
    
else
    % === AUTOMATIC MODE (original behavior) ===
    artifact_indices = auto_artifact_indices;
end

% --- Remove selected components ---
if ~isempty(artifact_indices)
    fprintf('Removing %d artifact components: [%s]\n', length(artifact_indices), num2str(artifact_indices(:)'));
    EEG = pop_subcomp(EEG, artifact_indices, 0);
    EEG.comments = pop_comments(EEG.comments, '', sprintf('Removed %d artifact ICs: [%s]', length(artifact_indices), num2str(artifact_indices(:)')), 1);
else
    fprintf('No components removed.\n');
end

% Visualize after ICA
if visualize
    viz_path = sprintf('diagnostics/viz/%s_06_after_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 6. After ICA/ICLabel Artifact Removal', subject_id), [], [], viz_path);
end

EEG_out = EEG;

end 
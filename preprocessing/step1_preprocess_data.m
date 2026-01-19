function EEG_out = step1_preprocess_data(EEG_in, low_cutoff, high_cutoff, downsample_rate, frontal_chans, ref_chans, visualize, subject_id)
    % step1_preprocess_data() - Performs initial preprocessing based on the tutorial.
    %
    % This function applies channel selection, filtering, downsampling, and
    % re-referencing, following the methods from Lin et al. (2014).
    %
    % Usage:
    %   >> EEG_out = step1_preprocess_data(EEG_in, low_cutoff, high_cutoff, downsample_rate, frontal_chans, ref_chans);
    %   >> EEG_out = step1_preprocess_data(EEG_in, low_cutoff, high_cutoff, downsample_rate, frontal_chans, ref_chans, true, 'Subject01');
    %
    % Inputs:
    %   EEG_in          - Input EEGLAB dataset structure.
    %   low_cutoff      - Low frequency cutoff for high-pass filter (Hz).
    %   high_cutoff     - High frequency cutoff for low-pass filter (Hz).
    %   downsample_rate - New sampling rate to downsample to (Hz).
    %   frontal_chans   - Cell array of frontal channel labels to keep.
    %   ref_chans       - Cell array of reference channel labels (e.g., A1, A2).
    %   visualize       - Optional boolean to enable visualization (default: false).
    %   subject_id      - Optional subject identifier for plot filenames (default: 'unknown').
    %
    % Outputs:
    %   EEG_out         - Preprocessed EEGLAB dataset structure.
    
    % Set defaults
    if nargin < 7
        visualize = false;
    end
    if nargin < 8 || isempty(subject_id)
        subject_id = 'unknown';
    end
    
    EEG = EEG_in;
    % Ensure optional fields expected by some EEGLAB functions exist
    if ~isfield(EEG, 'dipfit')
        EEG.dipfit = [];
    end
    if ~isfield(EEG, 'icaact')
        EEG.icaact = [];
    end
    if ~isfield(EEG, 'chaninfo')
        EEG.chaninfo = struct();
    end
    if ~isfield(EEG, 'event')
        EEG.event = [];
    end
    if ~isfield(EEG, 'epoch')
        EEG.epoch = [];
    end
    % Additional commonly-referenced optional fields
    if ~isfield(EEG, 'reject'), EEG.reject = struct(); end
    if ~isfield(EEG, 'stats'), EEG.stats = struct(); end
    if ~isfield(EEG, 'urevent'), EEG.urevent = []; end
    if ~isfield(EEG, 'icaweights'), EEG.icaweights = []; end
    if ~isfield(EEG, 'icasphere'), EEG.icasphere = []; end
    if ~isfield(EEG, 'icawinv'), EEG.icawinv = []; end
    if ~isfield(EEG, 'icachansind'), EEG.icachansind = []; end
    if ~isfield(EEG, 'specdata'), EEG.specdata = []; end
    if ~isfield(EEG, 'icaspec'), EEG.icaspec = []; end
    if ~isfield(EEG, 'history'), EEG.history = ''; end
    if ~isfield(EEG, 'ref'), EEG.ref = ''; end
    if ~isfield(EEG, 'filename'), EEG.filename = ''; end
    if ~isfield(EEG, 'filepath'), EEG.filepath = ''; end
    
    % --- 1. Look up Channel Locations ---
    needs_lookup = false;
    if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
        needs_lookup = true;
    else
        if ~isfield(EEG.chanlocs, 'X')
            needs_lookup = true;
        else
            try
                needs_lookup = all(cellfun(@isempty, {EEG.chanlocs.X}));
            catch
                needs_lookup = true;
            end
        end
    end
    if needs_lookup
        fprintf('Channel locations not found. Looking up standard locations...\n');
        try
            EEG = pop_chanedit(EEG, 'lookup','Standard-10-20-Cap81.ced');
            EEG.comments = pop_comments(EEG.comments, '', 'Looked up standard channel locations (Cap81).', 1);
        catch
            try
                EEG = pop_chanedit(EEG, 'lookup','standard-10-20.elc');
                EEG.comments = pop_comments(EEG.comments, '', 'Looked up standard channel locations (elc).', 1);
            catch
                warning('Could not automatically look up channel locations.');
            end
        end
    end
    
    % --- 2. Select Frontal and (if present) Reference Channels ---
    all_labels = {EEG.chanlocs.labels};
    all_labels_upper = upper(all_labels);
    frontal_upper = upper(frontal_chans);
    ref_upper = upper(ref_chans);
    % Map desired labels to indices (case-insensitive)
    idx_frontal = find(ismember(all_labels_upper, frontal_upper));
    idx_ref = find(ismember(all_labels_upper, ref_upper));
    desired_idx = unique([idx_frontal, idx_ref]);
    if isempty(desired_idx)
        error('None of the desired frontal/reference channels were found in this dataset.');
    end
    fprintf('Selecting %d channels (%d frontal, %d refs present) ...\n', numel(desired_idx), numel(idx_frontal), numel(idx_ref));
    EEG = pop_select(EEG, 'channel', desired_idx);
    EEG.comments = pop_comments(EEG.comments, '', 'Selected frontal/reference channels (available).', 1);
    
    % Visualize raw data (after channel selection)
    if visualize
        viz_path = sprintf('outputs/viz/%s_01_raw_after_chansel.png', subject_id);
        visualize_eeg_timeseries(EEG, sprintf('%s: 1. Raw EEG (After Channel Selection)', subject_id), [], [], viz_path);
    end
    
    % --- 3. Bandpass Filtering ---
    fprintf('Applying bandpass filter (%.1f-%.1f Hz)...\n', low_cutoff, high_cutoff);
    EEG = pop_eegfiltnew(EEG, 'locutoff', low_cutoff, 'hicutoff', high_cutoff);
    EEG.comments = pop_comments(EEG.comments, '', sprintf('Bandpass filtered from %.1f to %.1f Hz.', low_cutoff, high_cutoff), 1);
    
    % Visualize after filtering
    if visualize
        viz_path = sprintf('outputs/viz/%s_02_after_filtering.png', subject_id);
        visualize_eeg_timeseries(EEG, sprintf('%s: 2. After Bandpass Filtering', subject_id), [], [], viz_path);
    end
    
    % --- 4. Downsampling ---
    fprintf('Downsampling data to %d Hz...\n', downsample_rate);
    EEG = pop_resample(EEG, downsample_rate);
    EEG.comments = pop_comments(EEG.comments, '', sprintf('Downsampled to %d Hz.', downsample_rate), 1);
    
    % --- 5. Re-reference ---
    % Use linked mastoids (A1/A2) for re-referencing
    ref_indices = find(ismember(upper({EEG.chanlocs.labels}), ref_upper));
    if length(ref_indices) < 2
        error('Reference channels A1/A2 not found. Both mastoids are required for re-referencing.');
    end
    fprintf('Re-referencing to linked mastoids (A1/A2) and removing them...\n');
    EEG = pop_reref(EEG, ref_indices, 'keepref', 'off');
    EEG.comments = pop_comments(EEG.comments, '', 'Re-referenced to linked mastoids.', 1);
    
    % Visualize after re-referencing
    if visualize
        viz_path = sprintf('outputs/viz/%s_03_after_reref.png', subject_id);
        visualize_eeg_timeseries(EEG, sprintf('%s: 3. After Re-referencing', subject_id), [], [], viz_path);
    end
    
    % --- 6. Regression-Based Blink Removal (using vEOG) ---
    fprintf('Applying regression-based correction to remove blinks...\n');
    eog_ref_chan1 = 'FP1';
    eog_ref_chan2 = 'F7';
    all_chan_labels = {EEG.chanlocs.labels};
    
    try
        eog_idx1 = find(strcmpi(eog_ref_chan1, all_chan_labels));
        eog_idx2 = find(strcmpi(eog_ref_chan2, all_chan_labels));
        if isempty(eog_idx1) || isempty(eog_idx2)
            error('Could not find vEOG channels Fp1 or F7 for blink correction.');
        end
    catch
        error('Could not find vEOG channels Fp1 or F7 for blink correction.');
    end
    
    % --- Perform calculations in double precision for numerical stability ---
    original_data_type = class(EEG.data); % Store original data type (e.g., 'single')
    eeg_data_double = double(EEG.data);
    
    % Create the virtual EOG signal and calculate its variance
    vEOG_signal = eeg_data_double(eog_idx1, :) - eeg_data_double(eog_idx2, :);
    var_vEOG = var(vEOG_signal);
    
    cleaned_data_double = eeg_data_double;
    
    % Loop through all channels to clean them
    for i = 1:EEG.nbchan
        % Don't correct the channels that make up the vEOG signal itself
        if i == eog_idx1 || i == eog_idx2
            continue;
        end
        
        eeg_channel_signal = eeg_data_double(i, :);
        
        % Calculate covariance and then the regression coefficient 'b'
        C = cov(eeg_channel_signal, vEOG_signal);
        if var_vEOG > 0
            b = C(1, 2) / var_vEOG;
        else
            b = 0; % Avoid division by zero
        end
        
        % Subtract the scaled vEOG artifact from the EEG channel
        cleaned_data_double(i, :) = eeg_channel_signal - (b * vEOG_signal);
    end
    
    % Convert data back to its original type before assigning back to EEG object
    EEG.data = cast(cleaned_data_double, original_data_type);
    EEG.comments = pop_comments(EEG.comments, '', 'Applied regression-based blink removal using vEOG (Fp1-F7).', 1);
    
    % Visualize after blink removal
    if visualize
        viz_path = sprintf('outputs/viz/%s_04_after_blink_removal.png', subject_id);
        visualize_eeg_timeseries(EEG, sprintf('%s: 4. After Blink Removal (vEOG regression)', subject_id), [], [], viz_path);
    end
    
    % --- 7. Remove any remaining non-EEG Channels ---
    channels_to_remove = find(cellfun(@isempty, {EEG.chanlocs.X}));
    if ~isempty(channels_to_remove)
        fprintf('Removing %d non-EEG channels without locations.\n', length(channels_to_remove));
        EEG = pop_select(EEG, 'nochannel', channels_to_remove);
        EEG.comments = pop_comments(EEG.comments, '', 'Removed non-EEG channels.', 1);
    end
    
    EEG_out = EEG;
    
    end
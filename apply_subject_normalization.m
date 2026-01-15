function X_normalized = apply_subject_normalization(X_data, subject_ids, subject_stats)
% apply_subject_normalization - Apply per-subject z-score normalization to spectral data
%
% Inputs:
%   X_data       - Cell array of spectral images (freq x channels x 1)
%   subject_ids  - Cell array of subject IDs corresponding to each window
%   subject_stats - Map containing mean/std for each subject
%
% Output:
%   X_normalized - Cell array of normalized spectral images

X_normalized = cell(size(X_data));

for i = 1:length(X_data)
    subj_id = subject_ids{i};
    
    if isKey(subject_stats, subj_id)
        % Get subject's normalization parameters
        stats = subject_stats(subj_id);
        subj_mean = stats.mean;
        subj_std = stats.std;
        
        % Normalize this window
        img = X_data{i}; % freq x channels x 1
        img_flat = reshape(img, [], 1); % (freq*channels) x 1
        img_norm = (img_flat - subj_mean) ./ subj_std;
        img_norm = reshape(img_norm, size(img)); % back to freq x channels x 1
        
        X_normalized{i} = img_norm;
    else
        % No stats available, keep original (shouldn't happen in normal flow)
        X_normalized{i} = X_data{i};
        warning('No normalization stats found for subject %s', subj_id);
    end
end

end 
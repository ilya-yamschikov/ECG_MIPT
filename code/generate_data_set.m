addpath('features/');
addpath('../lib/wfdb/mcode/');

% file: indices in ecg record filenames, class of record
FILES_TO_LOAD = {{[1 1], 'HEALTHY'}, {[2 1], 'HEALTHY'}, {[3 1], 'HEALTHY'}};
% feature: function, type, return values count
FEATURES_TO_USE = {{@feature_RMS, 'NUMERIC', 1}, {@feature_SpectralDensityOnDifferentBand, 'NUMERIC', 4}};
OUTPUT_FILE = '../out/data_set.arff';

% PRINT FILE HEADER
out_file = fopen(OUTPUT_FILE, 'w');
fprintf(out_file, ['%% auto generated file from MATLAB on ' datestr(now) '\n\n']);
fprintf(out_file, '@RELATION ecg_mining\n\n');
for feature_idx = 1:length(FEATURES_TO_USE)
    feature = FEATURES_TO_USE{feature_idx};
    type = feature{2};
    for i = 1:feature{3}
        fprintf(out_file, ['@ATTRIBUTE ' func2str(feature{1}) ' ' feature{2} '\n']); 
    end
end
fprintf(out_file, '@ATTRIBUTE class {HEALTHY, ISHEMIC}\n\n'); 
%

for ecg_idx = 1:length(FILES_TO_LOAD)
    ecg_description = FILES_TO_LOAD{ecg_idx};
    file = ecg_description{1};
    [x, y_low, y_high, Fs] = read_mouse_ecg(file);
    y = normalize_hfq_ecg(y_high);
    
    for feature_idx = 1:length(FEATURES_TO_USE)
        feature = FEATURES_TO_USE{feature_idx};
        feature_function = feature{1};
        fprintf(out_file, '%f,', feature_function(y, Fs));
    end
    fprintf(out_file, '%s\n', ecg_description{2});
end

fclose(out_file);
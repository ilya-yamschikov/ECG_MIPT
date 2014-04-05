function [x, y_low, y_high, Fs] = read_ptbdb_ecg(filename, HighFrequencyLimit)
   raw_ecg = csvread(filename);
   % frequency & time axis
   x = raw_ecg(:, 1);
   y = raw_ecg(:, 2);
   Fs = 1 / (x(2) - x(1));
   % filter
   Wn = HighFrequencyLimit / (Fs/2);
   [b_low, a_low] = butter(2, Wn, 'low');
   [b_high, a_high] = butter(2, Wn, 'high');
   % ECG
   y_low = filter(b_low, a_low ,y);
   y_high = filter(b_high, a_high ,y);
end
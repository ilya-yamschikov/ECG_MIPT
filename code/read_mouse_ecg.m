function [x, y_low, y_high, Fs] = read_mouse_ecg(a)
    [y, Fs] = audioread(['..\data\new_data\' num2str(a(1)) '_' num2str(a(2)) '.wav']);
    y_low = y(:, 2);
    y_high = y(:, 1);
    x = (1:length(y)) / Fs;
end
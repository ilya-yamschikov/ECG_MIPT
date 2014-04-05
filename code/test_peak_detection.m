CUTOFF_FREQUENCY = 100;


filename = '..\data\new_data\1_1.wav';
[y, Fs] = audioread(filename);
y = y(:,2);
x = (1:length(y)) / Fs;

[b, a] = butter(2, CUTOFF_FREQUENCY / (Fs/2));

y_filtered = filter(b, a ,y);
plot(x, y, 'b-',x, y_filtered, 'r-');

figure
[x_fft, y_fft] = standartise_fft(fft(y_filtered), Fs);
plot(x_fft, y_fft, 'g-');
[max_intensity, idx] = max(y_fft);
disp(['characteristic frequency: ' num2str(x_fft(idx)) 'Hz with intensity ' num2str(y_fft(idx))]);

figure
[pks,locs] = findpeaks(y_filtered, 'threshold', 0.0000001);
plot(x, y_filtered, 'b-', x(locs), pks, 'rv');


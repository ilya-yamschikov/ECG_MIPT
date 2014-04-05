[x, y_low, y_high, Fs] = read_mouse_ecg(1, 1);

%plot(x, y_low, 'b-');
fs = resolve_charackteristic_frequency(y_low, Fs);
disp(['Main fq = ' fs]);
 
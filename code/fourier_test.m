dt = 1/50; % discretization
T = 100; % duration (sec)
x = 0:dt:T;

% func
frec = 1; % Hz
y1 = sin(x * 2*pi/frec) + sin(5*x * 2*pi/frec + pi/3);
y2 = [sin(x(1:(floor(length(x)/2))) * 2*pi/frec), sin(5*x((1 + floor(length(x)/2)):end) * 2*pi/frec + pi/3)];

plot(x,y1, 'r');
figure
plot(x,y2, 'g');

% fourier
res1 = fft(y1);
res2 = fft(y2);

figure
[x_freq, y1] = standartise_fft(res1, 1/dt);
plot(x_freq, y1, 'b');
figure
[x_freq, y2] = standartise_fft(res2, 1/dt);
plot(x_freq, y2, 'm');

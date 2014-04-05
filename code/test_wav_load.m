%filename = '..\data\new_data\2_1.wav';
%dt = 1/44100;

%[y, Fs] = audioread(filename);

%plot(y(:,1), 'r-');
%hold on
%plot(y(:,2), 'b-');

%vector_to_work = y(1:end,2);
%plot((1:length(vector_to_work))* dt, vector_to_work, 'b-');

%res = fft(vector_to_work);
%figure
%[x_freq, y1] = standartise_fft(res, 1/dt);
%plot(x_freq, y1, 'b');

%figure
%c = cwt (vector_to_work, 1:1024, 'db4');
%surf(c,'EdgeColor','none')
%colormap(gray)
%view(2)

files_to_load = 8;

for i = 1:files_to_load
    filename = ['..\data\new_data\3_' num2str(i) '.wav'];
    disp(['Read ' filename]);
    audioinfo(filename)
    [y, Fs] = audioread(filename);
    x = (1:length(y(:,1)))/Fs;
    subplot(files_to_load, 1, i)
    plot(x,y(:,1),'b-');
    hold on
    %plot(x,y(:,2),'r-');
end
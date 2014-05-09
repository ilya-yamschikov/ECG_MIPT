files_to_read = {'..\data\txts\2_1.txt', '..\data\txts\2_2.txt', '..\data\txts\2_6.txt'};
FILES_COUNT = length(files_to_read);

for i = 1:FILES_COUNT
    file_name = cell2mat(files_to_read(i));
    fileID = fopen(file_name, 'r');
    if (fileID == -1)
        disp(['file "' file_name '" not found']);
    end
    
    [DATA, SAMPLES_COUNT, FREQUENCY] = read_file(fileID);

    %hold on
    %plot(1:SAMPLES_COUNT, DATA(:,1), 'r');
    %plot(1:SAMPLES_COUNT, DATA(:,2), 'b');
    %hold off

    %FFT_res_low_freq = fft(DATA(:,1));
    %[x, y] = standartise_fft(FFT_res_low_freq, FREQUENCY);
    %figure
    %plot(x, y, 'g');
    %FFT_res_high_freq = fft(DATA(:,2));
    %[x, y] = standartise_fft(FFT_res_high_freq, FREQUENCY);
    %figure
    %plot(x, y, 'm');
    
    subplot(FILES_COUNT, 1, i);
    [x, y] = standartise_fft(fft(DATA(:,1)), FREQUENCY);
    plot(x(1:end/2), y(1:end/2), 'b');
end

figure
plot(DATA(:,1), 'r');
function value = feature_SpectralDensityOnDifferentBand(y, Fs)
    BORDERS = [0 250 500 750 1000];

    % check if y is vector and get a row from it  
    if (isrow(y))
        yRow = y;
    else 
        yRow = y';
        if (~isrow(yRow))
            throw(MException('feature_RMS:yNotVector', ['feature_RMS: y is not a vector: ' mat2str(y)]));
        end
    end
    
    [x, standartised_fft] = standartise_fft(fft(yRow), Fs);
    disp(['Max available frequency: ' num2str(x(end))]);
    fftRow = standartised_fft;
    if (~isrow(fftRow))
        fftRow = fftRow';
    end
    
    value = zeros(1, length(BORDERS) - 1);
    for i = 1:length(BORDERS)-1
       value(i) = calculate_RMS(fftRow(BORDERS(i) < x & x < BORDERS(i+1)));
    end
end
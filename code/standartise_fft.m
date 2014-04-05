function [x, standartised_fft] = standartise_fft(fft_result, freq)
    if (nargin < 2) 
        freq = 2;
    end
    
    N = length(fft_result);
    x = freq / 2 * linspace(0,1,round(N/2));
    standartised_fft = 2/N * abs(fft_result(1:round(N/2)));
end
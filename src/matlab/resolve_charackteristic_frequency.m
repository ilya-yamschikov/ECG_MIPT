function fq = resolve_charackteristic_frequency(y, Fq)
    [x, y_fft] = standartise_fft(fft(y), Fq);
    [max_fq_intensity, idx_max_fq] = max(y_fft);
    [pks,locs] = findpeaks(y_fft);
    top_peaks_idx = locs(pks > 0.1 * max_fq_intensity);
    plot(x, y_fft, 'b-', x(top_peaks_idx), y_fft(top_peaks_idx), 'ro');
end

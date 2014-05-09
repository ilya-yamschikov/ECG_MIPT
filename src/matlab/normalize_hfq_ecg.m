function y_norm = normalize_hfq_ecg(y)
    y_norm = y - mean(y);
    y_norm = y / median(abs(y_norm));
end
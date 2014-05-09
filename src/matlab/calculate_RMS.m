function rms = calculate_RMS(y)
    if (~isrow(y))
        throw(MException('calculate_RMS:yNotRow', ['feature_RMS: y is not a row: ' mat2str(y)]));
    end
    rms = sqrt(y * y' / length(y));
end
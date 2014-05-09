function value = feature_RMS(y, Fs)
    % check if y is vector and get a row from it  
    if (isrow(y))
        yRow = y;
    else 
        yRow = y';
        if (~isrow(yRow))
            throw(MException('feature_RMS:yNotVector', ['feature_RMS: y is not a vector: ' mat2str(y)]));
        end
    end
    
    value = calculate_RMS(yRow);
end
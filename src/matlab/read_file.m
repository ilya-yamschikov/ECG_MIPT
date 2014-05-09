function [DATA, SIZE, FREQUENCY] = read_file(fileID)
    buf = textscan(fileID, '%s %d', 1);
    SIZE = buf{2};
    header = textscan(fileID, '%s',3,'delimiter','\n');
    buf = textscan(fileID, '%s %d', 1);
    FREQUENCY = double(buf{2});
    header = textscan(fileID, '%s',2,'delimiter','\n');

    DATA = double(cell2mat(textscan(fileID, '%d %d', SIZE)));
end
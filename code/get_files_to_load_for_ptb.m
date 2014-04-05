function files = get_files_to_load_for_ptb()
    dir = '..\data\ptb_database_csv\';
    info_filename = [dir 'info.txt'];
    info_file = fopen(info_filename, 'r');
    if (info_file == -1)
       disp(['cannot open ' info_filename]);
    end
   
    info = textscan(info_file, '%s');
    files = cell(length(info{1}),1);
    for i = 1:length(files)
        file_sample = [dir info{1}{i}];
        description_filename = [file_sample '.descr'];
        description_file = fopen(description_filename, 'r');
        class = fgetl(description_file); 
        files{i} = {file_sample, class};
        fclose(description_file);
    end
    
    fclose(info_file);
end
clc
clear
close all

%%
folder_path = '.\rawpic\';
result_path = '.\result\';
jpg_files = dir(fullfile(folder_path, '*.jpg'));   % get all jpg pics
png_files = dir(fullfile(folder_path, '*.png'));   % get all png pics

image_files = [jpg_files; png_files];


for ii = 1:length(image_files)

    name = image_files(ii).name;
    path = strcat(folder_path, name);
    
    img_raw = imread(path);
    
    img_double  = im2double(img_raw);
    
    results_LAFE = LAFE_method_with_patternsearch(img_double);
    results_LAFE = im2uint8(results_LAFE);
    
    LAFE_name = strcat(strcat(result_path),image_files(ii).name);
    imwrite(results_LAFE,LAFE_name);

end

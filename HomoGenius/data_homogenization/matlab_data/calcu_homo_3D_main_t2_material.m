clc;clear;clear all;
pwd;

data_folderpath = "./data/1801-2400";
result_save_folderpath = "./FEM results/1801-2400";

lx = 1.0;
ly = 1.0;
lz = 1.0;
E = 1.0;


% 获取目录下所有的.mat文件
fileList = dir(fullfile(data_folderpath, '*.mat'));
% 遍历文件列表，找到文件名不包含"_"字符的.mat文件

% 提取文件名
fileNames = {fileList.name};

% 提取文件名中的数字并转换为数字
% 假设文件名的格式是固定的，例如 'file1.mat', 'file2.mat' 等
numericValues = cellfun(@(x) sscanf(x, '%d.mat'), fileNames);

% 对文件进行排序
[~, sortedIndex] = sort(numericValues);

% 应用排序索引
sortedFileListStruct = fileList(sortedIndex);

para = 10;
for index = 1:60
   
    
    % parfor的数组储存
    results_nu = cell(para, 1);
    results_CH = cell(para, 1);
    results_dis = cell(para, 1);
    
    parfor i = para*(index-1)+1 : para*index
    %     matname = matnames{i};
    %     loaddata = load(fullfile(matfile_folder,matname));
    %     idx = sprintf('%05d',i);
        uniform_random_numbers = rand; % 生成0到1的均匀分布的随机数
        v = 0.1 + (0.4 - 0.1) * uniform_random_numbers;

        mu = E/2.0/(1.0+v);
        lambda = E*v/(1.0+v)/(1.0-2.0*v);

        file_name = fullfile(data_folderpath,sortedFileListStruct(i).name);
        data = load(file_name).data;
        t1 = clock;
        [Homoed_C, virtual_X, real_X] = homo3D_Alloutput(lx,ly,lz,lambda,mu,data);
        t2 = clock;
        disp(['Current RVE computational cost:',num2str(etime(t2,t1))]);
        fprintf('%s has been computed !!! \n',sortedFileListStruct(i).name);
    
        [~, data_string, ~] = fileparts(sortedFileListStruct(i).name);
    
        data_nu = data * v;
        results_nu{i} = data_nu;
        results_CH{i} = Homoed_C;
        results_dis{i} = real_X;
        close();
    
    end
    
    for i = para*(index-1)+1 : para*index
    
        [~, data_string, ~] = fileparts(sortedFileListStruct(i).name);
        
        data_nu = results_nu{i};
        Homoed_C = results_CH{i};
        real_X = results_dis{i};
    
        nu_name = strcat('nu_',data_string,'.mat');
        save(fullfile(result_save_folderpath, nu_name),'data_nu');
        result_name = strcat('CH_',data_string,'.mat');
        save(fullfile(result_save_folderpath,result_name),'Homoed_C');
        real_displacement_name = strcat('displacement_',data_string,'.mat');
        save(fullfile(result_save_folderpath,real_displacement_name),'real_X');
    end
end
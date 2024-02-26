clc;clear;clear all;
pwd;

data_folderpath = "./data/res64/1-600";
result_save_folderpath = "./FEM results/res64";

lx = 1.0;
ly = 1.0;
lz = 1.0;
E = 1.0;
v = 0.3;
mu = E/2.0/(1.0+v);
lambda = E*v/(1.0+v)/(1.0-2.0*v);
% 获取目录下所有的.mat文件
fileList = dir(fullfile(data_folderpath, '*.mat'));
% 遍历文件列表，找到文件名不包含"_"字符的.mat文件

for i = 1:3
%     matname = matnames{i};
%     loaddata = load(fullfile(matfile_folder,matname));
%     idx = sprintf('%05d',i);
    file_name = fullfile(data_folderpath,fileList(i).name);
    data = load(file_name).data;
%    data = double(data);
    t1 = clock;
    [Homoed_C, virtual_X, real_X] = homo3D_Alloutput(lx,ly,lz,lambda,mu,data);
    t2 = clock;
    disp(['Current RVE computational cost:',num2str(etime(t2,t1))]);
    fprintf('%s has been computed !!! \n',num2str(i));

    [~, data_string, ~] = fileparts(fileList(i).name);

    nu_name = strcat('nu_',data_string,'.mat');
    data_nu = data * v;
    save(fullfile(result_save_folderpath, nu_name),'data_nu');

    result_name = strcat('CH_',data_string,'.mat');
    save(fullfile(result_save_folderpath,result_name),"Homoed_C");
    real_displacement_name = strcat('displacement_',data_string,'.mat');
    save(fullfile(result_save_folderpath,real_displacement_name),"real_X");
    %reshape_real_X = reshape(real_X,[size(data,1), size(data,2), size(data,3), 3, 6]);
    %reshape_real_displacement_name = strcat('reshaped_displacement_',data_string,'.mat');
    %save(fullfile(result_save_folderpath,real_displacement_name),"real_X");
    close();

end


% visual(Homoed_C);
% savefig('test.png');
% saveas(gcf, 'test.png');
% imwrite(test, 'testfile.tiff');
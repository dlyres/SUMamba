function [result] = extract_channels_times(input_channels,input_times)

rootpath = 'D:\Matlab\workspace\SSVEP\BETA\test\';

%在根目录下创建单样本汇合文件夹
mkdir(strcat(rootpath,'SpecialChannelsAndTimes\'))
for i=1:40

    %创建标签索引文件夹
    mkdir(strcat(rootpath,'SpecialChannelsAndTimes\', num2str(i)));

    %拼接文件名
    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    %导入数据
    EEG = load(setname);

    %定义一个空的二维数组,存储单个样本
    sample = zeros(length(input_channels), length(input_times));

    %将相同刺激类型的脑电数据组成单样本
    for j=1:220
        for k=1:length(input_channels)
            sample(k, :) = EEG.simple_situation(input_channels(k), input_times, j);
        end
         %单样本单独存储一个文件
        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, '\SpecialChannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

result = sample;
end


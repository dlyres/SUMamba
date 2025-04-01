%提取所有被试相同刺激类型
save_path = 'D:\Matlab\workspace\SSVEP\JFPM\test\';
setpath = 'D:\Matlab\workspace\SSVEP\JFPM\dataset\';
for j = 1:12

    %创建一个空的三维数组，存储相同刺激类型
    simple_situation = zeros(8, 1114, 15*10);
    start = 1;
    step = 14;
    for i = 1:10

        %拼接文件名
        setname = strcat(setpath, 'S', num2str(i), '.mat');

        %导入数据
        data = load(setname);

        %转换数据成[8, 1114, 15, 12]
        data_ori = permute(data.eeg, [2, 3, 4, 1]);

        %每个被试的相同刺激类型汇总
        simple_situation(:, :, start:start + step) = data_ori(:, :, :, j);
        start = start + step + 1;
    end

    %保存文件
    save_name = strcat(num2str(j),'_simple_situation.mat');
    save([save_path, save_name],'simple_situation');
end

%------------------------------------------------
%所有通道所有时间
%创建标签目录索引
rootpath = 'D:\Matlab\workspace\SSVEP\JFPM\test\';
for i=1:12

    %创建标签索引文件夹
    mkdir(strcat(rootpath, 'AllchannelsAndTimes\', num2str(i)));

    %拼接文件名
    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    %导入数据
    EEG = load(setname);

    %定义一个空的二维数组,存储单个样本
    sample = zeros(8, 1114);

    %将相同刺激类型的脑电数据组成单样本
    for j=1:150
        sample(:, :) = EEG.simple_situation(:, :, j);

        %单样本单独存储一个文件
        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, 'AllchannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

%------------------------------------------------
%特定通道特定时间
%0.15s-4.15s
right_sample = 1063;
left_sample = 39;
input_times = left_sample+1:right_sample;
length(input_times)

%索引引全部8个导联电极
input_channels = [1, 2, 3, 4, 5, 6, 7, 8];
sample = extract_channels_times(input_channels, input_times);
size(sample);

%带通滤波范围6-50Hz（包括6Hz、50Hz）
FIR()

%------------------------------------------------
%数据切分增强
data_enhance()


%------------------------------------------------
%提取特定通道特定时间频域特征
fre_points = 512;
extract_frequence(fre_points);


%提取全部刺激到pycharm文件夹中
stimulus_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
fre_points = 512;
split_stimulus(stimulus_list, fre_points);

%提取八种刺激到pycharm文件夹中
stimulus_list = [1, 2, 3, 4, 7, 8, 11, 12];
fre_points = 512;
split_stimulus(stimulus_list, fre_points);







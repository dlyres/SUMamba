%提取所有被试相同刺激类型
save_path = 'D:\Matlab\workspace\SSVEP\Benchmark\test\';
setpath = 'D:\Matlab\workspace\SSVEP\Benchmark\dataset\';
for j = 1:40

    %创建一个空的三维数组，存储相同刺激类型
    simple_situation = zeros(64, 1500, 6*35);
    start = 1;
    step = 5;
    for i = 1:35

        %拼接文件名
        setname = strcat(setpath, 'S', num2str(i), '.mat');

        %导入数据
        data_ori = load(setname);

        %每个被试的相同刺激类型汇总
        simple_situation(:, :, start:start + step) = data_ori.data(:, :, j, :);
        start = start + step + 1;
    end

    %保存文件
    save_name = strcat(num2str(j),'_simple_situation.mat');
    save([save_path, save_name],'simple_situation');
end

%------------------------------------------------
%所有通道所有时间
%创建标签目录索引
rootpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\';
for i=1:40

    %创建标签索引文件夹
    mkdir(strcat(rootpath, 'AllchannelsAndTimes\', num2str(i)));

    %拼接文件名
    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    %导入数据
    EEG = load(setname);

    %定义一个空的二维数组,存储单个样本
    sample = zeros(64, 1500);

    %将相同刺激类型的脑电数据组成单样本
    for j=1:210
        sample(:, :) = EEG.simple_situation(:, :, j);

        %单样本单独存储一个文件
        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, 'AllchannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

%------------------------------------------------
%特定通道特定时间
%0.5s-5.5s
right_sample = 5.5*1500/6;
left_sample = 0.5*1500/6;
times = left_sample+1:right_sample;
length(times)

%索引后脑顶叶区域和枕叶区域附近的共30个导联电极
channels_1 = 34:42;
channels_2 = 44:64;
channels = [channels_1, channels_2];
sample = extract_channels_times(channels,times);
size(sample);

%带通滤波范围6-50Hz（包括6Hz、50Hz）
FIR()

%------------------------------------------------
%数据切分增强
time_len = 1.5;
data_enhance(time_len);


%------------------------------------------------
%提取特定通道特定时间频域特征
fre_points = 512;
extract_frequence(fre_points);


%提取八种刺激到pycharm文件夹中
num = 8;
fre_points = 512;
stimulus_list = [25, 26, 27, 28, 29, 30, 31, 32];
split_stimulus(stimulus_list, num, fre_points);


%提取十六种刺激到pycharm文件夹中
num = 16;
fre_points = 512;
stimulus_list = [17, 18, 19, 20, 21, 22, 23, 24, 25,  26, 27, 28, 29, 30, 31, 32];
split_stimulus(stimulus_list, num, fre_points);




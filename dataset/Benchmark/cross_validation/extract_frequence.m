function [] = extract_frequence(fre_points)
%EXTRACT_FREQUENCE 此处显示有关此函数的摘要
%   此处显示详细说明

rootpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\';


if fre_points == 375
    savepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance_frequence_188\');
else
    savepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance_frequence_256\');
end
mkdir(savepath);
    
% 遍历文件夹
for i=1:40
        
    filepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance\', num2str(i), '\');
          
    %创建标签索引文件夹
    mkdir(strcat(savepath, num2str(i)));
    
    %遍历文件
        
    for j=1:1680
        setname = strcat(filepath, num2str(j), '_sample.mat');
        EEG = load(setname);
    
        if fre_points == 375
            %生成的频域信息点数
            N = 375;
            sample_size = size(EEG.sample);
            
            %生成三维数组，保存频域信息
            sample_frequence = zeros(2, sample_size(1), (N + 1)/2);
        
            for z=1:sample_size(1)
            
                %快速傅里叶变换
                fft_data = fft(EEG.sample(z, :), N);
            
                %频域的振幅信息
                fft_data_mod = abs(fft_data(1: (N + 1)/2));
            
                %频域的相位信息
                fft_data_angle = angle(fft_data(1: (N + 1)/2));
            
                %截止频率为采样频率的一半，保留变换后的一半频率数据
                    
                sample_frequence(1, z, :) = fft_data_mod;
                sample_frequence(2, z, :) = fft_data_angle;      
            end
            
            %单样本单独存储一个文件
            save_name = strcat(num2str(j), '_sample_fre.mat');
            setpath = strcat(savepath, num2str(i), '\');
            save([setpath, save_name], 'sample_frequence');
        else
            %生成的频域信息点数
            N = 512;
            sample_size = size(EEG.sample);
            
            %生成三维数组，保存频域信息
            sample_frequence = zeros(2, sample_size(1), N/2);
        
            for z=1:sample_size(1)
            
                %快速傅里叶变换
                fft_data = fft(EEG.sample(z, :), N);
            
                %频域的振幅信息
                fft_data_mod = abs(fft_data(1: N/2));
            
                %频域的相位信息
                fft_data_angle = angle(fft_data(1: N/2));
            
                %截止频率为采样频率的一半，保留变换后的一半频率数据
                    
                sample_frequence(1, z, :) = fft_data_mod;
                sample_frequence(2, z, :) = fft_data_angle;      
            end
            
            %单样本单独存储一个文件
            save_name = strcat(num2str(j), '_sample_fre.mat');
            setpath = strcat(savepath, num2str(i), '\');
            save([setpath, save_name], 'sample_frequence');
        end
    end  
end


end


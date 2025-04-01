function [] = data_enhance(time_len)

rootpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\';

%将5s的时间长度按1.5秒的时间窗口，步长为0.5s，进行滑动切分

if time_len == 1.5
    folderName = strcat(rootpath,'SpecialChannelsAndTimes_FIR_DataEnhance\');
    
    %在根目录下创建单样本汇合文件夹
    mkdir(folderName);
    
    for i=1:40
        savepath = strcat(folderName, num2str(i), '\');
    
        mkdir(savepath);
    
        filePath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR\', num2str(i), '\');
        num = 1;
        
        sample = zeros(30, 375);
        for j=1:210
            fileName = strcat(filePath, num2str(j), '_sample.mat');
            EEG = load(fileName);
            starts = 1;
            ends = 375;
            for z=num:num+7
                sample(:, :) = EEG.sample(:, starts:ends);
                save_name = strcat(num2str(z),'_sample.mat');
                save([savepath, save_name],'sample');
                starts = starts + 125;
                ends = ends + 125;   
            end
            num = num + 8;
        end
        
    end
    
    disp(num - 1);
end
end


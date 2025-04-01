function [] = data_enhance()

%将3s的时间长度按1.5秒的时间窗口，步长为0.5s，进行滑动切分

rootpath = 'D:\Matlab\workspace\SSVEP\BETA\test\';

folderName = strcat(rootpath,'SpecialChannelsAndTimes_FIR_DataEnhance\');

%在根目录下创建单样本汇合文件夹
mkdir(folderName);

for i=1:40
    savepath = strcat(folderName, num2str(i), '\');

    mkdir(savepath);

    filePath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR\', num2str(i), '\');
    num = 1;
    
    sample = zeros(30, 375);
    for j=1:220
        fileName = strcat(filePath, num2str(j), '_sample.mat');
        EEG = load(fileName);
        starts = 1;
        ends = 375;
        for z=num:num+3
            sample(:, :) = EEG.sample(:, starts:ends);
            save_name = strcat(num2str(z),'_sample.mat');
            save([savepath, save_name],'sample');
            starts = starts + 125;
            ends = ends + 125;   
        end
        num = num + 4;
    end
    
end

disp(num - 1);

end


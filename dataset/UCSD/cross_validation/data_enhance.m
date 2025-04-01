function [] = data_enhance()

%将4s的时间长度按1.5秒的时间窗口，步长为0.5s，进行滑动切分

rootpath = 'D:\Matlab\workspace\SSVEP\JFPM\test\';

folderName = strcat(rootpath,'SpecialChannelsAndTimes_FIR_DataEnhance\');

%在根目录下创建单样本汇合文件夹
mkdir(folderName);

for i=1:12
    savepath = strcat(folderName, num2str(i), '\');

    mkdir(savepath);

    filePath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR\', num2str(i), '\');
    num = 1;
    
    sample = zeros(8, 384);
    for j=1:150
        fileName = strcat(filePath, num2str(j), '_sample.mat');
        EEG = load(fileName);
        starts = 1;
        ends = 384;
        for z=num:num+5
            sample(:, :) = EEG.sample(:, starts:ends);
            save_name = strcat(num2str(z),'_sample.mat');
            save([savepath, save_name],'sample');
            starts = starts + 128;
            ends = ends + 128;   
        end
        num = num + 6;
    end
    
end

disp(num - 1);

end


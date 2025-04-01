function [] = FIR()
rootpath = 'D:\Matlab\workspace\SSVEP\BETA\test\';
filepath = strcat(rootpath, 'SpecialChannelsAndTimes\');
folderpath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR\');
for i=1:40
    savepath = strcat(folderpath, num2str(i), '\');
    mkdir(savepath);
    for j=1:220
        setpath = strcat(filepath, num2str(i), '\', num2str(j), '_sample.mat');
        EEG = pop_importdata('dataformat','matlab','nbchan',30,'data',setpath,'srate',250,'pnts',0,'xmin',0);
        EEG.setname = strcat(num2str(j), '_sample.mat');
        EEG = pop_eegfiltnew(EEG, 'locutoff',6,'hicutoff',50);
        sample = EEG.data;
        save([savepath, EEG.setname], 'sample'); 
    end
     
end
     
end


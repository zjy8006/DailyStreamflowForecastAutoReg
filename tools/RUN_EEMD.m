clear
close all
clc

addpath('./EEMD');

data_path='../time_series/';% . for curremt path; .. for the parent path

station='Yangxian' %'Yangxian' or 'Zhangjiashan'


switch station
    case 'Yangxian'
        save_path = '../yx_eemd/data/';
        data = xlsread([data_path,'YangXianDailyFlow1997-2014.xlsx'],1,'B2:B6575');%full=6574samples
        
    case 'Zhangjiashan'
        save_path = '../zjs_eemd/data/';
        data = xlsread([data_path,'ZhangJiaShanDailyFlow1997-2014.xlsx'],1,'B2:B6575');%full=6574samples
end

%Decompose the entire set
signals = eemd(data,0.2,100);
[m,n] = size(signals);
columns = {};
for i=1:n
    if i==1
        columns{i}='ORIG';
    else
        columns{i}=['IMF',num2str(i-1)];
    end
end
decompositions = array2table(signals, 'VariableNames', columns);
writetable(decompositions, [save_path,'EEMD_FULL.csv']);

% Decompose the training set
training_len = 5260;
train=data(1:training_len);%train

train_signals = eemd(train,0.2,100);
[m,n] = size(train_signals);
columns = {};
for i=1:n
    if i==1
        columns{i}='ORIG';
    else
        columns{i}=['IMF',num2str(i-1)];
    end
end
train_decompositions = array2table(train_signals, 'VariableNames', columns);
writetable(train_decompositions, [save_path,'EEMD_TRAIN.csv']);
save(['../results_analyze/results/',station,'-eemd.mat']);


% Decompose the appended set
for i=1:1314%1:1314
    test_num=i;
    appended_signals = eemd(data(1:(training_len+test_num)),0.2,100);%dev2-test
    [m,n] = size(appended_signals);
    columns = {};
    for i=1:n
        if i==1
            columns{i}='ORIG';
        else
            columns{i}=['IMF',num2str(i-1)];
        end
    end
    appended_decompositions = array2table(appended_signals, 'VariableNames', columns);
    a2=[save_path,'eemd-test/eemd_appended_test'];
    b2=num2str(training_len+test_num);
    c2='.csv';
    abc2=[a2,b2,c2];
    writetable(appended_decompositions, abc2);
end

% [m,n] = size(signals);
% t=1:m;
% t=t';
% raw = signals(:,1);
% for i=2:n
%     if i==n
%         eval(['R',num2str(i-1),'=','signals(:,i)',';']);
%     else
%         eval(['IMF',num2str(i-1),'=','signals(:,i)',';']);
%     end
% end


    
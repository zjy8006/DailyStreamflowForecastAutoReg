clear
close all
clc

data_path='../time_series/';

station='Yangxian' %'Yangxian' or 'Zhangjiashan'


switch station
    case 'Yangxian'
        save_path = '../yx_wd/data/';
        data = xlsread([data_path,'YangXianDailyFlow1997-2014.xlsx'],1,'B2:B6575');%full=6574samples
        
    case 'Zhangjiashan'
        save_path = '../zjs_wd/data/';
        data = xlsread([data_path,'ZhangJiaShanDailyFlow1997-2014.xlsx'],1,'B2:B6575');%full=6574samples
end

%% wavelet decomposition======start
%%%%%%%% set the hyperparameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% wavelet of haar, db2, bior 3.3, db5, coif3, 
% db10, db15, db20, db25, db30, db35, db40 and db45
mother_wavelet = 'db10';% set mother wavelet
lev = 3; %same decomposition level (1, 2, and 3) as VMD
columns = {};
for i=1:lev+2%±íÍ·
    if i==1
        columns{i}='ORIG';
    elseif i==lev+2
        columns{i}=['A',num2str(lev)];
    else
        columns{i}=['D',num2str(i-1)];
    end  
end
if exist([save_path,mother_wavelet,'-',num2str(lev)],'dir')==0
   mkdir([save_path,mother_wavelet,'-',num2str(lev)]);
end
if exist([save_path,mother_wavelet,'-',num2str(lev),'/wd-test/'],'dir')==0
   mkdir([save_path,mother_wavelet,'-',num2str(lev),'/wd-test/']);
end

%%%%%%%%%% Decompose the entire set
len=length(data);%the length of data
%%% Performe decomposition of the data set
[C,L]=wavedec(data,lev,mother_wavelet);
%%% Extract approximation and detail coefficients
%%% Extract the approximation coefficients from C
cA=appcoef(C,L,mother_wavelet,lev);
%%% Extract the detail coefficients from C
cD = detcoef(C,L,linspace(1,lev,lev));
%%% Reconstruct the level approximation and level details
A=wrcoef('a',C,L,mother_wavelet,lev); %the approximation
for i=1:lev
    eval(['D',num2str(i),'=','wrcoef(''d'',C,L,mother_wavelet,i)',';']); %the details
end
%%% combine the details, appromaximation and original data into a single parameter
signals=zeros(len,lev+2);
signals(:,lev+2)=A;
signals(:,1)=data;
for i=2:lev+1
    eval(['signals(:,i)','=','D',num2str(i-1),';']);
end
%%% save the decomposition results and the original data
decompositions = array2table(signals, 'VariableNames', columns);
writetable(decompositions, [save_path,mother_wavelet,'-',num2str(lev),'/WD_FULL.csv']);


%%%%%%%%%%%%% Decompose the training set
training_len = 5260;
train=data(1:training_len);%train
len=length(train);%the length of data
%%% Performe decomposition of the data set
[C,L]=wavedec(train,lev,mother_wavelet);
%%% Extract approximation and detail coefficients
%%% Extract the approximation coefficients from C
cA=appcoef(C,L,mother_wavelet,lev);
%%% Extract the detail coefficients from C
cD = detcoef(C,L,linspace(1,lev,lev));
%%% Reconstruct the level approximation and level details
A=wrcoef('a',C,L,mother_wavelet,lev); %the approximation
for i=1:lev
    eval(['D',num2str(i),'=','wrcoef(''d'',C,L,mother_wavelet,i)',';']); %the details
end
%%% combine the details, appromaximation and original data into a single parameter
train_signals=zeros(len,lev+2);
train_signals(:,lev+2)=A;
train_signals(:,1)=train;
for i=2:lev+1
    eval(['train_signals(:,i)','=','D',num2str(i-1),';']);
end
% save the decomposition results and the original data
train_decompositions = array2table(train_signals, 'VariableNames', columns);
writetable(train_decompositions, [save_path,mother_wavelet,'-',num2str(lev),'/WD_TRAIN.csv']);
save(['../results_analyze/results/',station,'-',mother_wavelet,'-',num2str(lev),'.mat']);

%%%%%%%%%%% plot the decomposition results and original data
% f1=figure
% set(f1,'position',[1500 1500 900 900]);
% %%% tight_subplot(rowas,columns,[v-space,h-space],[bottom,top],[left,right])
% ha = tight_subplot(lev+2,1,[.05 .08],[.05 .04],[.05 .02])
% for i=1:size(signals,2)
%     
%     if i==1
%         axes(ha(i));
%         plot(data,'r');
%         title('Original set');
%     elseif i==2
%         axes(ha(i));
%         plot(A,'b');
%         title(['Approximation A',num2str(lev)]);
%     else 
%         axes(ha(i));
%         plot(signals(:,i-2),'g');
%         title(['Detail D',num2str(i-2)]);
%     end
% end


%%%%%%%%%%%%% decompose the validation set one by one %%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1314
    %the test set
    test_num=i;
    append=data(1:(training_len+test_num));
    %performe decomposition
    [C,L]=wavedec(append,lev,mother_wavelet);
    % Extract approximation and detail coefficients
    % Extract the level 3 approximation coefficients from C
    cA=appcoef(C,L,mother_wavelet,lev);
    % Extract the level 3,2,and 1 detail coefficients from C
    cD = detcoef(C,L,linspace(1,lev,lev));
    % Reconstruct the level 3 approximation and level 1,2,3  details
    A=wrcoef('a',C,L,mother_wavelet,lev);
    for j=1:lev
        eval(['D',num2str(j),'=','wrcoef(''d'',C,L,mother_wavelet,j)',';']); %the details
    end
    %combine the details, approximation and orig into one variable
    appended_signals=zeros(length(append),lev+2);
    appended_signals(:,lev+2)=A;
    appended_signals(:,1)=append;
    for j=2:lev+1
        eval(['appended_signals(:,j)','=','D',num2str(j-1),';']);
    end
    % save the decomposition results and the original data
    appended_decompositions = array2table(appended_signals, 'VariableNames', columns);
    a2=[save_path,mother_wavelet,'-',num2str(lev),'/wd-test/wd_appended_test'];
    b2=num2str(training_len+test_num);
    c2='.csv';
    abc2=[a2,b2,c2];
    writetable(appended_decompositions, abc2);
end


clear;
close all;
clc;

addpath('./VMD')

data_path='../time_series/';

station='Yangxian' %'Yangxian' or 'Zhangjiashan'


switch station
    case 'Yangxian'
        save_path = '../yx_vmd/data/';
        data = xlsread([data_path,'YangXianDailyFlow1997-2014.xlsx'],1,'B2:B6575');%full=6574samples
        
    case 'Zhangjiashan'
        save_path = '../zjs_vmd/data/';
        data = xlsread([data_path,'ZhangJiaShanDailyFlow1997-2014.xlsx'],1,'B2:B6575');%full=6574samples
end

% % some sample parameters for VMD             VMD 的一些参数设置
alpha = 2000;       % moderate bandwidth constraint        宽带限制
tau = 0;            % noise-tolerance (no strict fidelity enforcement)   噪声容忍度
K = 8;              % number of modes, i.e., decomposition level 分解个数
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly    参数初始化
tol = 1e-9;         % the convergence tolerance   收敛允许误差

switch station
    case 'Yangxian'
        K=9;
        
    case 'Zhangjiashan'
        K=11;
end

columns = {};
for i=1:(K+1)
    if i==1
        columns{i}='ORIG';
    else
        columns{i}=['IMF',num2str(i-1)];
    end
end

% Decompose the entire set
f = data;%full
orig=f;
% Time Domain 0 to T     时域  0到T
T = length(f);
fs = 1/T;
f=f';
t = (1:T)/T;
freqs = t-0.5-1/T;
%%--------------- Run actual VMD code   运行VMD 指令
[imf, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);%u为13*10958的矩阵
figure
plot(omega); 
imf = imf';

signals=[orig,imf];
decompositions = array2table(signals, 'VariableNames', columns);
file_name=['VMD_FULL.csv'];
writetable(decompositions, [save_path,file_name]);

% DEcompose the training set
training_len = 5260;
f = data(1:training_len);%train
orig=f;
% Time Domain 0 to T     时域  0到T
T = length(f);
fs = 1/T;

f=f';
t = (1:T)/T;
freqs = t-0.5-1/T;
%%--------------- Run actual VMD code   运行VMD 指令
[imf, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);%u为13*10958的矩阵
figure
plot(omega); 
imf = imf';

train_signals=[orig,imf];
train_decompositions = array2table(train_signals, 'VariableNames', columns);
% file_name=['VMD_TRAIN_K',num2str(K),'_a',num2str(alpha),'.csv'];
file_name=['VMD_TRAIN.csv'];
writetable(train_decompositions, [save_path,file_name]);
save(['../results_analyze/results/',station,'-vmd.mat']);


figure
subplot(211);
plot(t,f,'b');
set(gca,'FontSize',8,'XLim',[0 t(end)]);
title('Original signal');
xlabel('Number of Time(day)');
ylabel('Daily inflow(m3)');
subplot(212);
% [Yf, f] = FFTAnalysis(x, Ts);
plot(freqs,abs(fft(f)),'b');
title('The spectrum of the original signal');
xlabel('f/Hz');
ylabel('|Y(f)|');

for k1 = 0:4:K-1
    figure
    for k2 = 1:min(4,K-k1)
        subplot(4,2,2*k2-1);
        plot(t,imf(:,k1+k2),'b');
        set(gca,'FontSize',8,'XLim',[0 t(end)]);
        title(sprintf('IMF%d', k1+k2));
        xlabel('Time/s');
        ylabel(sprintf('IMF%d', k1+k2));
        
        subplot(4,2,2*k2);
%         [yf, f] = FFTAnalysis(imf(k1+k2,:), fs);        
        plot(freqs, abs(fft(imf(:,k1+k2))),'b');
        title(sprintf('The spectrum of IMF%d', k1+k2));
        xlabel('f/Hz');
        ylabel(sprintf('|IMF%d(f)|',k1+k2));
    end
end

figure
subplot(4,2,2*k2-1);
plot(t,imf(:,k1+k2),'b');
set(gca,'FontSize',8,'XLim',[0 t(end)]);
title(sprintf('IMF%d', k1+k2));
xlabel('Time/s');
ylabel(sprintf('IMF%d', k1+k2));
        
subplot(4,2,2*k2)
%[yf, f] = FFTAnalysis(imf(k1+k2,:), fs);        
plot(freqs, abs(fft(imf(:,k1+k2))),'b');
title(sprintf('The spectrum of IMF%d', k1+k2));
xlabel('f/Hz');
ylabel(sprintf('|IMF%d(f)|',k1+k2));
savefig([station,'_vmd_k',num2str(K),'_a',num2str(alpha),'.fig']);

% Decompose the appended set
for i=1:1314  %1:1314
    test_num=i;
    f = data(1:(training_len+test_num));
    orig=f;
    f=f';
    % run vmd
    [imf, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
    imf = imf';
    %save vmd
    appended_signals=[orig,imf];
    appended_decompositions = array2table(appended_signals, 'VariableNames', columns);
    a2=[save_path,'vmd-test/vmd_appended_test'];
    b2=num2str(training_len+test_num);
    c2='.csv';
    abc2=[a2,b2,c2];
    writetable(appended_decompositions, abc2);
end

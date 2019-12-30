%--------------- Preparation   预备
% clear all;
close all;
clc;
% Time Domain 0 to T     时域  0到T
T = 1000;
fs = 1/T;
% f=xlsread('光伏电站出力（分辨率15min）','光伏出力15min','E2:E35041');
f = xlsread('E:/MATLAB/WorkSpace/HXX/data-1988-2017.xlsx',1,'B2:B10959');%30a
% f = xlsread('E:\MATLAB\WorkSpace\data-1998-2017.xlsx',1,'B2:B7306');%20a
f=f';
t=1:length(f);
freqs = 2*pi*(t-0.5-1/T)/(fs);

% some sample parameters for VMD             VMD 的一些参数设置
alpha = 2000;        % moderate bandwidth constraint        宽带限制
tau = 0;            % noise-tolerance (no strict fidelity enforcement)   噪声容忍度
K = 10 ;              % 4 modes    
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly    参数初始化
tol = 1e-9;
 
%--------------- Run actual VMD code   运行VMD 指令
[imf, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);%u为13*10958的矩阵
figure
plot(omega); 
imf = imf';
figure
subplot(211)
plot(t,f,'b')
set(gca,'FontSize',8,'XLim',[0 t(end)]);
title('原始信号')
xlabel('Number of Time(day)')
ylabel('Daily inflow(m3)');
subplot(212)
% [Yf, f] = FFTAnalysis(x, Ts);
plot(freqs,abs(fft(f)),'b')
title('原始信号的频谱')
xlabel('f/Hz')
ylabel('|Y(f)|');


% figure
% subplot(size(imf,1)+1,2,1);
% plot(t,f,'b');grid on;
% title('VMD分解');
% subplot(size(imf,1)+1,2,2);
% plot(freqs,abs(fft(f)),'b');grid on;
% title('对应频谱');
% for i = 2:size(imf,1)+1
%     subplot(size(imf,1)+1,2,i*2-1);
%     plot(t,imf(i-1,:),'b');grid on;
%     subplot(size(imf,1)+1,2,i*2);
%     plot(freqs,abs(fft(imf(i-1,:))),'b');grid on;
% end 

for k1 = 0:4:K-1
    figure
    for k2 = 1:min(4,K-k1)
        subplot(4,2,2*k2-1)
        plot(t,imf(:,k1+k2),'b')
        set(gca,'FontSize',8,'XLim',[0 t(end)]);
        title(sprintf('第%d个IMF', k1+k2))
        xlabel('Time/s')
        ylabel(sprintf('IMF%d', k1+k2));
        
        subplot(4,2,2*k2)
%         [yf, f] = FFTAnalysis(imf(k1+k2,:), fs);        
        plot(freqs, abs(fft(imf(:,k1+k2))),'b')
        title(sprintf('第%d个IMF的频谱', k1+k2))
        xlabel('f/Hz')
        ylabel(sprintf('|第%d个IMF(f)|',k1+k2));
    end
end

% GGG = imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5);
% GGG = imf(:,1);
% GGG = imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6);
% % GGG = imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7)+imf(:,8);
% % GGG = imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7);
% xlswrite('E:/data-1988-2017.xlsx',GGG,'vmd-重构','B2') 
% figure
% subplot(321) %subplot(n,m,s)n:图像行数，m:图形列数，s:第几个图形。
% % plot(t,imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5),'b')
% plot(t,imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6),'b')
% % plot(t,imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7)+imf(:,8))
% % plot(t,imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7))
% set(gca,'FontSize',8,'XLim',[0 t(end)]);
% title('低频信号')
% % xlabel('Number of Time(day)')
% % ylabel('C1+C2+C3+C4+C5');
% % ylabel('低频信号');
% % ylabel('C1+C2+C3+C4+C5+C6');
% % ylabel('C1+C2+C3+C4+C5+C6+C7+C8');
% % ylabel('C1+C2+C3+C4+C5+C6+C7');
% 
% subplot(322)
% % [Yf1, ff1] = FFTAnalysis(imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6), Ts);
% % [Yf1, ff1] = FFTAnalysis(imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7)+imf(:,8), Ts);
% % [Yf1, ff1] = FFTAnalysis(imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7), Ts);
% % plot(freqs,abs(fft(imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5))),'b')
% plot(freqs,abs(fft(imf(:,1)+imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6))),'b')
% % plot(freqs,abs(fft(imf(:,1))),'b')
% % xlabel('f/Hz')
% ylabel(sprintf('|Yf1|'));
% % FF1 = transpose(ff1);
% % YFF1 = transpose(Yf1);
% % xlswrite('E:/data-1988-2017.xlsx',YFF1,'Sheet5','G2') 
% % xlswrite('E:/data-1988-2017.xlsx',FF1,'Sheet5','H2')
%         
% % GG = transpose(imf(:,6)+imf(:,7)+imf(:,8));
% % GG = transpose(imf(:,7)+imf(:,8)+imf(:,9));
% % GG = imf(:,6)+imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)
% % GG = imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12)
% % GG = imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10);
% % GG = transpose(imf(:,9)+imf(:,10));
% GG = imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12);
% % GG = imf(:,9)+imf(:,10)+imf(:,11);
% % GG = imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11);
% xlswrite('E:/data-1988-2017.xlsx',GG,'vmd-重构','C2') 
% subplot(323)
% % plot(t,imf(:,7)+imf(:,8)+imf(:,9))
% % plot(t,imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12),'b')
% % plot(t,imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10))
% plot(t,imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12),'b')
% % plot(t,imf(:,9)+imf(:,10)+imf(:,11))
% % plot(t,imf(:,9)+imf(:,10))
% % plot(t,imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11))
% % title('C6+C7+C8')
% set(gca,'FontSize',8,'XLim',[0 t(end)]);
% title('中频信号')
% % xlabel('Number of Time(day)')
% % ylabel('C6+C7+C8');
% % ylabel('C7+C8+C9');
% % ylabel('中频信号');
% % ylabel('C7+C8+C9+C10');
% % ylabel('C9+C10');
% % ylabel('C7+C8+C9+C10+C11');
% % ylabel('C9+C10+C11');
% % ylabel('C8+C9+C10+C11');
% 
% subplot(324)
% % [Yf2, ff2] = FFTAnalysis(imf(:,6)+imf(:,7)+imf(:,8), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,7)+imf(:,8)+imf(:,9), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,6)+imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,9)+imf(:,10), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,9)+imf(:,10)+imf(:,11), Ts);
% % [Yf2, ff2] = FFTAnalysis(imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11), Ts);
% % plot(freqs,abs(fft(imf(:,2)+imf(:,3)+imf(:,4)+imf(:,5)+imf(:,6)+imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12))),'b')
% plot(freqs,abs(fft(imf(:,7)+imf(:,8)+imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12))),'b')
% % xlabel('f/Hz')
% ylabel(sprintf('|Yf2|'));
% % FF2 = transpose(ff2);
% % YFF2 = transpose(Yf2);
% % xlswrite('E:/data-1988-2017.xlsx',YFF2,'Sheet5','J2') 
% % xlswrite('E:/data-1988-2017.xlsx',FF2,'Sheet5','K2')
% 
% % G = transpose(imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15));
% % G = transpose(imf(:,10)+imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15));
% % G = transpose(imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14));
% % G = transpose(imf(:,11)+imf(:,12));
% % G = transpose(imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15));
% G = imf(:,13)+imf(:,14)+imf(:,15)+imf(:,16);
% xlswrite('E:/data-1988-2017.xlsx',G,'vmd-重构','D2') 
% subplot(325)
% % plot(t,imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15))
% % plot(t,imf(:,10)+imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15))
% % plot(t,imf(:,12)+imf(:,13)+imf(:,14))
% plot(t,imf(:,13)+imf(:,14)+imf(:,15)+imf(:,16),'b')
% % plot(t,imf(:,11)+imf(:,12))
% % plot(t,imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15))
% % title('C9+C10+C11+C12+C13+C14+C15')
% set(gca,'FontSize',8,'XLim',[0 t(end)]);
% title('高频信号')
% % xlabel('Number of Time(day)')
% % ylabel('C9+C10+C11+C12+C13+C14+R15');
% % ylabel('C10+C11+C12+C13+C14+R15');
% % ylabel('C11+C12+C13+C14+R15');
% xlabel('Number of Time(day)')
% % ylabel('高频信号');
% % ylabel('C12+C13+C14+R15');
% % ylabel('C12+C13+C14');
% 
% subplot(326)
% % [Yf3, ff3] = FFTAnalysis(imf(:,9)+imf(:,10)+imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15), Ts);
% % [Yf3, ff3] = FFTAnalysis(imf(:,10)+imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15), Ts);
% % [Yf3, ff3] = FFTAnalysis(imf(:,11)+imf(:,12)+imf(:,13)+imf(:,14), Ts);
% % [Yf3, ff3] = FFTAnalysis(imf(:,11)+imf(:,12), Ts);
% % [Yf3, ff3] = FFTAnalysis(imf(:,12)+imf(:,13)+imf(:,14)+imf(:,15), Ts);
% plot(freqs,abs(fft(imf(:,13)+imf(:,14)+imf(:,15)+imf(:,16))),'b');
% % xlabel('f/Hz')
% ylabel(sprintf('|Yf3|'));



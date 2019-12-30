close all;
clear
clc

stations={"Zhangjiashan", "Yangxian"};
% decomposers={"db2-1","db2-2","db2-3","db5-1","db5-2","db5-3",...
%     "db10-1","db10-2","db10-3","db15-1","db15-2","db15-3",...
%     "db20-1","db20-2","db20-3","db25-1","db25-2","db25-3",...
%     "db30-1","db30-2","db30-3","db35-1","db35-2","db35-3",...
%     "db40-1","db40-2","db40-3","db45-1","db45-2","db45-3",...
%     "bior 3.3-1","bior 3.3-2","bior 3.3-3","coif3-1","coif3-2","coif3-3",...
%     "haar-1","haar-2","haar-3","vmd","eemd"};


decomposers={"db10-3","vmd","eemd"};

for i =1:length(stations)
    station=string(stations(i))
    for j =1:length(decomposers)
        decomposer=string(decomposers(j))
        mat_file=strcat("../results_analyze/results/",station,"-",decomposer,".mat")
        load(mat_file);
        [m,n]=size(train_signals);
        columns = {};
        for i=1:n+2
            if decomposer=="eemd" || decomposer=="vmd"
                if i==1
                    columns{i}='ORIG';
                else
                    columns{i}=['IMF',num2str(i-1)];
                end
            else
                if i==1
                    columns{i}='ORIG';
                elseif i==n
                    columns{i}=['A',num2str(n-2)];
                else
                    columns{i}=['D',num2str(i-1)];
                end
            end
            if i==n+1
                columns{i}='UP';
            elseif i==n+2
                columns{i}='LOW';
            end
        end
        NumLags=20;
        pacfs=zeros(NumLags+1,n);
        up_bounds=zeros(NumLags+1,1);
        lo_bounds=zeros(NumLags+1,1);
        for i=1:n
            [pacf,lags,bounds] = parcorr(train_signals(:,i),'NumLags',NumLags);
            pacfs(:,i)=pacf;
            if i==1
                up_bounds(:,1)=bounds(1);
                lo_bounds(:,1)=bounds(2);
            end
        end
        PACF_DATA=[pacfs,up_bounds,lo_bounds];
        PACF_TABLE = array2table(PACF_DATA, 'VariableNames', columns);
        
        if station=="Zhangjiashan"
            if decomposer=="eemd" || decomposer=="vmd"
                writetable(PACF_TABLE, strcat("../zjs_",decomposer,"/data/PACF.csv"));
            else
                writetable(PACF_TABLE, strcat("../zjs_wd/data/",decomposer,"/PACF.csv"));
            end
            
        elseif station=="Yangxian"
            if decomposer=="eemd" || decomposer=="vmd"
                writetable(PACF_TABLE, strcat("../yx_",decomposer,"/data/PACF.csv"));
            else
                writetable(PACF_TABLE, strcat("../yx_wd/data/",decomposer,"/PACF.csv"));
            end
        end
        
        
    end
end

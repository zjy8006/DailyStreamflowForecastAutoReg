import pandas as pd
import numpy as np

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Current Path: {}'.format(root_path))

def gen_samples(data_path,save_path,column,lag,test_len,leading_time):
    """ 
    Generate hindcast samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -path: The source data file path for generate the learning samples.
        -column: The columns name for read the source data by pandas.
        -lag: The lags for autoregression.
        -test_len: The length of Test set.
    """
    save_path = save_path+str(leading_time)+"_ahead/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if '.xlsx' in data_path:
        dataframe = pd.read_excel(data_path)[column]
    elif '.csv' in data_path:
        dataframe = pd.read_csv(data_path)[column]
    # Drop NaN
    dataframe.dropna()
    # convert pandas dataframe to numpy array
    nparr = np.array(dataframe)
    # Create an empty pandas Dataframe
    full_data_set = pd.DataFrame()
    # Generate input series based on lags and add these series to full dataset
    for i in range(lag):
        x = pd.DataFrame(nparr[i:dataframe.shape[0] - (lag - i)],columns=['X' + str(i + 1)])
        x=x.reset_index(drop=True)
        full_data_set = pd.concat([full_data_set, x], axis=1,sort=False)
    # Generate label data
    label = pd.DataFrame(nparr[lag+leading_time-1:], columns=['Y'])
    label = label.reset_index(drop=True)
    # Add labled data to full_data_set
    full_data_set = full_data_set[:full_data_set.shape[0]-(leading_time-1)]
    full_data_set = full_data_set.reset_index(drop=True)
    full_data_set = pd.concat([full_data_set, label], axis=1,sort=False)
    # Get the length of this series
    series_len = full_data_set.shape[0]
    # Get the training and developing and testing set
    train_df = full_data_set[0:(series_len - test_len - test_len)]
    dev_df = full_data_set[(series_len - test_len - test_len):(series_len - test_len)]
    test_df = full_data_set[(series_len - test_len):series_len]
    assert (train_df['X1'].size + dev_df['X1'].size +test_df['X1'].size) == series_len
    # Get the max and min value of each series
    series_max = train_df.max(axis=0)
    series_min = train_df.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_df = 2 * (train_df - series_min) / (series_max - series_min) - 1
    dev_df = 2 * (dev_df - series_min) / (series_max - series_min) - 1
    test_df = 2 * (test_df - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max_ = pd.DataFrame(series_max, columns=['series_max'])
    series_min_ = pd.DataFrame(series_min, columns=['series_min'])
    # Merge max serie and min serie
    normalize_indicators = pd.concat([series_max_, series_min_], axis=1,sort=False)
    # Storage the normalied indicators to local disk
    normalize_indicators.to_csv(save_path + 'norm_id.csv')
    # print data set length
    print(100*'-')
    print('series length:{}'.format(series_len))
    print('train data set length:{}'.format(train_df.shape[0]))
    print('development data set length:{}'.format(dev_df.shape[0]))
    print('test set length length:{}'.format(test_df.shape[0]))
    assert test_len==dev_df.shape[0]
    assert test_len==test_df.shape[0]
    train_df.to_csv(save_path + 'minmax_unsample_train.csv', index=None)
    dev_df.to_csv(save_path + 'minmax_unsample_dev.csv', index=None)
    test_df.to_csv(save_path + 'minmax_unsample_test.csv', index=None)




def gen_multi_models_hindcast(path,decomposer,lev,test_len,lags_dict,leading_time,wavelet=None,):
    """ 
    Generate multi-models hindcast samples for autoregression problem. 
    Args:
        -path: The source data file path for generate the learning samples.
        -decomposer: The decomposition algorithm.
        -lev: The decomposition level.
        -test_len: The length of testing samples.
        -lags_dict: The lags dict for subsignals.
        -wavelet: The monther wavelet for Wavelet Transform (WD).
    """
    columns=[]
    lags=[]
    if decomposer=='wd':
        data_path = path+wavelet+'-'+str(lev)+'/'
        lags_dict_=lags_dict[wavelet+'-'+str(lev)]
        for i in range(1,lev+2):
            if i==lev+1:
                lags.append(lags_dict_['A'+str(i-1)])
                columns.append('A'+str(i-1))
            else:
                lags.append(lags_dict_['D'+str(i)])
                columns.append('D'+str(i))
    else:
        data_path = path
        lags_dict_=lags_dict[decomposer]
        for i in range(1,lev+1):
            lags.append(lags_dict_['IMF'+str(i)])
            columns.append('IMF'+str(i))
    print('columns:{}'.format(columns))
    print('lags:{}'.format(lags))

    if decomposer=='wd':
        assert lev+1 == len(columns)
    else:
        assert lev==len(columns)

    save_path = data_path+"multi_models_"+str(leading_time)+"_ahead_hindcast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    data = pd.read_csv(data_path+decomposer.upper()+'_FULL.csv')
    # Drop NaN
    data.dropna()
    for k in range(1,len(columns)+1):
        lag=lags[k-1]
        dataframe = data[columns[k-1]]
        nparr = np.array(dataframe)
        # Create an empty pandas Dataframe
        full_data_set = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lag):
            x = pd.DataFrame(nparr[i:dataframe.shape[0] - (lag - i)],columns=['X' + str(i + 1)])
            x=x.reset_index(drop=True)
            full_data_set = pd.concat([full_data_set, x], axis=1,sort=False)
        # Generate label data
        label = pd.DataFrame(nparr[lag+leading_time-1:], columns=['Y'])['Y']
        label = label.reset_index(drop=True)
        # Add labled data to full_data_set
        full_data_set = full_data_set[:full_data_set.shape[0]-(leading_time-1)]
        full_data_set = full_data_set.reset_index(drop=True)
        full_data_set = pd.concat([full_data_set, label], axis=1, sort=False)
        # Get the length of this series
        series_len = full_data_set.shape[0]
        # Get the training and developing and testing set
        train_df = full_data_set[0:(series_len - test_len - test_len)]
        dev_df = full_data_set[(series_len - test_len - test_len):(series_len - test_len)]
        test_df = full_data_set[(series_len - test_len):series_len]
        assert (train_df['X1'].size + dev_df['X1'].size +test_df['X1'].size) == series_len
        # Get the max and min value of each series
        series_max = train_df.max(axis=0)
        series_min = train_df.min(axis=0)
        # Normalize each series to the range between -1 and 1
        train_df.to_csv(save_path + 'train_samples_s'+str(k)+'.csv', index=None)
        dev_df.to_csv(save_path + 'dev_samples_s'+str(k)+'.csv', index=None)
        test_df.to_csv(save_path + 'test_samples_s'+str(k)+'.csv', index=None)
        train_df = 2 * (train_df - series_min) / (series_max - series_min) - 1
        dev_df = 2 * (dev_df - series_min) / (series_max - series_min) - 1
        test_df = 2 * (test_df - series_min) / (series_max - series_min) - 1
        # Generate pandas series for series' mean and standard devation
        series_max_ = pd.DataFrame(series_max, columns=['series_max'])
        series_min_ = pd.DataFrame(series_min, columns=['series_min'])
        # Merge max serie and min serie
        normalize_indicators = pd.concat([series_max_, series_min_], axis=1,sort=False)
        # Storage the normalied indicators to local disk
        normalize_indicators.to_csv(save_path + 'norm_id_s'+str(k)+'.csv')
        # print data set length
        print(100*'-')
        print('series length:{}'.format(series_len))
        print('train data set length:{}'.format(train_df.shape[0]))
        print('development data set length:{}'.format(dev_df.shape[0]))
        print('test set length length:{}'.format(test_df.shape[0]))
        assert test_len==dev_df.shape[0]
        assert test_len==test_df.shape[0]
        train_df.to_csv(save_path + 'minmax_unsample_train_s'+str(k)+'.csv', index=None)
        dev_df.to_csv(save_path + 'minmax_unsample_dev_s'+str(k)+'.csv', index=None)
        test_df.to_csv(save_path + 'minmax_unsample_test_s'+str(k)+'.csv', index=None)




def gen_multi_models_forecast(path,decomposer,lev,test_len,start_id,stop_id,lags_dict,leading_time,wavelet=None,): 
    """ 
    Generate multi-models forecast samples for autoregression problem. 
    Args:
        -path: The source data file path for generate the learning samples.
        -decomposer: The decomposition algorithm.
        -lev: The decomposition level.
        -test_len: The length of testing samples.
        -lags_dict: The lags dict for subsignals.
        -start_id: The start index for validation samples.
        -stop_id: The stop index for validation samples.
        -wavelet: The monther wavelet for Wavelet Transform (WD).
    """ 
    columns=[]
    lags=[]
    if decomposer=='wd':
        data_path = path+wavelet+'-'+str(lev)+'/'
        lags_dict_=lags_dict[wavelet+'-'+str(lev)]
        for i in range(1,lev+2):
            if i==lev+1:
                lags.append(lags_dict_['A'+str(i-1)])
                columns.append('A'+str(i-1))
            else:
                lags.append(lags_dict_['D'+str(i)])
                columns.append('D'+str(i))
    else:
        data_path = path
        lags_dict_=lags_dict[decomposer]
        for i in range(1,lev+1):
            lags.append(lags_dict_['IMF'+str(i)])
            columns.append('IMF'+str(i))

    print('columns:{}'.format(columns))
    print('lags:{}'.format(lags))

    save_path = data_path+"multi_models_"+str(leading_time)+"_ahead_forecast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate training samples
    train_decompose = pd.read_csv(data_path+decomposer.upper()+"_TRAIN.csv")
    for k in range(1,len(columns)+1):
        col = columns[k-1]
        lag = lags[k-1]
        print("Column:{}".format(col))
        print("Lag:{}".format(lag))
        train_df=train_decompose[col]
        train_df_size=train_df.shape[0]
        nparr = train_df.values
        print(nparr)
        train_inputs=pd.DataFrame()
        for i in range(lag):#0,1,2,3...,19
            print("X"+str(i+1))
            x = pd.DataFrame(nparr[i:train_df_size-(lag-i)],columns=['X'+str(i+1)])
            x = x.reset_index(drop=True)
            train_inputs=pd.concat([train_inputs,x],axis=1,sort=False)
        label=pd.DataFrame(nparr[lag+leading_time-1:],columns=['Y'])
        label=label.reset_index(drop=True)
        train_inputs=train_inputs[:train_inputs.shape[0]-(leading_time-1)]
        train_inputs=train_inputs.reset_index(drop=True)
        train_samples=pd.concat([train_inputs,label],axis=1,sort=False)
        print(train_samples)
        train_len = train_samples.shape[0]
        series_max = train_samples.max(axis=0)
        series_min = train_samples.min(axis=0)
        series_max_ = pd.DataFrame(series_max, columns=['series_max'])
        series_min_ = pd.DataFrame(series_min, columns=['series_min'])
        normalize_indicators = pd.concat([series_max_, series_min_], axis=1,sort=False)
        normalize_indicators.to_csv(save_path +'norm_id_s' + str(k) + '.csv')
        train_samples.to_csv(save_path+'train_samples_s'+str(k)+'.csv',index=None)
        train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
        train_samples.to_csv(save_path+'minmax_unsample_train_s'+str(k)+'.csv',index=None)
    
    
        # Generate development and testing samples
        test_path=data_path+decomposer.lower()+'-test/'
        test_imf_df = pd.DataFrame()#测试集样本:输入为分解序列，输出为分解序列
        for j in range(start_id,stop_id+1):#遍历每一个附加分解结果
            data = pd.read_csv(test_path+decomposer.lower()+'_appended_test'+str(j)+'.csv')#取出逐天将test数据附加到traindev后的附加数据集分解的结果
            imf = data[col]#取出这一天之前相应的signal component
            imf_size=imf.shape[0]
            nparr = np.array(imf)#转换为numpy array
            inputs = pd.DataFrame()#这一天之前分解结果形成的训练样本的输入
            for i in range(lag):#逐个生成输入特征变量
                x = pd.DataFrame(nparr[i:imf_size - (lag - i)],columns=['X' + str(i + 1)])
                x=x.reset_index(drop=True)
                inputs = pd.concat([inputs, x], axis=1,sort=False)#合并输入特征变量
            label = pd.DataFrame(nparr[lag+leading_time-1:], columns=['Y'])#生成输出标签列
            label = label.reset_index(drop=True)
            inputs=inputs[:inputs.shape[0]-(leading_time-1)]
            inputs=inputs.reset_index(drop=True)
            full_data_set = pd.concat([inputs, label], axis=1,sort=False)#合并输入与输出，生成样本
            # print(full_data_set.tail())
            series_len=full_data_set.shape[0]#获取样本的行数
            last_imf = full_data_set.iloc[series_len-1,:]#取出最后一行样本作为该天的样本
            print(last_imf)
            test_imf_df = pd.concat([test_imf_df,last_imf],axis=1,sort=False)#合并所有天的样本
            print(test_imf_df)
        test_imf_df=test_imf_df.T#转置测试集样本
        test_imf_df=test_imf_df.reset_index(drop=True)
        dev_samples = test_imf_df[:test_imf_df.shape[0]-test_len]
        test_samples = test_imf_df[test_imf_df.shape[0]-test_len:]
        dev_samples.to_csv(save_path+'dev_samples_s'+str(k)+'.csv',index=None)
        test_samples.to_csv(save_path+'test_samples_s'+str(k)+'.csv',index=None)
        dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
        test_samples = 2 * (test_samples - series_min) / (series_max - series_min) - 1
        assert test_len==dev_samples.shape[0]
        assert test_len==test_samples.shape[0]
        dev_samples.to_csv(save_path+'minmax_unsample_dev_s'+str(k)+'.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test_s'+str(k)+'.csv',index=None)


def gen_one_model_hindcast(path,decomposer,lev,test_len,lags_dict,leading_time,input_columns=None,wavelet=None,):
    """ 
    Generate one-model hindcast decomposition-ensemble samples. 
    Args:
    'path'-- ['string'] The path where the decomposition come from.
    'decomposer'-- ['string'] The decomposition algorithm used for decomposing the original time series.
    'lev'-- ['string'] The decomposition level.
    'test_len'-- ['int'] The size of development and testing samples.
    'lags_dict'-- ['dict list'] The lagged times for subsignals.
    'wavelet'-- ['string'] Choose monther wavelet for wavelet transform.
    """
    output_column=['ORIG']
    if input_columns == None:
        input_columns=[]
        if decomposer=='wd':
            wavelet_level=wavelet+'-'+str(lev)
            data_path = path+wavelet+'-'+str(lev)+'/'
            lags_dict_=lags_dict[wavelet+'-'+str(lev)]
            for i in range(1,lev+2):
                if i==lev+1:
                    input_columns.append('A'+str(i-1))
                else:
                    input_columns.append('D'+str(i))
        else:
            data_path = path
            lags_dict_=lags_dict[decomposer]
            for i in range(1,lev+1):
                input_columns.append('IMF'+str(i))
        save_path = data_path+"one_model_"+str(leading_time)+"_ahead_hindcast_pacf/"
            
    else:
        if decomposer=='wd':
            wavelet_level=wavelet+'-'+str(lev)
            data_path = path+wavelet+'-'+str(lev)+'/'
            lags_dict_=lags_dict[wavelet+'-'+str(lev)]
        else:
            data_path = path
            lags_dict_=lags_dict[decomposer]
        save_path = data_path+"one_model_"+str(leading_time)+"_ahead_hindcast_pacf_mis/"
    lags=[]
    for i in range(len(input_columns)):
        lags.append(lags_dict_[input_columns[i]])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('input columns:{}'.format(input_columns))
    print('output column:{}'.format(output_column))
    print('lags:{}'.format(lags))
    print('save path:{}'.format(save_path))
    
    decompose_file = data_path+decomposer.upper()+"_FULL.csv"
    decompositions = pd.read_csv(decompose_file)
    # Drop NaN
    decompositions.dropna()
    # Get the input data (the decompositions)
    input_data = decompositions[input_columns]
    # Get the output data (the original time series)
    output_data = decompositions[output_column]
    # Get the number of input features
    subsignals_num = input_data.shape[1]
    # Get the data size
    data_size = input_data.shape[0]
    # Compute the samples size
    samples_size = data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each subsignal
    full_samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one subsignal
        one_in = (input_data[input_columns[i]]).values
        oness = pd.DataFrame()
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])
            x = x.reset_index(drop=True)
            oness = pd.concat([oness,x],axis=1,sort=False)
        # make all sample size of each subsignal identical
        oness = oness.iloc[oness.shape[0]-samples_size:]
        oness = oness.reset_index(drop=True)
        full_samples = pd.concat([full_samples,oness],axis=1,sort=False)
    # Get the target
    target = (output_data.values)[max(lags)+leading_time-1:]
    target = pd.DataFrame(target,columns=['Y'])
    # Concat the features and target
    full_samples=full_samples[:full_samples.shape[0]-(leading_time-1)]
    full_samples = full_samples.reset_index(drop=True)
    full_samples = pd.concat([full_samples,target],axis=1,sort=False)
    full_samples = pd.DataFrame(full_samples.values,columns=samples_cols)
    full_samples.to_csv(save_path+'full_samples.csv')
    assert samples_size-leading_time+1 == full_samples.shape[0]
    train_samples = full_samples[0:(full_samples.shape[0] - test_len - test_len)]
    dev_samples = full_samples[(full_samples.shape[0] - test_len - test_len):(full_samples.shape[0] - test_len)]
    test_samples = full_samples[(full_samples.shape[0] - test_len):full_samples.shape[0]]
    assert test_len==dev_samples.shape[0]
    assert test_len==test_samples.shape[0]
    assert (train_samples['X1'].size + dev_samples['X1'].size +test_samples['X1'].size) == samples_size-leading_time+1
    # Get the max and min value of training set
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    train_samples.to_csv(save_path+ 'train_samples.csv', index=None)
    dev_samples.to_csv(save_path+ 'dev_samples.csv', index=None)
    test_samples.to_csv(save_path+'test_samples.csv', index=None)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
    test_samples = 2 * (test_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max_ = pd.DataFrame(series_max, columns=['series_max'])
    series_min_ = pd.DataFrame(series_min, columns=['series_min'])
    # Merge max serie and min serie
    normalize_indicators = pd.concat([series_max_, series_min_], axis=1,sort=False)
    # Storage the normalied indicators to local disk
    # print data set length
    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('Series length:{}'.format(samples_size))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))
    normalize_indicators.to_csv(save_path+'norm_id.csv')
    train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', index=None)
    dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv', index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)
    
   
def gen_one_model_forecast(path,decomposer,lev,start_id,stop_id,test_len,lags_dict,leading_time,input_columns=None,wavelet=None,):
    """ 
    Generate one-model forecast decomposition-ensemble samples. 
    Args:
    'path'-- ['string'] The path where the decomposition come from.
    'decomposer'-- ['string'] The decomposition algorithm used for decomposing the original time series.
    'lev'-- ['string'] The decomposition level.
    'start_id'-- ['int'] The start index of appended decomposition file.
    'stop_id'-- ['int'] The stop index of appended decomposition file.
    'test_len'-- ['int'] The size of development and testing samples.
    'lags_dict'-- ['dict list'] The lagged times for subsignals.
    'leading_time'-- ['int'] The leading time.
    'wavelet'-- ['string'] Choose monther wavelet for wavelet transform.
    """
    output_column=['ORIG']
    if input_columns == None:
        input_columns=[]
        if decomposer=='wd':
            wavelet_level=wavelet+'-'+str(lev)
            data_path = path+wavelet+'-'+str(lev)+'/'
            lags_dict_=lags_dict[wavelet+'-'+str(lev)]
            for i in range(1,lev+2):
                if i==lev+1:
                    input_columns.append('A'+str(i-1))
                else:
                    input_columns.append('D'+str(i))
        else:
            data_path = path
            lags_dict_=lags_dict[decomposer]
            for i in range(1,lev+1):
                input_columns.append('IMF'+str(i))
        save_path = data_path+"one_model_"+str(leading_time)+"_ahead_forecast_pacf/"

    else:
        if decomposer=='wd':
            wavelet_level=wavelet+'-'+str(lev)
            data_path = path+wavelet+'-'+str(lev)+'/'
            lags_dict_=lags_dict[wavelet+'-'+str(lev)]
        else:
            data_path = path
            lags_dict_=lags_dict[decomposer]
        save_path = data_path+"one_model_"+str(leading_time)+"_ahead_forecast_pacf_mis/"
    lags=[]
    print(lags_dict_)
    print(input_columns)
    for i in range(len(input_columns)):
        lags.append(lags_dict_[input_columns[i]])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('input columns:{}'.format(input_columns))
    print('output column:{}'.format(output_column))
    print('lags:{}'.format(lags))
    print('Save path:{}'.format(save_path))
    
    # !!!!!!Generate training samples
    train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
    train_decompositions = pd.read_csv(train_decompose_file)
    # Drop NaN
    train_decompositions.dropna()
    # Get the input data (the decompositions)
    train_input_data = train_decompositions[input_columns]
    # Get the output data (the original time series)
    train_output_data = train_decompositions[output_column]
    # Get the number of input features
    subsignals_num = train_input_data.shape[1]
    # Get the data size
    train_data_size = train_input_data.shape[0]
    # Compute the samples size
    train_samples_size = train_data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    train_samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one input feature
        one_in = (train_input_data[input_columns[i]]).values #subsignal
        oness = pd.DataFrame() #restor input features
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:train_data_size-(lags[i]-j)],columns=['X' + str(j + 1)])
            x = x.reset_index(drop=True)
            oness = pd.concat([oness,x],axis=1,sort=False)
        oness = oness.iloc[oness.shape[0]-train_samples_size:] 
        oness = oness.reset_index(drop=True)
        train_samples = pd.concat([train_samples,oness],axis=1,sort=False)
    # Get the target
    target = (train_output_data.values)[max(lags)+leading_time-1:]
    target = pd.DataFrame(target,columns=['Y'])
    print("target:{}".format(target))
    # Concat the features and target
    train_samples=train_samples[:train_samples.shape[0]-(leading_time-1)]
    train_samples=train_samples.reset_index(drop=True)
    train_samples = pd.concat([train_samples,target],axis=1,sort=False)
    train_samples = pd.DataFrame(train_samples.values,columns=samples_cols)
    train_samples.to_csv(save_path+'train_samples.csv',index=None)
    # assert train_samples_size == train_samples.shape[0]
    # normalize the train_samples
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples.to_csv(save_path+'train_samples.csv',index=None)
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max_ = pd.DataFrame(series_max, columns=['series_max'])
    series_min_ = pd.DataFrame(series_min, columns=['series_min'])
    # Merge max serie and min serie
    normalize_indicators = pd.concat([series_max_, series_min_], axis=1,sort=False)
    normalize_indicators.to_csv(save_path+"norm_id.csv")
    # !!!!!!!!!!!Generate development and testing samples
    dev_test_samples = pd.DataFrame()
    appended_file_path = data_path+decomposer+"-test/"
    for k in range(start_id,stop_id+1):
        #  Load data from local dick
        appended_decompositions = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        appended_decompositions.dropna()
        # Get the input data (the decompositions)
        input_data = appended_decompositions[input_columns]
        # Get the output data (the original time series)
        output_data = appended_decompositions[output_column]
        # Get the number of input features
        subsignals_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate input colmuns for each subsignal
        appended_samples = pd.DataFrame()
        for i in range(subsignals_num):
            # Get one subsignal
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])
                x = x.reset_index(drop=True)
                oness = pd.concat([oness,x],axis=1,sort=False)
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            appended_samples = pd.concat([appended_samples,oness],axis=1,sort=False)
        # Get the target
        target = (output_data.values)[max(lags)+leading_time-1:]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        appended_samples=appended_samples[:appended_samples.shape[0]-(leading_time-1)]
        appended_samples=appended_samples.reset_index(drop=True)
        appended_samples = pd.concat([appended_samples,target],axis=1,sort=False)
        appended_samples = pd.DataFrame(appended_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = appended_samples.iloc[appended_samples.shape[0]-1:]
        dev_test_samples = pd.concat([dev_test_samples,last_sample],axis=0,sort=False)
    dev_test_samples = dev_test_samples.reset_index(drop=True)
    dev_samples=dev_test_samples.iloc[0:dev_test_samples.shape[0]-test_len]
    test_samples=dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
    dev_samples.to_csv(save_path+'dev_samples.csv',index=None)
    test_samples.to_csv(save_path+'test_samples.csv',index=None)
    dev_samples = 2*(dev_samples-series_min)/(series_max-series_min)-1
    test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
    assert test_len==dev_samples.shape[0]
    assert test_len==test_samples.shape[0]
    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))
    train_samples.to_csv(save_path+'minmax_unsample_train.csv',index=None)
    dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    
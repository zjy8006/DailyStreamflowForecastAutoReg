import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from plot_utils import plot_rela_pred,plot_history,plot_error_distribution,plot_subsignals_preds
from metrics_ import PPTS,normalized_mean_square_error
import os

def plot_outliers(y_true,y_pred):
    t=list(range(y_true.shape[0]))
    plt.plot(t,y_true,c='b')
    plt.plot(t,y_pred,c='r')
    plt.plot(t,(y_pred-y_true)/y_true,c='y')



def ensemble_optimization(root_path,station,variables,orig_df,pattern,decomposer=None,lev=None,wavelet=None,criterion='RMSE'):
    # load variables
    lags_dict=variables['lags_dict']
    full_len=variables['full_len']
    train_len = variables['train_len']
    dev_len = variables['dev_len']
    test_len = variables['test_len']

    if pattern.find('one')<0 and pattern.find('multi')<0:
        print('Moonscale Pattern')
        leading_time=int(pattern.split('_')[0])
    else:
        print('Decomposition ensemble pattern')
        leading_time=int(pattern.split('_')[2])

    def dict_to_list(dictionary):
        results=[]
        for key in dictionary:
            results.append(dictionary[key])
        return results

    if decomposer==None:
        subsignals_num=None
        model_path = root_path+'/'+station+'_orig/projects/lstm-models-history/'+pattern+'/'
        lags = lags_dict['orig']
        leading_time=int(pattern.split('_')[0])
        train_samples_len=train_len-lags-leading_time+1

    else:
        leading_time=int(pattern.split('_')[2])
        if wavelet == None:
            model_path = root_path+'/'+station+'_'+decomposer+'/projects/lstm-models-history/'+pattern+'/'
            lags = dict_to_list(lags_dict[decomposer])
            train_samples_len=train_len-max(lags)-leading_time+1
            subsignals_num=lev
            assert lev == len(lags)
        else:
            model_path = root_path+'/'+station+'_'+decomposer+'/projects/lstm-models-history/'+wavelet+'-'+str(lev)+'/'+pattern+'/'
            lags = dict_to_list(lags_dict[wavelet+'-'+str(lev)])
            train_samples_len=train_len-max(lags)-leading_time+1
            subsignals_num=lev+1
            assert lev+1 == len(lags)

    print('Station:{}'.format(station))
    print('Decomposer:{}'.format(decomposer))
    print('Decomposition level:{}'.format(lev))
    print('Prediction pattern:{}'.format(pattern))
    print('Wavelet:{}'.format(wavelet))
    print('full_len:{}'.format(full_len))
    print('train_len:{}'.format(train_len))
    print('dev_len:{}'.format(dev_len))
    print('test_len:{}'.format(test_len))
    print('train_samples_len:{}'.format(train_samples_len))
    print('lags:{}'.format(lags))
    print('Subsignals num:{}'.format(subsignals_num))
    print('Models are developed on {}'.format(criterion))

    if pattern.find('one_model')>=0 or decomposer==None:
        print("################")
        signal_model = model_path+'history/'
        criterion_dict = {}
        for files in os.listdir(signal_model):
            if files.find('.csv')>=0 and (files.find('HISTORY')<0 and files.find('metrics')<0):
                # print(files)
                data = pd.read_csv(signal_model+files)
                dev_y = data['dev_y'][0:dev_len]
                dev_pred = data['dev_pred'][0:dev_len]
                if criterion=='RMSE':
                    criterion_dict[files]=data['rmse_dev'][0]
                elif criterion=='NMSE':
                    NMSE = normalized_mean_square_error(y_true=dev_y,y_pred=dev_pred)
                    criterion_dict[files]=NMSE

        key_min = min(criterion_dict.keys(), key=(lambda k: criterion_dict[k]))
        data = pd.read_csv(signal_model+key_min)
        train_y = data['train_y'][data.shape[0]-train_samples_len:]
        train_pred = data['train_pred'][data.shape[0]-train_samples_len:]
        train_pred[train_pred<0.0]=0.0
        train_y=train_y.reset_index(drop=True)
        train_pred = train_pred.reset_index(drop=True)
        train_results=pd.concat([train_y,train_pred],axis=1,sort=False)
        dev_y = data['dev_y'][0:dev_len]
        dev_pred = data['dev_pred'][0:dev_len]
        dev_pred[dev_pred<0.0]=0.0
        dev_results=pd.concat([dev_y,dev_pred],axis=1,sort=False)
        test_y=data['test_y'][0:test_len]
        test_pred=data['test_pred'][0:test_len]
        test_pred[test_pred<0.0]=0.0
        test_results=pd.concat([test_y,test_pred],axis=1,sort=False)

        max_streamflow = max(orig_df)
        ratio_train = train_pred/max_streamflow
        ratio_dev = dev_pred/max_streamflow
        ratio_test = test_pred/max_streamflow
        rto_train = pd.DataFrame(ratio_train,columns=['train'])['train']
        rto_dev = pd.DataFrame(ratio_dev,columns=['dev'])['dev']
        rto_test = pd.DataFrame(ratio_test,columns=['test'])['test']
        ratio_df = pd.concat([rto_train,rto_dev,rto_test],axis=1)
        ration1_5=ratio_df[ratio_df>1.5]
        ration2=ratio_df[ratio_df>2]
        count_1_5 = pd.concat([
            ration1_5['train'].value_counts(),
            ration1_5['dev'].value_counts(),
            ration1_5['test'].value_counts()], axis=1)
        count_2 = pd.concat([
            ration2['train'].value_counts(),
            ration2['dev'].value_counts(),
            ration2['test'].value_counts()], axis=1)
        count_1_5.to_csv(model_path+'pred_div_maxtrue_ratio1_5_count.csv')
        count_2.to_csv(model_path+'pred_div_maxtrue_ratio2_count.csv')
        ratio_df.to_csv(model_path+'pred_div_maxtrue_ratio.csv')

        print('test_y=\n{}'.format(test_y))
        print('test_pred=\n{}'.format(test_pred))
        

        train_results.to_csv(model_path+'model_train_results.csv',index=None)
        dev_results.to_csv(model_path+'model_dev_results.csv',index=None)
        test_results.to_csv(model_path+'model_test_results.csv',index=None)

        plot_rela_pred(train_y.values,train_pred.values,model_path+'train_pred.png')
        plot_rela_pred(dev_y.values,dev_pred.values,model_path+'dev_pred.png')
        plot_rela_pred(test_y.values,test_pred.values,model_path+'test_pred.png')
        
        train_nse = r2_score(y_true=train_y.values,y_pred=train_pred.values)
        dev_nse = r2_score(y_true=dev_y.values,y_pred=dev_pred.values)
        test_nse = r2_score(y_true=test_y.values,y_pred=test_pred.values)
        train_nmse = normalized_mean_square_error(y_true=train_y,y_pred=train_pred)
        dev_nmse = normalized_mean_square_error(y_true=dev_y,y_pred=dev_pred)
        test_nmse = normalized_mean_square_error(y_true=test_y,y_pred=test_pred)
        train_rmse = math.sqrt(mean_squared_error(train_y.values, train_pred.values))
        dev_rmse = math.sqrt(mean_squared_error(dev_y.values, dev_pred.values))
        test_rmse = math.sqrt(mean_squared_error(test_y.values, test_pred.values))
        train_nrmse = math.sqrt(mean_squared_error(train_y.values, train_pred.values))/(sum(train_y.values)/len(train_y.values))
        dev_nrmse = math.sqrt(mean_squared_error(dev_y.values, dev_pred.values))/(sum(dev_y.values)/len(dev_y.values))
        test_nrmse = math.sqrt(mean_squared_error(test_y.values, test_pred.values))/(sum(test_y.values)/len(test_y.values))
        train_mae = mean_absolute_error(y_true=train_y.values,y_pred=train_pred.values)
        dev_mae = mean_absolute_error(y_true=dev_y.values,y_pred=dev_pred.values)
        test_mae = mean_absolute_error(y_true=test_y.values,y_pred=test_pred.values)
        train_mape=np.mean(np.abs((train_y.values - train_pred.values) / train_y.values)) * 100
        dev_mape=np.mean(np.abs((dev_y.values - dev_pred.values) / dev_y.values)) * 100
        test_mape=np.mean(np.abs((test_y.values - test_pred.values) / test_y.values)) * 100
        train_ppts = PPTS(train_y.values,train_pred.values,5)
        dev_ppts = PPTS(dev_y.values,dev_pred.values,5)
        test_ppts = PPTS(test_y.values,test_pred.values,5)
        print('#'*25+'train_ppts:\n{}'.format(train_ppts))
        print('#'*25+'dev_ppts:\n{}'.format(dev_ppts))
        print('#'*25+'test_ppts:\n{}'.format(test_ppts))
        metrics={
            'optimal':key_min,
            'train_nse':train_nse,
            'train_nmse':train_nmse,
            'train_rmse':train_rmse,
            'train_nrmse':train_nrmse,
            'train_mae':train_mae,
            'train_mape':train_mape,
            'train_ppts':train_ppts,
            'dev_nse':dev_nse,
            'dev_nmse':dev_nmse,
            'dev_rmse':dev_rmse,
            'dev_nrmse':dev_nrmse,
            'dev_mae':dev_mae,
            'dev_mape':dev_mape,
            'dev_ppts':dev_ppts,
            'test_nse':test_nse,
            'test_nmse':test_nmse,
            'test_rmse':test_rmse,
            'test_nrmse':test_nrmse,
            'test_mae':test_mae,
            'test_mape':test_mape,
            'test_ppts':test_ppts,
        }

        metrics = pd.DataFrame(metrics,index=[0])
        metrics.to_csv(model_path+'model_metrics.csv')

    else:
        train_ens_pred = pd.DataFrame()
        dev_ens_pred = pd.DataFrame()
        test_ens_pred = pd.DataFrame()
        train_ens_y = pd.DataFrame()
        dev_ens_y = pd.DataFrame()
        test_ens_y = pd.DataFrame()
        subsignal_metrics=pd.DataFrame()

        for i in range(1,subsignals_num+1):
            sub_signal = 's'+str(i)
            signal_model = model_path+'history/'+sub_signal+'/'
            criterion_dict = {}
            for files in os.listdir(signal_model):
                if files.find('.csv')>=0 and (files.find('HISTORY')<0 and files.find('metrics')<0):
                    # print(files)
                    data = pd.read_csv(signal_model+files)
                    dev_y = data['dev_y'][0:dev_len]
                    dev_pred = data['dev_pred'][0:dev_len]
                    if criterion=='RMSE':
                        criterion_dict[files]=data['rmse_dev'][0]
                    elif criterion=='NMSE':
                        NMSE = normalized_mean_square_error(y_true=dev_y,y_pred=dev_pred)
                        criterion_dict[files]=NMSE
                    

            key_min = min(criterion_dict.keys(), key=(lambda k: criterion_dict[k]))
            data = pd.read_csv(signal_model+key_min)
            train_y = data['train_y'][data.shape[0]-train_samples_len:]
            train_pred = data['train_pred'][data.shape[0]-train_samples_len:]
            train_y=train_y.reset_index(drop=True)
            train_pred = train_pred.reset_index(drop=True)
            dev_y = data['dev_y'][0:dev_len]
            dev_pred = data['dev_pred'][0:dev_len]
            test_y=data['test_y'][0:test_len]
            test_pred=data['test_pred'][0:test_len]

            train_nse = r2_score(y_true=train_y.values,y_pred=train_pred.values)
            dev_nse = r2_score(y_true=dev_y.values,y_pred=dev_pred.values)
            test_nse = r2_score(y_true=test_y.values,y_pred=test_pred.values)
            train_nmse = normalized_mean_square_error(y_true=train_y,y_pred=train_pred)
            dev_nmse = normalized_mean_square_error(y_true=dev_y,y_pred=dev_pred)
            test_nmse = normalized_mean_square_error(y_true=test_y,y_pred=test_pred)
            train_rmse = math.sqrt(mean_squared_error(train_y.values, train_pred.values))
            dev_rmse = math.sqrt(mean_squared_error(dev_y.values, dev_pred.values))
            test_rmse = math.sqrt(mean_squared_error(test_y.values, test_pred.values))
            train_nrmse = math.sqrt(mean_squared_error(train_y.values, train_pred.values))/(sum(train_y.values)/len(train_y.values))
            dev_nrmse = math.sqrt(mean_squared_error(dev_y.values, dev_pred.values))/(sum(dev_y.values)/len(dev_y.values))
            test_nrmse = math.sqrt(mean_squared_error(test_y.values, test_pred.values))/(sum(test_y.values)/len(test_y.values))
            train_mae = mean_absolute_error(y_true=train_y.values,y_pred=train_pred.values)
            dev_mae = mean_absolute_error(y_true=dev_y.values,y_pred=dev_pred.values)
            test_mae = mean_absolute_error(y_true=test_y.values,y_pred=test_pred.values)
            train_mape=np.mean(np.abs((train_y.values - train_pred.values) / train_y.values)) * 100
            dev_mape=np.mean(np.abs((dev_y.values - dev_pred.values) / dev_y.values)) * 100
            test_mape=np.mean(np.abs((test_y.values - test_pred.values) / test_y.values)) * 100
            train_ppts = PPTS(train_y.values,train_pred.values,5)
            dev_ppts = PPTS(dev_y.values,dev_pred.values,5)
            test_ppts = PPTS(test_y.values,test_pred.values,5)

            print('#'*25+'train_ppts:\n{}'.format(train_ppts))
            print('#'*25+'dev_ppts:\n{}'.format(dev_ppts))
            print('#'*25+'test_ppts:\n{}'.format(test_ppts))

            metrics={
                'optimal':key_min,
                'train_nse':train_nse,
                'train_nmse':train_nmse,
                'train_rmse':train_rmse,
                'train_nrmse':train_nrmse,
                'train_mae':train_mae,
                'train_mape':train_mape,
                'train_ppts':train_ppts,
                'dev_nse':dev_nse,
                'dev_nmse':dev_nmse,
                'dev_rmse':dev_rmse,
                'dev_nrmse':dev_nrmse,
                'dev_mae':dev_mae,
                'dev_mape':dev_mape,
                'dev_ppts':dev_ppts,
                'test_nse':test_nse,
                'test_nmse':test_nmse,
                'test_rmse':test_rmse,
                'test_nrmse':test_nrmse,
                'test_mae':test_mae,
                'test_mape':test_mape,
                'test_ppts':test_ppts,
            }

            metrics = pd.DataFrame(metrics,index=['s'+str(i)])
            subsignal_metrics=pd.concat([subsignal_metrics,metrics],sort=False)

            train_ens_pred = pd.concat([train_ens_pred,train_pred],axis=1)
            dev_ens_pred = pd.concat([dev_ens_pred,dev_pred],axis=1)
            test_ens_pred = pd.concat([test_ens_pred,test_pred],axis=1)
            train_ens_y = pd.concat([train_ens_y,train_y],axis=1)
            dev_ens_y = pd.concat([dev_ens_y,dev_y],axis=1)
            test_ens_y = pd.concat([test_ens_y,test_y],axis=1)

        subsignal_metrics.to_csv(model_path+'subsignals_metrics.csv')
        plot_subsignals_preds(subsignals_y=test_ens_y,subsignals_pred=test_ens_pred,fig_savepath=model_path+'subsignals_pred.png')
        train_pred=train_ens_pred.sum(axis=1)
        dev_pred=dev_ens_pred.sum(axis=1)
        test_pred=test_ens_pred.sum(axis=1)
        train_pred[train_pred<0.0]=0.0
        dev_pred[dev_pred<0.0]=0.0
        test_pred[test_pred<0.0]=0.0

        train_pred = train_pred.values
        dev_pred = dev_pred.values
        test_pred = test_pred.values

        print('train_pred len:{}'.format(len(train_pred)))
    

        train_y=orig_df[(train_len-train_samples_len):train_len]
        print('train_y len:{}'.format(train_y.shape[0]))
        dev_y = orig_df[train_len:train_len+test_len]
        test_y = orig_df[train_len+test_len:]
        train_y=train_y.reset_index(drop=True)
        dev_y = dev_y.reset_index(drop=True)
        test_y = test_y.reset_index(drop=True)
        train_y=train_y.values
        dev_y=dev_y.values
        test_y=test_y.values

        max_streamflow = max(orig_df)
        ratio_train = train_pred/max_streamflow
        ratio_dev = dev_pred/max_streamflow
        ratio_test = test_pred/max_streamflow
        rto_train = pd.DataFrame(ratio_train,columns=['train'])['train']
        rto_dev = pd.DataFrame(ratio_dev,columns=['dev'])['dev']
        rto_test = pd.DataFrame(ratio_test,columns=['test'])['test']
        ratio_df = pd.concat([rto_train,rto_dev,rto_test],axis=1)
        ration1_5=ratio_df[ratio_df>1.5]
        ration2=ratio_df[ratio_df>2]
        count_1_5 = pd.concat([
            ration1_5['train'].value_counts(),
            ration1_5['dev'].value_counts(),
            ration1_5['test'].value_counts()], axis=1)
        count_2 = pd.concat([
            ration2['train'].value_counts(),
            ration2['dev'].value_counts(),
            ration2['test'].value_counts()], axis=1)
        count_1_5.to_csv(model_path+'pred_div_maxtrue_ratio1_5_count.csv')
        count_2.to_csv(model_path+'pred_div_maxtrue_ratio2_count.csv')
        ratio_df.to_csv(model_path+'pred_div_maxtrue_ratio.csv')
        

        train_nse = r2_score(y_true=train_y,y_pred=train_pred)
        train_nmse = normalized_mean_square_error(y_true=train_y,y_pred=train_pred)
        train_rmse = math.sqrt(mean_squared_error(train_y, train_pred))
        train_nrmse = math.sqrt(mean_squared_error(train_y, train_pred))/(sum(train_y)/len(train_y))
        train_mae=mean_absolute_error(train_y, train_pred)
        train_mape=np.mean(np.abs((train_y - train_pred) / train_y)) * 100
        train_ppts = PPTS(train_y,train_pred,5)

        dev_nse = r2_score(y_true=dev_y,y_pred=dev_pred)
        dev_nmse = normalized_mean_square_error(y_true=dev_y,y_pred=dev_pred)
        dev_rmse = math.sqrt(mean_squared_error(dev_y, dev_pred))
        dev_nrmse = math.sqrt(mean_squared_error(dev_y, dev_pred))/(sum(dev_y)/len(dev_y))
        dev_mae=mean_absolute_error(dev_y, dev_pred)
        dev_mape=np.mean(np.abs((dev_y - dev_pred) / dev_y)) * 100
        dev_ppts = PPTS(dev_y,dev_pred,5)

        test_nse = r2_score(y_true=test_y,y_pred=test_pred)
        test_nmse = normalized_mean_square_error(y_true=test_y,y_pred=test_pred)
        test_rmse = math.sqrt(mean_squared_error(test_y, test_pred))
        test_nrmse = math.sqrt(mean_squared_error(test_y, test_pred))/(sum(test_y)/len(test_y))
        test_mae=mean_absolute_error(test_y, test_pred)
        test_mape=np.mean(np.abs((test_y - test_pred) / test_y)) * 100
        test_ppts = PPTS(test_y,test_pred,5)
        model_metrics={
            'train_nse':train_nse,
            'train_nmse':train_nmse,
            'train_rmse':train_rmse,
            'train_nrmse':train_nrmse,
            'train_mae':train_mae,
            'train_mape':train_mape,
            'train_ppts':train_ppts,
            'dev_nse':dev_nse,
            'dev_nmse':dev_nmse,
            'dev_rmse':dev_rmse,
            'dev_nrmse':dev_nrmse,
            'dev_mae':dev_mae,
            'dev_mape':dev_mape,
            'dev_ppts':dev_ppts,
            'test_nse':test_nse,
            'test_nmse':test_nmse,
            'test_rmse':test_rmse,
            'test_nrmse':test_nrmse,
            'test_mae':test_mae,
            'test_mape':test_mape,
            'test_ppts':test_ppts,
        }
        model_train_results= {'train_y':train_y,'train_pred':train_pred,}
        model_dev_results= {'dev_y':dev_y,'dev_pred':dev_pred, }
        model_test_results= {'test_y':test_y,'test_pred':test_pred,}
        MODEL_METRICS = pd.DataFrame(model_metrics,index=np.arange(start=0,stop=1,step=1))
        MODEL_TRAIN_RESULTS = pd.DataFrame(model_train_results,index=np.arange(start=0,stop=train_samples_len,step=1))
        MODEL_DEV_RESULTS = pd.DataFrame(model_dev_results,index=np.arange(start=0,stop=dev_len,step=1))
        MODEL_TEST_RESULTS = pd.DataFrame(model_test_results,index=np.arange(start=0,stop=test_len,step=1))
        MODEL_METRICS.to_csv(model_path+'model_metrics.csv')
        MODEL_TRAIN_RESULTS.to_csv(model_path+'model_train_results.csv')
        MODEL_DEV_RESULTS.to_csv(model_path+'model_dev_results.csv')
        MODEL_TEST_RESULTS.to_csv(model_path+'model_test_results.csv')
        plot_rela_pred(train_y,train_pred,model_path+'train_pred.png')
        plot_rela_pred(dev_y,dev_pred,model_path+'dev_pred.png')
        plot_rela_pred(test_y,test_pred,model_path+'test_pred.png')
    plt.close('all')
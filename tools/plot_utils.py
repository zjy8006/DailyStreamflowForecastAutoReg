import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
from fit_line import compute_linear_fit

def plot_rela_pred(records, predictions, fig_savepath):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model
        fig_savepath: the path where the plot figure will be saved.
    """
    length = records.size
    t = np.linspace(start=1, stop=length, num=length)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.xticks()
    plt.yticks()
    plt.xlabel('Time(d)', )
    plt.ylabel("flow(" + r"$m^3$" + "/s)", )
    plt.plot(t, records, '-', color='blue', label='records',linewidth=1.0)
    plt.plot(t, predictions, '--', color='red', label='predictions',linewidth=1.0)
    plt.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.005,1.2),
        shadow=False,
        frameon=False,
        )
    plt.subplot(1, 2, 2,aspect='equal')
    pred_min =predictions.min()
    pred_max = predictions.max()
    record_min = records.min()
    record_max = records.max()
    print('pred_min={}'.format(pred_min))
    print('pred_max={}'.format(pred_max))
    print('record_min={}'.format(record_min))
    print('record_max={}'.format(record_max))
    if pred_min<record_min:
        xymin = pred_min
    else:
        xymin = record_min
    if pred_max>record_max:
        xymax = pred_max
    else:
        xymax=record_max
    print('xymin={}'.format(xymin))
    print('xymax={}'.format(xymax))
    xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * xx + coeff[1]
    # print('a:{}'.format(coeff[0]))
    # print('b:{}'.format(coeff[1]))
    # linear_fit = coeff[0] * predictions + coeff[1]
    # ideal_fit = 1 * predictions
    plt.xticks()
    plt.yticks()
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', )
    plt.ylabel('records(' + r'$m^3$' + '/s)', )
    # plt.plot(predictions, records, 'o', color='blue', label='',markersize=6.5)
    plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    # plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    # plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.xlim([xymin,xymax])
    plt.ylim([xymin,xymax])
    plt.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=False,
        frameon=False,
        )
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.tight_layout()
    plt.savefig(fig_savepath, format='PNG', dpi=300)
    # plt.show()

def plot_history(history,path1,path2):
    hist = pd.DataFrame(history.history)
    hist['epoch']=history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs error')
    plt.plot(hist['epoch'],hist['mean_absolute_error'],label='Train error')
    plt.plot(hist['epoch'],hist['val_mean_absolute_error'],label='Val error')
    # plt.ylim([0,0.04])
    plt.legend()
    plt.savefig(path1, format='PNG', dpi=300)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    # plt.ylim([0,0.04])
    plt.legend()
    plt.savefig(path2, format='PNG', dpi=300)
    # plt.show()

def plot_error_distribution(predictions,labels,fig_savepath):
    """
    Plot error distribution from the predictions and labels.
    Args:
        predictions: prdictions obtained by ml 
        labels: real records
        fig_savepath: path to save a error distribution figure

    Return
        A figure of error distribution
    """
    plt.figure()
    error = predictions - labels
    # plt.hist(error,bins=25)
    plt.hist(error, 50, density=True,log=True, facecolor='g', alpha=0.75)
    plt.xlabel('Prediction Error')
    plt.ylabel('count')
    plt.savefig(fig_savepath, format='PNG', dpi=300)
    # plt.show()


def plot_subsignals_preds(subsignals_y, subsignals_pred, fig_savepath):
    
    plt.figure(num=1,figsize=(8,subsignals_y.shape[1]))
    # plt.figure(num=1)
    for i in range(1,subsignals_y.shape[1]+1):
        y_true = subsignals_y.iloc[:,i-1]
        y_pred = subsignals_pred.iloc[:,i-1]
        t = np.linspace(start=1, stop=y_true.shape[0]+1, num=y_true.shape[0])
        plt.subplot(subsignals_y.shape[1],2,2*i-1)#1,3,5
        plt.xticks()
        plt.yticks()
        plt.xlabel('Time(d)', )
        plt.ylabel("flow(" + r"$m^3$" + "/s)", )
        plt.plot(t, y_true, '-', color='blue', label='records',linewidth=1.0)
        plt.plot(t, y_pred, '--', color='red', label='predictions',linewidth=1.0)
        plt.legend()
        plt.subplot(subsignals_y.shape[1],2,2*i)#2,3,6
        xx,linear_fit,xymin,xymax=compute_linear_fit(y_true,y_pred)
        plt.xlabel('predictions(' + r'$m^3$' + '/s)')
        plt.ylabel('records(' + r'$m^3$' + '/s)', )
        plt.plot(y_pred, y_true, 'o', color='blue', label='', linewidth=1.0)
        plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        plt.legend()
        plt.subplots_adjust(left=0.1, bottom=0.06, right=0.98, top=0.98, hspace=0.3, wspace=0.3)
        plt.savefig(fig_savepath, format='PNG', dpi=300)
    # plt.show()
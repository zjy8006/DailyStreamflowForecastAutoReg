import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from metrics_ import PPTS

def dump_train_dev_to_excel(
        path,
        train_y=None,
        train_pred=None,
        dev_y=None,
        dev_pred=None,
        test_y=None,
        test_pred=None,
):
    writer = pd.ExcelWriter(path)
    # convert the test_pred numpy array into Dataframe series
    # test_y = pd.DataFrame(test_pred, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, columns=['test_pred'])['test_pred']

    # train_y = pd.DataFrame(train_y, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(train_pred, columns=['train_pred'])['train_pred']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, columns=['dev_pred'])['dev_pred']
    results = pd.DataFrame(
        pd.concat(
            [test_y, test_pred, train_y, train_pred, dev_y, dev_pred], axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()


""" 
y_aa = np.array([1,2,3,4,5,6,7,8,7,9,1,0,11,12,15,14,17])
aa = pd.DataFrame([1,2,3,4,5,6,7,8,9],columns=['aa'])['aa']
y_bb = np.array([11,12,13,14,15,16])
bb = pd.DataFrame(['a','b','c','d','e','f','g','h'],columns=['bb'])['bb']
dump_to_excel('F:/ml_fp_lytm/tf_projects/test/test.xlsx',aa,y_aa,bb,y_bb) 
"""


def dump_train_dev_test_to_excel(
        path,
        train_y=None,
        train_pred=None,
        r2_train=None,
        rmse_train=None,
        mae_train=None,
        mape_train=None,
        ppts_train=None,
        dev_y=None,
        dev_pred=None,
        r2_dev=None,
        rmse_dev=None,
        mae_dev=None,
        mape_dev=None,
        ppts_dev=None,
        test_y=None,
        test_pred=None,
        r2_test=None,
        rmse_test=None,
        mae_test=None,
        mape_test=None,
        ppts_test=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        r2_train: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        r2_dev: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        r2_test: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_train = pd.Index(np.linspace(1, train_y.size, train_y.size))
    index_dev = pd.Index(np.linspace(1, dev_y.size, dev_y.size))
    index_test = pd.Index(np.linspace(1, test_y.size, test_y.size))

    # convert the train_pred numpy array into Dataframe series
    train_y = pd.DataFrame(list(train_y), index=index_train, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    r2_train = pd.DataFrame([r2_train], columns=['r2_train'])['r2_train']
    rmse_train = pd.DataFrame([rmse_train], columns=['rmse_train'])['rmse_train']
    mae_train = pd.DataFrame([mae_train], columns=['mae_train'])['mae_train']
    mape_train = pd.DataFrame([mape_train], columns=['mape_train'])['mape_train']
    ppts_train = pd.DataFrame([ppts_train],columns=['ppts_train'])['ppts_train']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y = pd.DataFrame(list(dev_y), index=index_dev, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, index=index_dev, columns=['dev_pred'])['dev_pred']
    r2_dev = pd.DataFrame([r2_dev], columns=['r2_dev'])['r2_dev']
    rmse_dev = pd.DataFrame([rmse_dev], columns=['rmse_dev'])['rmse_dev']
    mae_dev = pd.DataFrame([mae_dev], columns=['mae_dev'])['mae_dev']
    mape_dev = pd.DataFrame([mape_dev], columns=['mape_dev'])['mape_dev']
    ppts_dev = pd.DataFrame([ppts_dev], columns=['ppts_dev'])['ppts_dev']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    r2_test = pd.DataFrame([r2_test], columns=['r2_test'])['r2_test']
    rmse_test = pd.DataFrame([rmse_test], columns=['rmse_test'])['rmse_test']
    mae_test = pd.DataFrame([mae_test], columns=['mae_test'])['mae_test']
    mape_test = pd.DataFrame([mape_test], columns=['mape_test'])['mape_test']
    ppts_test = pd.DataFrame([ppts_test], columns=['ppts_test'])['ppts_test']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                r2_train,
                rmse_train,
                mae_train,
                mape_train,
                ppts_train,
                dev_y,
                dev_pred,
                r2_dev,
                rmse_dev,
                mae_dev,
                mape_dev,
                ppts_dev,
                test_y,
                test_pred,
                r2_test,
                rmse_test,
                mae_test,
                mape_test,
                ppts_test,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()

def dump_train_dev_test_to_csv(
        path,
        train_y=None,
        train_pred=None,
        r2_train=None,
        rmse_train=None,
        mae_train=None,
        mape_train=None,
        ppts_train=None,
        dev_y=None,
        dev_pred=None,
        r2_dev=None,
        rmse_dev=None,
        mae_dev=None,
        mape_dev=None,
        ppts_dev=None,
        test_y=None,
        test_pred=None,
        r2_test=None,
        rmse_test=None,
        mae_test=None,
        mape_test=None,
        ppts_test=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        r2_train: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        r2_dev: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        r2_test: R square value for testing records and predictions, type float.
    """

    index_train = pd.Index(np.linspace(0, train_y.size-1, train_y.size))
    index_dev = pd.Index(np.linspace(0, dev_y.size-1, dev_y.size))
    index_test = pd.Index(np.linspace(0, test_y.size-1, test_y.size))

    # convert the train_pred numpy array into Dataframe series
    train_y = pd.DataFrame(list(train_y), index=index_train, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    r2_train = pd.DataFrame([r2_train], columns=['r2_train'])['r2_train']
    rmse_train = pd.DataFrame([rmse_train], columns=['rmse_train'])['rmse_train']
    mae_train = pd.DataFrame([mae_train], columns=['mae_train'])['mae_train']
    mape_train = pd.DataFrame([mape_train], columns=['mape_train'])['mape_train']
    ppts_train = pd.DataFrame([ppts_train],columns=['ppts_train'])['ppts_train']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y = pd.DataFrame(list(dev_y), index=index_dev, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, index=index_dev, columns=['dev_pred'])['dev_pred']
    r2_dev = pd.DataFrame([r2_dev], columns=['r2_dev'])['r2_dev']
    rmse_dev = pd.DataFrame([rmse_dev], columns=['rmse_dev'])['rmse_dev']
    mae_dev = pd.DataFrame([mae_dev], columns=['mae_dev'])['mae_dev']
    mape_dev = pd.DataFrame([mape_dev], columns=['mape_dev'])['mape_dev']
    ppts_dev = pd.DataFrame([ppts_dev], columns=['ppts_dev'])['ppts_dev']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    r2_test = pd.DataFrame([r2_test], columns=['r2_test'])['r2_test']
    rmse_test = pd.DataFrame([rmse_test], columns=['rmse_test'])['rmse_test']
    mae_test = pd.DataFrame([mae_test], columns=['mae_test'])['mae_test']
    mape_test = pd.DataFrame([mape_test], columns=['mape_test'])['mape_test']
    ppts_test = pd.DataFrame([ppts_test], columns=['ppts_test'])['ppts_test']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                r2_train,
                rmse_train,
                mae_train,
                mape_train,
                ppts_train,
                dev_y,
                dev_pred,
                r2_dev,
                rmse_dev,
                mae_dev,
                mape_dev,
                ppts_dev,
                test_y,
                test_pred,
                r2_test,
                rmse_test,
                mae_test,
                mape_test,
                ppts_test,
            ],
            axis=1))
    results.to_csv(path)

def dump_train_dev12_test_to_excel(
        path,
        train_y=None,
        train_pred=None,
        r2_train=None,
        rmse_train=None,
        mae_train=None,
        mape_train=None,
        ppts_train=None,
        dev_y1=None,
        dev1_pred=None,
        r2_dev1=None,
        rmse_dev1=None,
        mae_dev1=None,
        mape_dev1=None,
        ppts_dev1=None,
        dev_y2=None,
        dev2_pred=None,
        r2_dev2=None,
        rmse_dev2=None,
        mae_dev2=None,
        mape_dev2=None,
        ppts_dev2=None,
        test_y=None,
        test_pred=None,
        r2_test=None,
        rmse_test=None,
        mae_test=None,
        mape_test=None,
        ppts_test=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        r2_train: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        r2_dev: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        r2_test: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_train = pd.Index(np.linspace(1, train_y.size, train_y.size))
    index_dev1 = pd.Index(np.linspace(1, dev_y1.size, dev_y1.size))
    index_dev2 = pd.Index(np.linspace(1, dev_y2.size, dev_y2.size))
    index_test = pd.Index(np.linspace(1, test_y.size, test_y.size))

    # convert the train_pred numpy array into Dataframe series
    train_y = pd.DataFrame(list(train_y), index=index_train, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    r2_train = pd.DataFrame([r2_train], columns=['r2_train'])['r2_train']
    rmse_train = pd.DataFrame([rmse_train], columns=['rmse_train'])['rmse_train']
    mae_train = pd.DataFrame([mae_train], columns=['mae_train'])['mae_train']
    mape_train = pd.DataFrame([mape_train], columns=['mape_train'])['mape_train']
    ppts_train = pd.DataFrame([ppts_train],columns=['ppts_train'])['ppts_train']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y1 = pd.DataFrame(list(dev_y1), index=index_dev1, columns=['dev_y1'])['dev_y1']
    dev1_pred = pd.DataFrame(dev1_pred, index=index_dev1, columns=['dev1_pred'])['dev1_pred']
    r2_dev1 = pd.DataFrame([r2_dev1], columns=['r2_dev1'])['r2_dev1']
    rmse_dev1 = pd.DataFrame([rmse_dev1], columns=['rmse_dev1'])['rmse_dev1']
    mae_dev1 = pd.DataFrame([mae_dev1], columns=['mae_dev1'])['mae_dev1']
    mape_dev1 = pd.DataFrame([mape_dev1], columns=['mape_dev1'])['mape_dev1']
    ppts_dev1 = pd.DataFrame([ppts_dev1], columns=['ppts_dev1'])['ppts_dev1']

    dev_y2 = pd.DataFrame(list(dev_y2), index=index_dev2, columns=['dev_y2'])['dev_y2']
    dev2_pred = pd.DataFrame(dev2_pred, index=index_dev2, columns=['dev2_pred'])['dev2_pred']
    r2_dev2 = pd.DataFrame([r2_dev2], columns=['r2_dev2'])['r2_dev2']
    rmse_dev2 = pd.DataFrame([rmse_dev2], columns=['rmse_dev2'])['rmse_dev2']
    mae_dev2 = pd.DataFrame([mae_dev2], columns=['mae_dev2'])['mae_dev2']
    mape_dev2 = pd.DataFrame([mape_dev2], columns=['mape_dev2'])['mape_dev2']
    ppts_dev2 = pd.DataFrame([ppts_dev2], columns=['ppts_dev2'])['ppts_dev2']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    r2_test = pd.DataFrame([r2_test], columns=['r2_test'])['r2_test']
    rmse_test = pd.DataFrame([rmse_test], columns=['rmse_test'])['rmse_test']
    mae_test = pd.DataFrame([mae_test], columns=['mae_test'])['mae_test']
    mape_test = pd.DataFrame([mape_test], columns=['mape_test'])['mape_test']
    ppts_test = pd.DataFrame([ppts_test], columns=['ppts_test'])['ppts_test']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                r2_train,
                rmse_train,
                mae_train,
                mape_train,
                ppts_train,
                dev_y1,
                dev1_pred,
                r2_dev1,
                rmse_dev1,
                mae_dev1,
                mape_dev1,
                ppts_dev1,
                dev_y2,
                dev2_pred,
                r2_dev2,
                rmse_dev2,
                mae_dev2,
                mape_dev2,
                ppts_dev2,
                test_y,
                test_pred,
                r2_test,
                rmse_test,
                mae_test,
                mape_test,
                ppts_test,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()

def dump_test_to_excel(
        path,
        test_y=None,
        test_pred=None,
        r2_test=None,
        rmse_test=None,
        mae_test=None,
        mape_test=None,
        ppts_test=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        r2_train: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        r2_dev: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        r2_test: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_test = pd.Index(np.linspace(1, test_y.size, test_y.size))

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    r2_test = pd.DataFrame([r2_test], columns=['r2_test'])['r2_test']
    rmse_test = pd.DataFrame([rmse_test], columns=['rmse_test'])['rmse_test']
    mae_test = pd.DataFrame([mae_test], columns=['mae_test'])['mae_test']
    mape_test = pd.DataFrame([mape_test], columns=['mape_test'])['mape_test']
    ppts_test = pd.DataFrame([ppts_test], columns=['ppts_test'])['ppts_test']

    results = pd.DataFrame(
        pd.concat(
            [
                test_y,
                test_pred,
                r2_test,
                rmse_test,
                mae_test,
                mape_test,
                ppts_test,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()

def dum_pred_results(path,train_predictions,train_y,dev_predictions,dev_y,test_predictions,test_y):
    """ 
    Dump real records (labels) and predictions as well as evaluation criteria (metrix R2,RMSE,MAE,MAPE,PPTS) to excel.
    Args:
        path: The local disk path to dump data into.
        train_predictions: predictions of training set with numpy array type.
        train_y: records of training set with numpy array type.
        dev_predictions: predictions of development set with numpy array type.
        dev_y: records of development set with numpy array type.
        test_predictions: predictions of testing set with numpy array type.
        test_y: records of testing set with numpy array type.
    
    Return:
    An MS-office excel file
    """
    # compute R square
    r2_train = r2_score(train_y, train_predictions)
    r2_dev = r2_score(dev_y, dev_predictions)
    r2_test = r2_score(test_y, test_predictions)
    # compute MSE
    rmse_train = math.sqrt(mean_squared_error(train_y, train_predictions))
    rmse_dev = math.sqrt(mean_squared_error(dev_y, dev_predictions))
    rmse_test = math.sqrt(mean_squared_error(test_y, test_predictions))
    # compute MAE
    mae_train = mean_absolute_error(train_y, train_predictions)
    mae_dev = mean_absolute_error(dev_y, dev_predictions)
    mae_test = mean_absolute_error(test_y, test_predictions)
    # compute MAPE
    mape_train=np.mean(np.abs((train_y - train_predictions) / train_y)) * 100
    mape_dev=np.mean(np.abs((dev_y - dev_predictions) / dev_y)) * 100
    mape_test=np.mean(np.abs((test_y - test_predictions) / test_y)) * 100
    # compute PPTS
    ppts_train = PPTS(train_y,train_predictions,5)
    ppts_dev = PPTS(dev_y,dev_predictions,5)
    ppts_test = PPTS(test_y,test_predictions,5)

    dump_train_dev_test_to_csv(
            path=path,
            train_y=train_y,
            train_pred=train_predictions,
            r2_train=r2_train,
            rmse_train=rmse_train,
            mae_train=mae_train,
            mape_train=mape_train,
            ppts_train=ppts_train,
            dev_y=dev_y,
            dev_pred=dev_predictions,
            r2_dev=r2_dev,
            rmse_dev=rmse_dev,
            mae_dev=mae_dev,
            mape_dev=mape_dev,
            ppts_dev=ppts_dev,
            test_y=test_y,
            test_pred=test_predictions,
            r2_test=r2_test,
            rmse_test=rmse_test,
            mae_test=mae_test,
            mape_test=mape_test,
            ppts_test=ppts_test,
            )


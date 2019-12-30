import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def compute_linear_fit(records,predictions):
    pred_min =predictions.min()
    pred_max = predictions.max()
    record_min = records.min()
    record_max = records.max()
    if pred_min<record_min:
        xymin = pred_min
    else:
        xymin = record_min
    if pred_max>record_max:
        xymax = pred_max
    else:
        xymax=record_max
    xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * xx + coeff[1]
    return xx,linear_fit,xymin,xymax

def compute_2_linear_fit(records1,predictions1,records2,predictions2):
    pred_min1 =predictions1.min()
    pred_max1 = predictions1.max()
    record_min1 = records1.min()
    record_max1 = records1.max()
    if pred_min1<record_min1:
        xymin1 = pred_min1
    else:
        xymin1 = record_min1
    if pred_max1>record_max1:
        xymax1 = pred_max1
    else:
        xymax1=record_max1

    pred_min2 =predictions2.min()
    pred_max2 = predictions2.max()
    record_min2 = records2.min()
    record_max2 = records2.max()
    if pred_min2<record_min2:
        xymin2 = pred_min2
    else:
        xymin2 = record_min2
    if pred_max2>record_max2:
        xymax2 = pred_max2
    else:
        xymax2=record_max2

    if xymin1<=xymin2:
        xymin=xymin1
    else:
        xymin=xymin2
    if xymax1>=xymax2:
        xymax=xymax1
    else:
        xymax=xymax2

    xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
    coeff1 = np.polyfit(predictions1, records1, 1)
    coeff2 = np.polyfit(predictions2, records2, 1)
    linear_fit1 = coeff1[0] * xx + coeff1[1]
    linear_fit2 = coeff2[0] * xx + coeff2[1]
    return xx,linear_fit1,linear_fit2,xymin,xymax


def compute_list_linear_fit(
    records_list,
    predictions_list,
    ):
    linear_list=[]
    xymin_list=[]
    xymax_list=[]
    for i in range(len(predictions_list)):
        records=records_list[i]
        predictions=predictions_list[i]
        pred_min =predictions.min()
        pred_max = predictions.max()
        record_min = records.min()
        record_max = records.max()
        if pred_min<record_min:
            xymin = pred_min
        else:
            xymin = record_min
        if pred_max>record_max:
            xymax = pred_max
        else:
            xymax=record_max
        xymin_list.append(xymin)
        xymax_list.append(xymax)

    xymin = min(xymin_list)
    xymax = max(xymax_list)
    xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
    
    for i in range(len(predictions_list)):
        records=records_list[i]
        predictions=predictions_list[i]
        coeff = np.polyfit(predictions, records, 1)
        linear_fit = coeff[0] * xx + coeff[1]
        linear_list.append(linear_fit)
    return xx,linear_list,xymin,xymax
    
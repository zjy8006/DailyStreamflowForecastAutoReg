from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.feature_selection import chi2
import tensorflow as tf
from IPython.display import clear_output
import json
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Current Path: {}'.format(root_path))

stations=['zjs','yx']
decomposers=['db2-1','db2-2','db2-3',
'db5-1','db5-2','db5-3',
'db10-1','db10-2','db10-3',
'db15-1','db15-2','db15-3',
'db20-1','db20-2','db20-3',
'db25-1','db25-2','db25-3',
'db30-1','db30-2','db30-3',
'db35-1','db35-2','db35-3',
'db40-1','db40-2','db40-3',
'db45-1','db45-2','db45-3',
'bior 3.3-1','bior 3.3-2','bior 3.3-3',
'coif3-1','coif3-2','coif3-3',
'haar-1','haar-2','haar-3','vmd','eemd']

selected_subsignals={}
for station in stations:
    selected_={}
    for decomposer in decomposers:
        if decomposer=='eemd' or decomposer=='vmd':
            dftrain = pd.read_csv(root_path+'/'+station+'_'+decomposer+'/data/'+decomposer.upper()+'_TRAIN.csv')
        else:
            dftrain = pd.read_csv(root_path+'/'+station+'_wd/data/'+decomposer+'/'+'WD_TRAIN.csv')

        y_train = dftrain.pop('ORIG')
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(dftrain,y_train)
        feat_importances = pd.Series(model.feature_importances_, index=dftrain.columns)
        print(feat_importances)
        median=feat_importances.median()
        mean=feat_importances.mean()
        selected = feat_importances[feat_importances>mean].index.to_list()
        selected_[decomposer]=selected
    selected_subsignals[station]=selected_
print(selected_subsignals)
with open(root_path+'/results_analyze/results/selected_subsignals.json', 'w') as fp:
    json.dump(selected_subsignals, fp)







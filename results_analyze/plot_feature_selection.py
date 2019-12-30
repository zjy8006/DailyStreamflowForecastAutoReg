from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
import tensorflow as tf
from IPython.display import clear_output
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Current Path: {}'.format(root_path))
import sys
sys.path.append(root_path+'\\tools\\')

dftrain = pd.read_csv(root_path+'\\zjs_vmd\\data\\VMD_TRAIN.csv')
y_train = dftrain.pop('ORIG')

model = GradientBoostingRegressor(n_estimators=100)
model.fit(dftrain,y_train)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=dftrain.columns)
median=feat_importances.median()
mean=feat_importances.mean()
feat_importances = pd.Series(feat_importances.values,
index=[r'$IMF_{1}$',r'$IMF_{2}$',r'$IMF_{3}$',r'$IMF_{4}$',r'$IMF_{5}$',r'$IMF_{6}$',r'$IMF_{7}$',r'$IMF_{8}$',r'$IMF_{9}$',r'$IMF_{10}$',r'$IMF_{11}$',])
print(feat_importances[feat_importances>mean].index.to_list())
plt.figure(figsize=(3.54,2.54))
plt.xlabel('Importance')
feat_importances.nlargest(11).plot(kind='barh',label='')
plt.vlines(x=median,ymin=-1,ymax=11,linestyles='--',colors='r',label='median')
plt.vlines(x=mean,ymin=-1,ymax=11,linestyles='--',colors='purple',label='mean')
plt.legend()
plt.tight_layout()
plt.savefig(root_path+'\\graph\\zjs_vmd_feature_selection.eps',format='EPS',dpi=2000)
plt.savefig(root_path+'\\graph\\zjs_vmd_feature_selection.tif',format='TIFF',dpi=1200)
plt.show()




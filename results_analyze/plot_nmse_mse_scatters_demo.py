import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib import cm
import matplotlib.pylab
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size'] = 6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap'] = 'plasma'
# plt.rcParams['axes.linewidth']=0.8
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
graph_path = root_path+'/graph/'

import sys
sys.path.append(root_path+'/tools/')
from fit_line import compute_linear_fit,compute_list_linear_fit
nmse = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one')
nmse_records = 
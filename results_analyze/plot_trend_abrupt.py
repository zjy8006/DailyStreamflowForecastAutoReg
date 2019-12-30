import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import math
from scipy.stats import norm
import sys
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
graphs_path=root_path+'\\graph\\'
sys.path.append(root_path+'/tools/')
from mann_kendall import plot_trend,plot_abrupt


Yangxian = pd.read_excel(root_path+'/time_series/YangXianRunoff1967-2014.xlsx', sheet_name='Year')['AnnualRunoff']
zhangjiashan = pd.read_excel(root_path+'/time_series/ZhangJiaShanRunoff1967-2017.xlsx', sheet_name='Year')['AnnualRunoff']
zhangjiashan = zhangjiashan[0:48]
print(zhangjiashan)

# plot_trend_abrupt(Yangxian, 0.95, 1967, 2014, 'Yangxian',parent_path+'/time_series/abrupt_change_YX.eps')
# plot_trend_abrupt(Hanzhong, 0.95, 1972, 2014, 'Hanzhong',parent_path+'/time_series/abrupt_change_HZ.eps')
# plot_trend_abrupt(zhangjiashan, 0.95, 1970, 2017, 'Zhangjiashan',parent_path+'/time_series/abrupt_change_ZJS.eps')

fig = plt.figure(figsize=(7.48,4.5))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
plot_trend(Yangxian,ax=ax1,start=1967,end=2014,series_name='Yangxian',fig_id='a')
plot_trend(zhangjiashan,ax=ax1,start=1967,end=2014,series_name='Zhangjiashan',fig_id='a')
ax1.set_xlim(1967,2014)

ax3=fig.add_subplot(2,2,3)
plot_abrupt(Yangxian,ax=ax3,start=1967,end=2014,series_name='Yangxian',fig_id='b')
ax4=fig.add_subplot(2,2,4)
plot_abrupt(zhangjiashan,ax=ax4,start=1967,end=2014,series_name='Zhangjiashan',fig_id='c')
plt.subplots_adjust(left=0.08, bottom=0.14, right=0.98, top=0.98, hspace=0.35, wspace=0.2)
plt.savefig(graphs_path+'trand_abrupt.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'trand_abrupt.tif',format='TIFF',dpi=1200)
plt.show()
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
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir)) # for runing code in CMD
graphs_path=root_path+'\\graph\\'

print('Root path:{}'.format(root_path))
print('Graphs path:{}'.format(graphs_path))


def get_Sk(timeSeries):
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()
    print('Time series is list:{}'.format(isinstance(timeSeries, list)))
    N = len(timeSeries)
    results = []
    for n in range(1, N+1):
        sumss = []
        for i in range(n-1):
            sums = []
            for j in range(i+1, n):
                val = timeSeries[j]-timeSeries[i]
                if val > 0:
                    sums.append(1)
                elif val < 0:
                    sums.append(-1)
                else:
                    sums.append(0)
            sumss.append(sum(sums))
        results.append(sum(sumss))
    return results


def get_S(timeSeries):
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()
    print('Time series is list:{}'.format(isinstance(timeSeries, list)))
    N = len(timeSeries)
    sumss = []
    for i in range(N-1):
        sums = []
        for j in range(i+1, N):
            val = timeSeries[j]-timeSeries[i]
            if val > 0:
                sums.append(1)
            elif val < 0:
                sums.append(-1)
            else:
                sums.append(0)
        sumss.append(sum(sums))
    return sum(sumss)


def get_Z(timeSeries):
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()
    print('Time series is list:{}'.format(isinstance(timeSeries, list)))
    N = len(timeSeries)
    S = get_S(timeSeries)
    var = N*(N-1)*(2*N+5)/18.0
    if S > 0:
        Z = (S-1)/math.sqrt(var)
    elif S < 0:
        Z = (S+1)/math.sqrt(var)
    return Z


def get_Z_alpha(confidence):
    var = 1-(1-confidence)/2
    return norm._ppf(var)


def get_trend(timeSeries, confidence):
    S = get_S(timeSeries)
    Z = get_Z(timeSeries)
    Z_alpha = get_Z_alpha(confidence)
    print('S={}'.format(S))
    print('Z={}'.format(Z))
    print('Z_α={}'.format(Z_alpha))
    if abs(Z) >= Z_alpha and Z > 0:
        trend = 'Up trend'
    elif abs(Z) >= Z_alpha and Z < 0:
        trend = 'Down trend'
    elif abs(Z) < Z_alpha:
        trend = 'No trend'
    return trend


def get_UFK(timeSeries):
    Sk = get_Sk(timeSeries)
    E_Sk = sum(Sk)/len(Sk)
    var_Sk = np.var(Sk)
    UFK = []
    for i in range(len(timeSeries)):
        UFK.append((Sk[i]-E_Sk)/math.sqrt(var_Sk))
    return UFK


def get_UBK(timeSeries):
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()
    print('Time series is list:{}'.format(isinstance(timeSeries, list)))
    reverse_series = list(reversed(timeSeries))
    Sk = get_Sk(reverse_series)
    E_Sk = sum(Sk)/len(Sk)
    var_Sk = np.var(Sk)
    UBK = []
    for i in range(len(reverse_series)):
        UBK.append((Sk[i]-E_Sk)/math.sqrt(var_Sk))
    return UBK


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0]
             [1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('Lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def plot_trend_abrupt(timeSeries, confidence, start_year, end_year, series_name, fig_savepath):
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()
    print('Time series is list:{}'.format(isinstance(timeSeries, list)))
    UFK = get_UFK(timeSeries)
    UBK = get_UBK(timeSeries)
    years = list(range(start_year, end_year+1))
    X = []
    Z_alpha = get_Z_alpha(confidence)
    Z_up = Z_alpha*np.ones(len(years))
    Z_low = -Z_alpha*np.ones(len(years))
    Y = []
    for i in range(len(UFK)-1):
        line1 = np.zeros(shape=(2, 2))
        line2 = np.zeros(shape=(2, 2))
        line1[0][0] = years[i]
        line1[1][0] = years[i+1]
        line1[0][1] = UFK[i]
        line1[1][1] = UFK[i+1]
        line2[0][0] = years[i]
        line2[1][0] = years[i+1]
        line2[0][1] = UBK[i]
        line2[1][1] = UBK[i+1]
        if (line1[0][1]-line2[0][1] > 0 and line1[1][1]-line2[1][1] < 0) or (line1[0][1]-line2[0][1] < 0 and line1[1][1]-line2[1][1] > 0):
            x, y = line_intersection(line1, line2)
            print('Potential abrupt change point = {}'.format(x))
            print('Potential abrupt change vales = {}'.format(y))
            if y > -Z_alpha and y < Z_alpha:
                X.append(round(x, 2))
                Y.append(round(y, 2))
    print('List of abrupt change points:{}'.format(X))
    print('List of abrupt change values:{}'.format(Y))
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.xlim(start_year, end_year)
    float_years = np.arange(
        start=start_year, stop=end_year+1, step=1, dtype=np.float)
    print(len(float_years))
    print(len(timeSeries))
    coeff = np.polyfit(float_years, timeSeries, 1)
    linear_trend = coeff[0] * float_years + coeff[1]
    # plt.text(start_year+(end_year-start_year)/3.2,max(timeSeries)+10,''+str(X))
    plt.xlabel('Time (Year)\n(a)', )
    plt.ylabel('Records (' + r'$10^8m^3$' + '/s)', )
    plt.plot(years, timeSeries, color='green', label=series_name)
    plt.plot(years, linear_trend, '--',
             color='black', label='Linear trend')
    plt.legend(
        # loc='upper left',
        loc=0,
        # loc=3,
        # bbox_to_anchor=(0.22,1.005, 1,0.1005),
        # ncol=3,
        shadow=False,
        frameon=False,
        )
    plt.subplot(2, 1, 2)
    plt.text(start_year+(end_year-start_year)/3.2,
             1.6, 'Abrput change points: '+str(X))
    plt.xlim(start_year, end_year)
    plt.xlabel('Time (Year)\n(b)', )
    plt.ylabel('UF~UB', )
    plt.plot(years, UFK, color='b', label='UF')
    plt.plot(years, UBK, color='fuchsia', label='UB')
    plt.plot(years, Z_up, '--', color='r',
             label=str(confidence*100)+'% confidence level')
    plt.plot(years, Z_low, '--', color='r', label='')
    plt.legend(
        # loc='upper left',
        # loc=0,
        loc=3,
        bbox_to_anchor=(0.22, 1.00, 1, 0.100),
        ncol=3,
        shadow=False,
        frameon=False,
        )
    plt.subplots_adjust(left=0.09, bottom=0.1, right=0.98,
                        top=0.98, hspace=0.5, wspace=0.2)
    plt.savefig(fig_savepath, format='EPS', transparent=False, dpi=1200)
    plt.show()


def plot_abrupt(timeSeries, **kwargs):
    """ Plot abrupt of a time series using Mann-Kendall method 
    Parameters:
    -----------------------------------------------------------
    * timeSeries: list of float data

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `start` [float, optional]:
        The start of the time series, if known.

    * `end` [float, optional]:
        The end of the time series, if known.

    * `confidence` [float, optional]:
        The confidence for validating the abrupt, if known.

    * `series_name` [string, optional]:
        The series name, if known.

    * `fig_id` [string, optional]:
        The index of the ax, if known.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()

    ax = kwargs.get('ax', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    confidence = kwargs.get('confidence', None)
    series_name = kwargs.get('series_name', None)
    fig_id = kwargs.get('fig_id', None)

    if start is None and end is None:
        start = 1
        end = len(timeSeries)
    elif (start is None and end is not None) or (start is not None and end is None):
        raise Exception('Both the start and the end of a time series should be given')

    years = list(range(start, end+1))

    if series_name is None:
        series_name = 'Time series'

    if ax is None:
        ax = plt.gca()

    if confidence is None:
        confidence = 0.95

    UFK = get_UFK(timeSeries)
    UBK = get_UBK(timeSeries)
    years = list(range(start, end+1))
    X = []
    Z_alpha = get_Z_alpha(confidence)
    Z_up = Z_alpha*np.ones(len(years))
    Z_low = -Z_alpha*np.ones(len(years))
    Y = []
    for i in range(len(UFK)-1):
        line1 = np.zeros(shape=(2, 2))
        line2 = np.zeros(shape=(2, 2))
        line1[0][0] = years[i]
        line1[1][0] = years[i+1]
        line1[0][1] = UFK[i]
        line1[1][1] = UFK[i+1]
        line2[0][0] = years[i]
        line2[1][0] = years[i+1]
        line2[0][1] = UBK[i]
        line2[1][1] = UBK[i+1]
        if (line1[0][1]-line2[0][1] > 0 and line1[1][1]-line2[1][1] < 0) or (line1[0][1]-line2[0][1] < 0 and line1[1][1]-line2[1][1] > 0):
            x, y = line_intersection(line1, line2)
            print('Potential abrupt change point = {}'.format(x))
            print('Potential abrupt change vales = {}'.format(y))
            if y > -Z_alpha and y < Z_alpha:
                X.append(int(x))
                Y.append(round(y, 2))
    print('List of abrupt change points:{}'.format(X))
    print('List of abrupt change values:{}'.format(Y))

    # ax.text(start+(end-start)/10,2.2, 'Abrput change year of '+series_name+':'+str(X))
    plt.xticks(np.arange(start=start,stop=end,step=5),rotation=45)
    ax.set_xlim(left=start, right=end)
    ax.set_ylim(bottom=-2.5,top=4)
    if fig_id is None:
        ax.set_xlabel('Time (Year)', )
    else:
        ax.set_xlabel('Time (Year)\n('+fig_id+')', )
    ax.set_ylabel('UF~UB', )
    ax.plot(years, UFK, color='b', label='UF')
    ax.plot(years, UBK, color='fuchsia', label='UB')
    ax.plot(years, Z_up, '--', color='r',label=str(confidence*100)+'% confidence level')
    ax.plot(years, Z_low, '--', color='r', label='')
    ax.legend(
        loc=9,
        # bbox_to_anchor=(0.1, 1.00, 1, 0.100),
        # ncol=3,
        shadow=False,
        frameon=False,
        )


def plot_trend(timeSeries, **kwargs):
    """ Plot trand of a time series 
    Parameters:
    --------------------------------------------------------------------
    * timeSeries: list of float data

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `start` [float, optional]:
        The start of the time series, if known.

    * `end` [float, optional]:
        The end of the time series, if known.

    * `series_name` [string, optional]:
        The series name, if known.

    * `fig_id` [string, optional]:
        The index of the ax, if known.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    if isinstance(timeSeries, pd.DataFrame):
        timeSeries = (timeSeries.values).flatten()
    elif isinstance(timeSeries, pd.Series):
        timeSeries = timeSeries.tolist()
    elif isinstance(timeSeries, np.ndarray):
        timeSeries = timeSeries.flatten()

    ax = kwargs.get('ax', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    series_name = kwargs.get('series_name', None)
    fig_id = kwargs.get('fig_id', None)

    if start is None and end is None:
        start = 1
        end = len(timeSeries)
    elif (start is None and end is not None) or (start is not None and end is None):
        raise Exception('Both the start and end of a time series should be given')

    years = list(range(start, end+1))

    if series_name is None:
        series_name = 'Time series'

    if ax is None:
        ax = plt.gca()

    ax.set_xlim(start, end)
    float_years = np.arange(start=start, stop=end+1, step=1, dtype=np.float)
    print(len(float_years))
    print(len(timeSeries))
    coeff = np.polyfit(float_years, timeSeries, 1)
    linear_trend = coeff[0] * float_years + coeff[1]
    # plt.text(start_year+(end_year-start_year)/3.2,max(timeSeries)+10,''+str(X))
    if fig_id is None:
        ax.set_xlabel('Time (Year)', )
    else:
        ax.set_xlabel('Time (Year)\n('+fig_id+')', )
    ax.set_ylabel('Records (' + r'$10^8m^3$' + '/s)', )
    ax.plot(years, timeSeries, label=series_name)
    ax.plot(years, linear_trend, '--', label='Linear trend')
    ax.legend(
        loc=0,
        # bbox_to_anchor=(0.22,1.005, 1,0.1005),
        # ncol=3,
        shadow=False,
        frameon=False,
        )
    return ax


if __name__ == '__main__':
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

from math import *
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#### 参数设置
init_asset = 1  # 初识资产的净值
start_time = '2013-01-01'  # 起始时间，为起始季度的第一个月,输入格式为YYYY-MM-DD
end_time = '2018-10-01'  # 结束时间，为结束季度的第一个月，输入格式为YYYY-MM-DD
path_macro = '/Users/maxiaohang/宏观数据(4).xlsx'  # 宏观数据的存放路径
threshold = 0.65  # 判定为有效指标的最小胜率
####

def get_file_name(path, filetype):#读取文件的名字
    file_name = []
    os.chdir(path)
    for root, dir, files in os.walk(os.getcwd()):
        for file in files:
            if os.path.splitext(file)[1] in filetype:
                # print(os.path.splitext(file)[1])
                file_name.append(file)
    return file_name

def consistency_judge(x):  # 判断经济指标与GDP的变动一致性，1为相同，0为不同.
    if x:
        return 1
    else:
        return 0

def signal_compute(x):
    if x:
        return 1
    else:
        return -1

def macro_signal(month, eco_data, delay_n, delay_k, type):
    """
    利用宏观经济指标计算信号
    :param month: month为当月最后一天
    :param eco_data: 宏观经济数据，其索引为每月的最后一天，类型为DataFrame
    :param delay_n: 经济数据滞后GDP数据的期数，单位为月
    :param delay_k: 计算指标信号
    :param type: 0表示相对于k个月前的值计算，1表示相对于k个月均线计算
    :return: +1表示上升，-1表示下降
    """
    month = pd.to_datetime(month) - relativedelta(months=delay_n)
    d1 = eco_data[(eco_data.index >= month - relativedelta(months=delay_k)) &
                                 (eco_data.index < month + relativedelta(months=1))].mean()  # k个月的均值
    d2 = eco_data[(eco_data.index >= month - relativedelta(months=delay_k))&
                  (eco_data.index < month - relativedelta(months=delay_k-1))][0]  # k个月前的值
    month_value = eco_data[(eco_data.index >= month)&
                  (eco_data.index < month + relativedelta(months=1))][0]   # 获取季度开始前n个月的数据
    return signal_compute(month_value > d1)*type + signal_compute(month_value > d2)*(1-type)



if __name__ == '__main__':
    reset_time = str(pd.to_datetime(start_time) - relativedelta(months=3))[0:7]  # 计算GDP变化方向起始时间要往前一个季度
    start_time_1 = str(pd.to_datetime(start_time) - relativedelta(months=1))[0:7]
    end_time_1 = str(pd.to_datetime(end_time) + relativedelta(months=3))[0:7]
    delay_n = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2])  # 新增指标：工业增加值:当月同比公布时间为次月中
    # delay_n = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1])  # 新增指标：工业增加值:当月同比公布时间为次月中
    macro_data = pd.read_excel(path_macro)
    macro_data = macro_data.iloc[2:]
    macro_data.index = macro_data.iloc[:, 0]
    GDP = macro_data[macro_data.columns[1]]
    GDP = GDP.rename(index=macro_data[macro_data.columns[0]])
    GDP = pd.DataFrame(GDP[(GDP.index > reset_time) &
                           (GDP.index < end_time_1)]).dropna()
    GDP_1 = GDP.iloc[1:]  # GDP的第一个数据是为了计算趋势加入的，现在要去掉
    GDP_trend = GDP.diff().iloc[1:] > 0
    GDP_signal = [signal_compute(i) for i in GDP_trend[GDP_trend.columns[0]]]  # GDP的变动信号值

    date_index = pd.to_datetime(macro_data[macro_data.columns[0]])
    date_index = date_index[(date_index > pd.to_datetime(start_time))&
                            (date_index < pd.to_datetime(end_time) + relativedelta(months=1))]
    date_index.index = date_index
    date_index = date_index.resample('3M').apply(lambda x : x[-1])  # 获取回测期间每个季度月初的日期
    effective_indicator = []  # 获取指标的有效信号的胜率
    effective_indicator_name = []  # 获取具有有效信号的指标的名称
    wining_rate = []
    GDP_inidcate_signal = np.zeros([1, len(date_index)])  # GDP季度前瞻信号
    for i in range(19):
        eco_data = macro_data[macro_data.columns[3 + i]]
        n = delay_n[0 + i]
        signal_id = [(n, k, j) for k in range(1, 7) for j in [0, 1]]
        eco_wiming_rate = []
        eco_effective_signal = np.zeros([1, len(date_index)])  # 有效信号的平均值
        effective_num = 0  # 指标的有效信号的个数
        for s in signal_id:
            eco_signal = [macro_signal(month, eco_data, s[0], s[1], s[2]) for month in date_index]
            rate = np.sum([consistency_judge(x)
                                           for x in np.array(eco_signal) == np.array(GDP_signal)]) / len(GDP_signal)
            eco_wiming_rate.append(rate)
            if rate > threshold:
                effective_num = effective_num + 1
                eco_effective_signal = eco_effective_signal + eco_signal
        df = pd.DataFrame({'指标({})'.format(macro_data.columns[3 + i]): signal_id, '指标胜率': eco_wiming_rate})
        wining_rate.append(df)
        if effective_num > 0:
            effective_indicator.append(df[df[df.columns[1]] > threshold])
            effective_indicator_name.append(i+3)
            eco_effective_signal = eco_effective_signal/effective_num
        GDP_inidcate_signal = GDP_inidcate_signal + eco_effective_signal
    effective_indicator_name = [macro_data.columns[i] for i in effective_indicator_name]






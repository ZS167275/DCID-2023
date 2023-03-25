# coding:utf-8
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


def calc_max_coutinut_times(a, b=1):
    t = 0
    w = 1
    for k, v in enumerate(a):
        if k > 0:
            if v == b and a[k - 1] == b:
                t += 1
                if w < t:
                    w = t
            else:
                t = 1
    return w


def findMaxAverage(nums, k):
    average = []  # 平均数的列表，最后取最大即可
    sum_, start = 0, 0  # 末尾下划线避免与关键字冲突
    for end in range(len(nums)):
        sum_ += nums[end]
        if end >= k - 1:  # 达到窗口大小
            average.append(sum_ / k)  # 计算平均值
            sum_ -= nums[start]  # 减去窗口外的元素
            start += 1  # 滑动窗口一位
    return max(average)


def findMinAverage(nums, k):
    average = []  # 平均数的列表，最后取最大即可
    sum_, start = 0, 0  # 末尾下划线避免与关键字冲突
    for end in range(len(nums)):
        sum_ += nums[end]
        if end >= k - 1:  # 达到窗口大小
            average.append(sum_ / k)  # 计算平均值
            sum_ -= nums[start]  # 减去窗口外的元素
            start += 1  # 滑动窗口一位
    return min(average)


path = '../data/'

for mm in [3, 4, 5]:

    static_info = pd.read_csv(path + '账户静态信息.csv')
    time_info = pd.read_csv(path + '账户交易信息.csv')
    train_label = pd.read_csv(path + '训练集标签.csv')
    test_no_label = pd.read_csv(path + 'test_dataset.csv')

    print(time_info.shape)

    static_info.columns = ['zhdh', 'khrq', 'khjgdh', 'xb', 'age']

    static_info['khrq_y'] = static_info['khrq'].apply(lambda x: str(x).split('-')[0])
    static_info['khrq_m'] = static_info['khrq'].apply(lambda x: str(x).split('-')[1])
    static_info['khrq_d'] = static_info['khrq'].apply(lambda x: str(x).split('-')[2])

    static_info['khrq_y'] = static_info['khrq_y'].astype(int)
    static_info['khrq_m'] = static_info['khrq_m'].astype(int)
    static_info['khrq_d'] = static_info['khrq_d'].astype(int)

    static_info['khrq'] = static_info['khrq'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d')))

    # 截止到 2020年已经开户的年份
    static_info['khrq_y'] = static_info['khrq_y'].max() - static_info['khrq_y'] + 1
    # 初次开户的岁数
    static_info['age_khrq_y'] = static_info['age'] - static_info['khrq_y']


    # 自然数编码
    def label_encode(series):
        unique = list(series.unique())
        return series.map(dict(zip(
            unique, range(series.nunique())
        )))


    for col in ['khjgdh']:
        static_info[col] = label_encode(static_info[col])

    # 静态类别特征
    train = pd.merge(train_label, static_info, on=['zhdh'], how='left', copy=False)
    test = pd.merge(test_no_label, static_info, on=['zhdh'], how='left', copy=False)

    time_info['duichong_1'] = time_info['jyje'] < 0
    time_info['duichong_2'] = time_info.groupby(['zhdh'])['duichong_1'].shift(-1)
    time_info['duichong_2'] = time_info['duichong_2'].fillna(False)

    time_info['duichong'] = time_info['duichong_1'] + time_info['duichong_2']
    print(time_info.shape)
    time_info = time_info[time_info['duichong'] == 0].reset_index(drop=True)
    print(time_info.shape)

    time_info['jyje'] = np.log(time_info['jyje'])
    time_info['zhye'] = np.log(time_info['zhye'])

    # 扩展时间
    prefix = 'jyrq'
    time_info['t_jyrq'] = pd.to_datetime(time_info['jyrq'], format='%Y-%m-%d')
    time_info[prefix + 'month'] = time_info['t_jyrq'].dt.month

    time_info[prefix + 'day'] = time_info['t_jyrq'].dt.day
    time_info[prefix + 'weekofyear'] = time_info['t_jyrq'].dt.weekofyear
    time_info[prefix + 'dayofyear'] = time_info['t_jyrq'].dt.dayofyear
    time_info[prefix + 'dayofweek'] = time_info['t_jyrq'].dt.dayofweek
    time_info[prefix + 'is_wknd'] = time_info['t_jyrq'].dt.dayofweek // 6
    time_info[prefix + 'is_month_start'] = time_info['t_jyrq'].dt.is_month_start.astype(int)
    time_info[prefix + 'is_month_end'] = time_info['t_jyrq'].dt.is_month_end.astype(int)
    del time_info['t_jyrq']

    time_info['hour'] = time_info['jysj'].apply(lambda x: str(x).split(':')[0])
    time_info['minute'] = time_info['jysj'].apply(lambda x: str(x).split(':')[1])
    time_info['second'] = time_info['jysj'].apply(lambda x: str(x).split(':')[2])

    # 流水单的时间转化为时间戳
    time_info['jyrq_jysj'] = time_info['jyrq'] + ' ' + time_info['jysj']
    time_info['jyrq_jysj'] = time_info['jyrq_jysj'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))

    time_info = time_info.sort_values(['jyrq_jysj'])
    time_info['jyrq_jysj_down'] = time_info.groupby(['zhdh'])['jyrq_jysj'].shift(1)
    time_info['jyrq_jysj_down'] = time_info['jyrq_jysj'] - time_info['jyrq_jysj_down']

    time_info['jyrq_jysj_leak'] = time_info.groupby(['zhdh'])['jyrq_jysj'].shift(-1)
    time_info['jyrq_jysj_leak'] = time_info['jyrq_jysj_leak'] - time_info['jyrq_jysj']

    time_info['jyje_down'] = time_info.groupby(['zhdh'])['jyje'].shift(1)
    time_info['jyje_down_jyje'] = time_info['jyje_down'] / time_info['jyje']

    time_info['jdbj_down'] = time_info.groupby(['zhdh'])['jdbj'].shift(1)
    time_info['jdbj_down'] = time_info['jdbj'] - time_info['jdbj_down']

    time_info['fuhao_tmp'] = time_info['jdbj'].apply(lambda x: -1 if x == 0 else 1)
    time_info['fuhao_tmp'] = time_info['fuhao_tmp'] * time_info['jyje']

    time_info['nx_zhdh'] = 'zh_' + time_info['zhdh']
    time_info['nx_dfzh'] = 'df_' + time_info['dfzh']
    time_info['nx_weight'] = time_info['jdbj']

    time_info = time_info[time_info['jyrqmonth'] == mm].reset_index(drop=True)

    for i in [2, 3, 5, 7, 10, 20, 30]:
        tmp = time_info.groupby(['zhdh'])['fuhao_tmp'].apply(
            lambda x: findMaxAverage(list(x), i if len(list(x)) >= i else len(list(x)))).reset_index()
        tmp.columns = ['zhdh', 'windows_max_{}_jyje'.format(i)]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)

    for i in [2, 3, 5, 7, 10, 20, 30]:
        tmp = time_info.groupby(['zhdh'])['fuhao_tmp'].apply(
            lambda x: findMinAverage(list(x), i if len(list(x)) >= i else len(list(x)))).reset_index()
        tmp.columns = ['zhdh', 'windows_min_{}_jyje'.format(i)]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)

    ff = []
    for t in [1, 10, 30, 60]:
        ff.append('jyrq_jysj_down_1_{}'.format(t))
        time_info[time_info['jdbj'] == 1]['jyrq_jysj_down_1_{}'.format(t)] = 0
        time_info.loc[time_info['jyrq_jysj_down'] <= t, 'jyrq_jysj_down_1_{}'.format(t)] = 1

    for t in [1, 10, 30, 60]:
        ff.append('jyrq_jysj_down_0_{}'.format(t))
        time_info[time_info['jdbj'] == 0]['jyrq_jysj_down_0_{}'.format(t)] = 0
        time_info.loc[time_info['jyrq_jysj_down'] <= t, 'jyrq_jysj_down_0_{}'.format(t)] = 1

    # 流水全局统计特征
    for col in ['dfzh'] + ff:
        tmp = time_info.groupby(['zhdh']).agg({col: ['count', 'nunique']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 每天的转账频次
    day_zhdh = time_info.groupby(['zhdh', 'jyrq']).agg({'dfzh': ['count', 'nunique']}).reset_index()
    day_zhdh.columns = [x[0] if x[1] == '' else 'day_' + x[0] + '_' + x[1] for x in day_zhdh.columns]
    day_zhdh['day_dfzh_nunique_count'] = day_zhdh['day_dfzh_nunique'] / day_zhdh['day_dfzh_count']

    for col in ['day_dfzh_nunique', 'day_dfzh_count', 'day_dfzh_nunique_count']:
        tmp = day_zhdh.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 月的转账频次
    time_info['jyrq_month'] = time_info['jyrq'].apply(lambda x: '-'.join(str(x).split('-')[:-1]))
    month_zhdh = time_info.groupby(['zhdh', 'jyrq_month']).agg({'dfzh': ['count', 'nunique']}).reset_index()
    month_zhdh.columns = [x[0] if x[1] == '' else 'month_' + x[0] + '_' + x[1] for x in month_zhdh.columns]
    month_zhdh['month_dfzh_nunique_count'] = month_zhdh['month_dfzh_nunique'] / month_zhdh['month_dfzh_count']
    del time_info['jyrq_month']

    for col in ['month_dfzh_nunique', 'month_dfzh_count', 'month_dfzh_nunique_count']:
        tmp = month_zhdh.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 每天每小时的转账频次
    time_info['jyrq_hour'] = time_info['jyrq'] + ' ' + time_info['hour']
    tmp_info = time_info.groupby(['zhdh', 'jyrq_hour']).agg({'dfzh': ['count', 'nunique']}).reset_index()
    tmp_info.columns = [x[0] if x[1] == '' else 'hour_' + x[0] + '_' + x[1] for x in tmp_info.columns]
    tmp_info['hour_dfzh_nunique_count'] = tmp_info['hour_dfzh_nunique'] / tmp_info['hour_dfzh_count']
    del time_info['jyrq_hour']

    for col in ['hour_dfzh_nunique', 'hour_dfzh_count', 'hour_dfzh_nunique_count']:
        tmp = tmp_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 每天每分钟时的转账频次
    time_info['jyrq_hour'] = time_info['jyrq'] + ' ' + time_info['hour'] + ' ' + time_info['minute']
    tmp_info = time_info.groupby(['zhdh', 'jyrq_hour']).agg({'dfzh': ['count', 'nunique']}).reset_index()
    tmp_info.columns = [x[0] if x[1] == '' else 'minute_' + x[0] + '_' + x[1] for x in tmp_info.columns]
    tmp_info['minute_dfzh_nunique_count'] = tmp_info['minute_dfzh_nunique'] / tmp_info['minute_dfzh_count']
    del time_info['jyrq_hour']

    for col in ['minute_dfzh_nunique', 'minute_dfzh_count', 'minute_dfzh_nunique_count']:
        tmp = tmp_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 分别统计收入和支出
    for i in [0, 1]:
        for col in ['dfzh', 'dfhh', 'jyqd', 'zydh']:
            tmp = time_info[time_info['jdbj'] == i].groupby(['zhdh']).agg({col: ['count', 'nunique']}).reset_index()
            tmp.columns = [x[0] if x[1] == '' else 'jdbj_{}_'.format(i) + x[0] + '_' + x[1] for x in tmp.columns]
            train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
            test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
            del tmp

    for i in [0, 1]:
        # time_info['jyrq_month'] = time_info['jyrq'].apply(lambda x: '-'.join(str(x).split('-')[:-1]))
        month_zhdh = time_info[time_info['jdbj'] == i].groupby(['zhdh', 'jyrq']).agg(
            {'dfzh': ['count', 'nunique']}).reset_index()
        month_zhdh.columns = [x[0] if x[1] == '' else 'day_' + x[0] + '_' + x[1] for x in month_zhdh.columns]
        month_zhdh['day_dfzh_nunique_count'] = month_zhdh['day_dfzh_nunique'] / month_zhdh['day_dfzh_count']
        # del time_info['jyrq_month']
        for col in ['day_dfzh_nunique', 'day_dfzh_count', 'day_dfzh_nunique_count']:
            tmp = month_zhdh.groupby(['zhdh']).agg({col: ['mean', 'var', 'count', 'nunique']}).reset_index()
            tmp.columns = [x[0] if x[1] == '' else 'jdbj_{}_'.format(i) + x[0] + '_' + x[1] for x in tmp.columns]
            train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
            test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
            del tmp

    # 分别计算 支出 收入 月统计特征
    for i in [0, 1]:
        time_info['jyrq_month'] = time_info['jyrq'].apply(lambda x: '-'.join(str(x).split('-')[:-1]))
        month_zhdh = time_info[time_info['jdbj'] == i].groupby(['zhdh', 'jyrq_month']).agg(
            {'dfzh': ['count', 'nunique']}).reset_index()
        month_zhdh.columns = [x[0] if x[1] == '' else 'month_' + x[0] + '_' + x[1] for x in month_zhdh.columns]
        month_zhdh['month_dfzh_nunique_count'] = month_zhdh['month_dfzh_nunique'] / month_zhdh['month_dfzh_count']
        del time_info['jyrq_month']
        for col in ['month_dfzh_nunique', 'month_dfzh_count', 'month_dfzh_nunique_count']:
            tmp = month_zhdh.groupby(['zhdh']).agg({col: ['mean', 'var', 'count', 'nunique']}).reset_index()
            tmp.columns = [x[0] if x[1] == '' else 'jdbj_{}_'.format(i) + x[0] + '_' + x[1] for x in tmp.columns]
            train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
            test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
            del tmp

    # 分别计算 支出 收入 月统计特征
    for i in [0, 1]:
        time_info['jyrq_month'] = time_info['jyrq'] + time_info['jysj'].apply(
            lambda x: ':'.join(str(x).split(':')[:-1]))
        month_zhdh = time_info[time_info['jdbj'] == i].groupby(['zhdh', 'jyrq_month']).agg(
            {'dfzh': ['count', 'nunique']}).reset_index()
        month_zhdh.columns = [x[0] if x[1] == '' else 'minute_' + x[0] + '_' + x[1] for x in month_zhdh.columns]
        month_zhdh['minute_dfzh_nunique_count'] = month_zhdh['minute_dfzh_nunique'] / month_zhdh['minute_dfzh_count']
        del time_info['jyrq_month']
        for col in ['minute_dfzh_nunique', 'minute_dfzh_count', 'minute_dfzh_nunique_count']:
            tmp = month_zhdh.groupby(['zhdh']).agg({col: ['mean', 'var', 'count', 'nunique']}).reset_index()
            tmp.columns = [x[0] if x[1] == '' else 'jdbj_{}_'.format(i) + x[0] + '_' + x[1] for x in tmp.columns]
            train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
            test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
            del tmp

    for i in [0, 1]:
        for col in ['jyje', 'zhye', 'dfmccd', 'jyrq_jysj']:
            tmp = time_info[time_info['jdbj'] == i].groupby(['zhdh']).agg(
                {col: ['sum', 'mean', 'max', 'min', 'std', np.ptp]}).reset_index()
            tmp.columns = [x[0] if x[1] == '' else 'jdbj_{}_'.format(i) + x[0] + '_' + x[1] for x in tmp.columns]
            train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
            test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
            del tmp

    # 时间的特征处理
    for col in ['jyrq_jysj']:
        tmp = time_info.groupby(['zhdh']).agg({col: ['max', 'min', np.ptp, 'last']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 交以金额离散化
    time_info['jyje_count'] = time_info.groupby(['jyje'])['jyje'].transform('count')
    time_info['dfzh_count'] = time_info.groupby(['dfzh'])['dfzh'].transform('count')
    time_info['jdbj_count'] = time_info.groupby(['jdbj'])['jdbj'].transform('count')
    time_info['dfhh_count'] = time_info.groupby(['dfhh'])['dfhh'].transform('count')
    time_info['jyrq_count'] = time_info.groupby(['jyrq'])['jyrq'].transform('count')
    time_info['jyqd_count'] = time_info.groupby(['jyqd'])['jyqd'].transform('count')
    time_info['zydh_count'] = time_info.groupby(['zydh'])['zydh'].transform('count')
    time_info['dfmccd_count'] = time_info.groupby(['dfmccd'])['dfmccd'].transform('count')
    time_info['zhdh_dfzh_count'] = time_info.groupby(['zhdh', 'dfzh'])['zhdh'].transform('count')
    time_info['zhdh_jdbj_count'] = time_info.groupby(['zhdh', 'jdbj'])['zhdh'].transform('count')
    time_info['zhdh_dfhh_count'] = time_info.groupby(['zhdh', 'dfhh'])['zhdh'].transform('count')
    time_info['zhdh_jyrq_count'] = time_info.groupby(['zhdh', 'jyrq'])['zhdh'].transform('count')
    time_info['zhdh_jyqd_count'] = time_info.groupby(['zhdh', 'jyqd'])['zhdh'].transform('count')
    time_info['zhdh_zydh_count'] = time_info.groupby(['zhdh', 'zydh'])['zhdh'].transform('count')
    time_info['zhdh_dfmccd_count'] = time_info.groupby(['zhdh', 'dfmccd'])['zhdh'].transform('count')

    # 流水全局连续特征
    for col in [
        'jdbj',
        'jyrq_jysj_down',
        # 'jyrq_jysj_leak',
        'zhye', 'dfzh_count', 'jyje_count', 'jdbj_count', 'dfhh_count', 'jyrq_count',
        'jyqd_count', 'zydh_count', 'dfmccd_count',
        'zhdh_dfzh_count', 'zhdh_jdbj_count', 'zhdh_dfhh_count', 'zhdh_jyrq_count', 'zhdh_jyqd_count',
        'zhdh_zydh_count',
        'zhdh_dfmccd_count',
        'jyje',
        'jyrqmonth', 'jyrqday', 'jyrqweekofyear', 'jyrqdayofyear',
        'jyrqdayofweek', 'jyrqis_wknd', 'jyrqis_month_start',
        'jyrqis_month_end',
        'dfmccd',
    ]:
        tmp = time_info[time_info['jdbj'] == 1].groupby(['zhdh']).agg(
            {col: ['mean', 'min', 'max', np.ptp, 'last']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_1_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

        tmp = time_info[time_info['jdbj'] == 0].groupby(['zhdh']).agg(
            {col: ['mean', 'min', 'max', np.ptp, 'last']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_0_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

        tmp = time_info.groupby(['zhdh']).agg({col: ['mean', 'min', 'max', np.ptp, 'last']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else 'global_' + x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # dfzh
    # dfzh_zhdh_count dfzh_zhdh_nunique
    dfzh_feature = time_info.groupby(['dfzh']).agg({'zhdh': ['count', 'nunique']}).reset_index()
    dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]

    time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)
    time_info['dfzh_zhdh_nunique_count'] = time_info['dfzh_zhdh_nunique'] / time_info['dfzh_zhdh_count']

    for col in ['dfzh_zhdh_count', 'dfzh_zhdh_nunique', 'dfzh_zhdh_nunique_count']:
        tmp = time_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # dfzh_1_jyje_sum
    dfzh_feature = time_info[time_info['jdbj'] == 1].groupby(['dfzh']).agg({'jyje': ['sum']}).reset_index()
    dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_1_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]
    time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)

    dfzh_feature = time_info[time_info['jdbj'] == 0].groupby(['dfzh']).agg({'jyje': ['sum']}).reset_index()
    dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_0_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]
    time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)

    for col in ['dfzh_0_jyje_sum', 'dfzh_1_jyje_sum']:
        tmp = time_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # dfzh
    # dfzh_zhdh_count dfzh_zhdh_nunique
    dfzh_feature = time_info.groupby(['dfzh']).agg({'jyje': ['nunique']}).reset_index()
    dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]

    time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)

    for col in ['dfzh_jyje_nunique']:
        tmp = time_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # dfzh
    # dfzh_zhdh_count dfzh_zhdh_nunique
    dfzh_feature = time_info.groupby(['dfzh']).agg({'jyrq_jysj': ['mean', 'max', 'min', 'var', np.ptp]}).reset_index()
    dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]

    time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)

    for col in ['dfzh_jyrq_jysj_mean', 'dfzh_jyrq_jysj_max',
                'dfzh_jyrq_jysj_min', 'dfzh_jyrq_jysj_var', 'dfzh_jyrq_jysj_ptp']:
        tmp = time_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
        tmp.columns = [x[0] if x[1] == '' else x[0] + '_' + x[1] for x in tmp.columns]
        train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
        test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
        del tmp

    # 交叉特征
    # 最后一次交以距离开户时间差
    train['global_jyrq_jysj_khrq'] = train['global_jyrq_jysj_max'] - train['khrq']
    test['global_jyrq_jysj_khrq'] = test['global_jyrq_jysj_max'] - test['khrq']

    train['jdbj_1_dfzh_count/jdbj_0_dfzh_count'] = train['jdbj_1_dfzh_count'] / train['jdbj_0_dfzh_count']
    test['jdbj_1_dfzh_count/jdbj_0_dfzh_count'] = test['jdbj_1_dfzh_count'] / test['jdbj_0_dfzh_count']

    train['jdbj_1_dfzh_nunique/jdbj_0_dfzh_nunique'] = train['jdbj_1_dfzh_nunique'] / train['jdbj_0_dfzh_nunique']
    test['jdbj_1_dfzh_nunique/jdbj_0_dfzh_nunique'] = test['jdbj_1_dfzh_nunique'] / test['jdbj_0_dfzh_nunique']

    # 连续交易的最大次数
    tt = time_info.groupby(['zhdh'])['jdbj'].apply(lambda x: calc_max_coutinut_times(list(x), b=1)).reset_index()
    tt.columns = ['zhdh', 'jdbj_list_1_times']

    train = pd.merge(train, tt, on=['zhdh'], how='left', copy=False)
    test = pd.merge(test, tt, on=['zhdh'], how='left', copy=False)

    train.replace(np.inf, np.nan, inplace=True)
    train.replace(-np.inf, np.nan, inplace=True)

    test.replace(np.inf, np.nan, inplace=True)
    test.replace(-np.inf, np.nan, inplace=True)

    data = pd.concat([train, test], copy=False).reset_index(drop=True)
    data.replace(-np.inf, np.nan, inplace=True)
    data.replace(np.inf, np.nan, inplace=True)

    feature = [x for x in data.columns if
               x not in ['zhdh', 'black_flag']]
    label = 'black_flag'

    train = data[:train.shape[0]]
    test = data[train.shape[0]:]

    train = train.fillna(-1)
    test = test.fillna(-1)

    train.replace(np.inf, 999, inplace=True)
    train.replace(-np.inf, -999, inplace=True)

    feat_imp = pd.DataFrame()
    feat_imp['name'] = feature

    nsp = 10
    skf = StratifiedKFold(n_splits=nsp, random_state=42, shuffle=True)

    oof_valid = np.zeros(shape=(train.shape[0], 1))
    oof_test = np.zeros(shape=(test.shape[0], nsp))

    for index, (tr_index, va_index) in enumerate(skf.split(train[feature], train[label])):
        X_train, y_train = train[feature].iloc[tr_index], train[label].iloc[tr_index]
        X_valid, y_valid = train[feature].iloc[va_index], train[label].iloc[va_index]

        print(X_train.shape, y_train.shape)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_jobs': -1,
            'seed': 1024 + 42 * index,
            'verbose': -1
        }
        callbacks = [log_evaluation(period=250),
                     early_stopping(stopping_rounds=250),
                     ]

        print('Starting training...')

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=[lgb_eval],
                        callbacks=callbacks,
                        # fobj=focal_loss,
                        # feval=eval_error
                        # feval=lgb_f1_score
                        )
        y_val_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration).reshape(-1, 1)
        oof_valid[va_index] = y_val_pred

        y_test_pred = gbm.predict(test[feature].values, num_iteration=gbm.best_iteration)
        oof_test[:, index] = y_test_pred

        feat_imp['imp_{}'.format(index)] = gbm.feature_importance()

    feat_imp['imp'] = 0
    for i in range(0, nsp):
        feat_imp['imp'] += feat_imp['imp_{}'.format(i)]

    train['predict'] = list(oof_valid.reshape(1, -1)[0])
    tmp_train = train[['zhdh', label, 'predict']]

    # 截取数据
    tmp_train = tmp_train.sort_values(['predict'], ascending=False).reset_index(drop=True)
    tmp_train['p'] = 0
    tmp_train.loc[tmp_train.index < train.shape[0] * 0.25, 'p'] = 1

    f1 = f1_score(tmp_train['black_flag'], tmp_train['p'])
    print(f1)

    tmp_train[['zhdh', 'black_flag', 'predict']].to_csv('../submit/train_{}_lgb.csv'.format(mm), index=False)

    test['predict'] = np.mean(oof_test, axis=1)

    submit = test[['zhdh', 'predict']]
    submit = submit.sort_values(['predict'], ascending=False).reset_index(drop=True)
    submit['black_flag'] = 0
    submit.loc[submit.index < submit.shape[0] * 0.25, 'black_flag'] = 1

    submit[['zhdh', 'black_flag', 'predict']].to_csv('../submit/test_{}_lgb.csv'.format(mm), index=False)

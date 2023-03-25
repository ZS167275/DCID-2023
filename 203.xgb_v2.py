# coding:utf-8
import datetime
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


def gen_thres_new(df_train, oof_preds):
    df_train['oof_preds'] = oof_preds
    quantile_point = df_train['black_flag'].mean()
    thres = df_train['oof_preds'].quantile(1 - quantile_point)

    _thresh = []
    for thres_item in np.arange(thres - 0.2, thres + 0.2, 0.01):
        _thresh.append(
            [thres_item, f1_score(df_train['black_flag'], np.where(oof_preds > thres_item, 1, 0), average='macro')])

    _thresh = np.array(_thresh)
    best_id = _thresh[:, 1].argmax()
    best_thresh = _thresh[best_id][0]

    print("阈值: {}\n训练集的f1: {}".format(best_thresh, _thresh[best_id][1]))
    return best_thresh


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    # y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    submit = pd.DataFrame()
    submit['bz_predict'] = y_hat
    submit['flag'] = y_true
    submit = submit.sort_values(['bz_predict'], ascending=False).reset_index(drop=True)
    submit['black_flag'] = 0
    submit.loc[submit.index < submit.shape[0] * 0.25, 'black_flag'] = 1
    return 'f1', f1_score(submit['black_flag'], submit['flag']), True


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


def gen_target_encoding_feats(train, test, encode_cols, target_col, n_fold=5):
    '''生成target encoding特征'''
    # for training set - cv
    tg_feats = np.zeros((train.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[encode_cols], train[target_col])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values

    for idx, encode_col in enumerate(encode_cols):
        train[f'{encode_col}_mean_target'] = tg_feats[:, idx]

    # for testing set
    for col in encode_cols:
        target_mean_dict = train.groupby(col)[target_col].mean()
        test[f'{col}_mean_target'] = test[col].map(target_mean_dict)

    return train, test


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

static_info = pd.read_csv(path + '账户静态信息.csv')
time_info = pd.read_csv(path + '账户交易信息.csv')
train_label = pd.read_csv(path + '训练集标签.csv')
test_no_label = pd.read_csv(path + 'test_dataset.csv')


time_info['tt'] = time_info['jyrq'] + ' ' + time_info['jysj']
time_info['tt'] = pd.to_datetime(time_info['tt'])
time_info = time_info.sort_values(['zhdh','tt'])

time_info['ttt'] = time_info['jdbj'].apply(lambda x:1 if x == 0 else -1)

time_info['ttt_jyje'] = time_info['jyje'] * time_info['ttt']
time_info['tttt'] = time_info['zhye'] + time_info['ttt_jyje']


a = time_info[['zhdh','jyje','zhye','tttt','ttt_jyje','ttt']]
a['ttttt'] = a.groupby(['zhdh'])['tttt'].shift(-1)
a['zhye'] = a['zhye'].astype('float')
a['ttttt'] = a['ttttt'].astype('float')
a['ttttt_tttt'] = a['zhye'] - a['ttttt']
a = a.dropna()
a['ttttt_tttt'] = a['ttttt_tttt'].astype(int)
ff1 = a.groupby(['zhdh']).agg({'ttttt_tttt':['mean','var']}).reset_index()
ff1.columns = ['zhdh','ttttt_tttt_mean','ttttt_tttt_var']

del time_info['ttt']
del time_info['ttt_jyje']
del time_info['tt']
del time_info['tttt']
# 3月模型
add_train_3 = pd.read_csv('../submit/train_3_lgb.csv')
add_train_3.columns = ['zhdh', 'black_flag_3', 'predict_3']
add_test_3 = pd.read_csv('../submit/test_3_lgb.csv')
add_test_3.columns = ['zhdh', 'black_flag_3', 'predict_3']
# 4月模型
add_train_4 = pd.read_csv('../submit/train_4_lgb.csv')
add_train_4.columns = ['zhdh', 'black_flag_4', 'predict_4']
add_test_4 = pd.read_csv('../submit/test_4_lgb.csv')
add_test_4.columns = ['zhdh', 'black_flag_4', 'predict_4']
# 5月模型
add_train_5 = pd.read_csv('../submit/train_5_lgb.csv')
add_train_5.columns = ['zhdh', 'black_flag_5', 'predict_5']
add_test_5 = pd.read_csv('../submit/test_5_lgb.csv')
add_test_5.columns = ['zhdh', 'black_flag_5', 'predict_5']

add_train = pd.merge(add_train_3[['zhdh', 'predict_3']], add_train_4[['zhdh', 'predict_4']], on=['zhdh'])
add_train = pd.merge(add_train, add_train_5[['zhdh', 'predict_5']], on=['zhdh'])

add_test = pd.merge(add_test_3[['zhdh', 'predict_3']], add_test_4[['zhdh', 'predict_4']], on=['zhdh'])
add_test = pd.merge(add_test, add_test_5[['zhdh', 'predict_5']], on=['zhdh'])

# # 3月模型
# add_train_xgb_3 = pd.read_csv('../submit/train_3_xgb.csv')
# add_train_xgb_3.columns = ['zhdh', 'black_flag_xgb_3', 'predict_xgb_3']
# add_test_xgb_3 = pd.read_csv('../submit/test_3_xgb.csv')
# add_test_xgb_3.columns = ['zhdh', 'black_flag_xgb_3', 'predict_xgb_3']
# # 4月模型
# add_train_xgb_4 = pd.read_csv('../submit/train_4_xgb.csv')
# add_train_xgb_4.columns = ['zhdh', 'black_flag_xgb_4', 'predict_xgb_4']
# add_test_xgb_4 = pd.read_csv('../submit/test_4_xgb.csv')
# add_test_xgb_4.columns = ['zhdh', 'black_flag_xgb_4', 'predict_xgb_4']
# # 5月模型
# add_train_xgb_5 = pd.read_csv('../submit/train_5_xgb.csv')
# add_train_xgb_5.columns = ['zhdh', 'black_flag_xgb_5', 'predict_xgb_5']
# add_test_xgb_5 = pd.read_csv('../submit/test_5_xgb.csv')
# add_test_xgb_5.columns = ['zhdh', 'black_flag_xgb_5', 'predict_xgb_5']
#
# add_train_xgb = pd.merge(add_train_xgb_3[['zhdh', 'predict_xgb_3']], add_train_xgb_4[['zhdh', 'predict_xgb_4']],
#                          on=['zhdh'])
# add_train_xgb = pd.merge(add_train_xgb, add_train_xgb_5[['zhdh', 'predict_xgb_5']], on=['zhdh'])
#
# add_test_xgb = pd.merge(add_test_xgb_3[['zhdh', 'predict_xgb_3']], add_test_xgb_4[['zhdh', 'predict_xgb_4']],
#                         on=['zhdh'])
# add_test_xgb = pd.merge(add_test_xgb, add_test_xgb_5[['zhdh', 'predict_xgb_5']], on=['zhdh'])

###########################


# # 3月模型
# add_train_cat_3 = pd.read_csv('../submit/train_cat_3.csv')
# add_train_cat_3.columns = ['zhdh', 'black_flag_cat_3', 'predict_cat_3']
# add_test_cat_3 = pd.read_csv('../submit/test_3_xgb.csv')
# add_test_cat_3.columns = ['zhdh', 'black_flag_cat_3', 'predict_cat_3']
# # 4月模型
# add_train_cat_4 = pd.read_csv('../submit/train_cat_4.csv')
# add_train_cat_4.columns = ['zhdh', 'black_flag_cat_4', 'predict_cat_4']
# add_test_cat_4 = pd.read_csv('../submit/test_cat_4.csv')
# add_test_cat_4.columns = ['zhdh', 'black_flag_cat_4', 'predict_cat_4']
# # 5月模型
# add_train_cat_5 = pd.read_csv('../submit/train_cat_5.csv')
# add_train_cat_5.columns = ['zhdh', 'black_flag_cat_5', 'predict_cat_5']
# add_test_cat_5 = pd.read_csv('../submit/test_cat_5.csv')
# add_test_cat_5.columns = ['zhdh', 'black_flag_cat_5', 'predict_cat_5']

# add_train_cat = pd.merge(add_train_cat_3[['zhdh', 'predict_cat_3']], add_train_cat_4[['zhdh', 'predict_cat_4']],
#                          on=['zhdh'])
# add_train_cat = pd.merge(add_train_cat, add_train_cat_5[['zhdh', 'predict_cat_5']], on=['zhdh'])
#
# add_test_cat = pd.merge(add_test_cat_3[['zhdh', 'predict_cat_3']], add_test_cat_4[['zhdh', 'predict_cat_4']],
#                         on=['zhdh'])
# add_test_cat = pd.merge(add_test_cat, add_test_cat_5[['zhdh', 'predict_cat_5']], on=['zhdh'])
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

train = pd.merge(train, add_train, on=['zhdh'], how='left', copy=False)
test = pd.merge(test, add_test, on=['zhdh'], how='left', copy=False)

# test = pd.merge(test, ff1, on=['zhdh'], how='left', copy=False)
# train = pd.merge(train, ff1, on=['zhdh'], how='left', copy=False)



# gen_target_encoding_feats()
#
# train = pd.merge(train, add_train_xgb, on=['zhdh'], how='left', copy=False)
# test = pd.merge(test, add_test_xgb, on=['zhdh'], how='left', copy=False)
#
# train = pd.merge(train, add_train_cat, on=['zhdh'], how='left', copy=False)
# test = pd.merge(test, add_test_cat, on=['zhdh'], how='left', copy=False)


time_info['duichong_1'] = time_info['jyje'] < 0
time_info['duichong_2'] = time_info.groupby(['zhdh'])['duichong_1'].shift(-1)
time_info['duichong_2'] = time_info['duichong_2'].fillna(False)

time_info['duichong'] = time_info['duichong_1'] + time_info['duichong_2']
print(time_info.shape)
time_info = time_info[time_info['duichong'] == 0].reset_index(drop=True)
print(time_info.shape)

del time_info['duichong']
del time_info['duichong_1']
del time_info['duichong_2']

time_info['jyje'] = np.log(time_info['jyje'])
time_info['zhye'] = np.log(time_info['zhye'])

# 扩展时间
prefix = 'jyrq'
time_info['t_jyrq'] = pd.to_datetime(time_info['jyrq'], format='%Y-%m-%d')

# 时间偏移，消除时间影响
tc_offset = time_info.groupby(['zhdh'])['t_jyrq'].max().reset_index()
tc_offset.columns = ['zhdh', 't_jyrq_offset_t']

tc_offset['t_jyrq_offset_t'] = pd.to_datetime('2020-05-31') - tc_offset['t_jyrq_offset_t']
tc_offset['t_jyrq_offset_t'] = tc_offset['t_jyrq_offset_t'].dt.days

time_info = pd.merge(time_info, tc_offset, on=['zhdh'])
time_info['t_jyrq'] = time_info[['t_jyrq', 't_jyrq_offset_t']].apply(lambda x: x[0] + datetime.timedelta(days=x[1]),
                                                                     axis=1)
time_info['jyrq'] = time_info['t_jyrq'].astype(str)

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

nx_feature = time_info[['nx_zhdh', 'nx_dfzh', 'nx_weight']]

nx_elist = [(x[0], x[1]) for x in nx_feature.values]
G = nx.Graph()
G.add_edges_from(nx_elist)

nx_f1 = nx.degree_centrality(G)

nx_f2 = nx.pagerank(G, alpha=0.9, max_iter=2000, tol=0.05)

time_info['nx_f1'] = time_info['nx_zhdh'].map(nx_f1)
time_info['nx_f2'] = time_info['nx_zhdh'].map(nx_f2)

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
# time_info['ff'] = 1
# pv_feature_min = pd.pivot_table(time_info,'ff','zhdh','jyqd',aggfunc='sum').reset_index()
# pv_feature_min = pv_feature_min.fillna(0)
# train = pd.merge(train, pv_feature_min, on=['zhdh'], how='left', copy=False)
# test = pd.merge(test, pv_feature_min, on=['zhdh'], how='left', copy=False)

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
    time_info['jyrq_month'] = time_info['jyrq'] + time_info['jysj'].apply(lambda x: ':'.join(str(x).split(':')[:-1]))
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
    'zhdh_dfzh_count', 'zhdh_jdbj_count', 'zhdh_dfhh_count', 'zhdh_jyrq_count', 'zhdh_jyqd_count', 'zhdh_zydh_count',
    'zhdh_dfmccd_count',
    'jyje',
    'jyrqmonth', 'jyrqday', 'jyrqweekofyear', 'jyrqdayofyear',
    'jyrqdayofweek', 'jyrqis_wknd', 'jyrqis_month_start',
    'jyrqis_month_end',
    'dfmccd',
    # 'nx_f1','nx_f2'

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

    # tmp = time_info[time_info['jdbj']==0].groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
    # tmp.columns = [x[0] if x[1] == '' else 'd_jdbj_0_' + x[0] + '_' + x[1] for x in tmp.columns]
    # train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
    # test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
    # del tmp
    #
    # tmp = time_info[time_info['jdbj']==1].groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
    # tmp.columns = [x[0] if x[1] == '' else 'd_jdbj_1_' + x[0] + '_' + x[1] for x in tmp.columns]
    # train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
    # test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
    # del tmp

# dfzh_1_jyje_sum
dfzh_feature = time_info[time_info['jdbj'] == 1].groupby(['dfzh']).agg({'jyje': ['sum']}).reset_index()
dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_1_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]
time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)

dfzh_feature = time_info[time_info['jdbj'] == 0].groupby(['dfzh']).agg({'jyje': ['sum']}).reset_index()
dfzh_feature.columns = [x[0] if x[1] == '' else 'dfzh_0_' + x[0] + '_' + x[1] for x in dfzh_feature.columns]
time_info = pd.merge(time_info, dfzh_feature, on=['dfzh'], how='left', copy=False)
# time_info['dfzh_1_jyje_sum/dfzh_0_jyje_sum'] = time_info['dfzh_0_jyje_sum'] / time_info['dfzh_1_jyje_sum']

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
# time_info['dfzh_jyje_nunique_count'] = time_info['dfzh_zhdh_nunique'] / time_info['dfzh_zhdh_count']

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
# time_info['dfzh_jyje_nunique_count'] = time_info['dfzh_zhdh_nunique'] / time_info['dfzh_zhdh_count']

for col in ['dfzh_jyrq_jysj_mean', 'dfzh_jyrq_jysj_max',
            'dfzh_jyrq_jysj_min', 'dfzh_jyrq_jysj_var', 'dfzh_jyrq_jysj_ptp']:
    tmp = time_info.groupby(['zhdh']).agg({col: ['mean', 'var']}).reset_index()
    tmp.columns = [x[0] if x[1] == '' else x[0] + '_' + x[1] for x in tmp.columns]
    train = pd.merge(train, tmp, on=['zhdh'], how='left', copy=False)
    test = pd.merge(test, tmp, on=['zhdh'], how='left', copy=False)
    del tmp

# w2v
# time_info['jdbj'] = time_info['jdbj'].astype(str)
# w2v_f = time_info.groupby(['zhdh'])['jdbj'].apply(lambda x:list(x)).reset_index()
# w2v_f.columns = ['zhdh','jdbj_list']
#
# sentences = w2v_f['jdbj_list'].to_list()
# model = Word2Vec(min_count=1, vector_size=10)
# model.build_vocab(sentences)  # prepare the model vocabulary
# model.train(sentences, total_examples=model.corpus_count, epochs=10)  # train word vectors
# a_emb = []
# for index, t in enumerate(sentences):
#     try:
#         v = model.wv.get_vector(str(t[0]))
#     except:
#         v = np.zeros(shape=(1, 10))
#     a_emb.append(np.array(v).reshape(1, -1))
# tmp_user_w2v = np.array(a_emb).reshape(-1, 10)
# for i in range(0, 10):
#     w2v_f.loc[:, 'v2v_{}'.format(i)] = tmp_user_w2v[:, i]
# del w2v_f['jdbj_list']
# model.save("./output/data/{}_{}.bin".format(key1, dt))  # 保存完整的模型,除包含词-向量,还保存词频等训练所需信息
# model.wv.save_word2vec_format("./output/data/{}_{}.dict".format(key1, dt))  # 保存的模型仅包含词-向量信息
# return w2v
# train = pd.merge(train, w2v_f, on=['zhdh'], how='left', copy=False)
# test = pd.merge(test, w2v_f, on=['zhdh'], how='left', copy=False)

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

train = train.fillna(-999)
test = test.fillna(-999)

train.replace(np.inf, 0, inplace=True)
train.replace(-np.inf, 0, inplace=True)

test.replace(np.inf, 0, inplace=True)
test.replace(-np.inf, 0, inplace=True)

nsp = 5

train = train.fillna(-1)
test = test.fillna(-1)

train.replace(np.inf, 999, inplace=True)
train.replace(-np.inf, -999, inplace=True)

test.replace(np.inf, 999, inplace=True)
test.replace(-np.inf, -999, inplace=True)

train['predict_5/predict_3'] = train['predict_5'] / train['predict_3']
train['predict_5/predict_4'] = train['predict_5'] / train['predict_4']
train['predict_5/predict_3_4'] = train['predict_5'] / (train['predict_3'] * train['predict_4'])

train['predict_5/predict_3_rank'] = train['predict_5/predict_3'].rank()
train['predict_5/predict_4_rank'] = train['predict_5/predict_4'].rank()
train['predict_5/predict_3_4_rank'] = train['predict_5/predict_3_4'].rank()

test['predict_5/predict_3'] = test['predict_5'] / test['predict_3']
test['predict_5/predict_4'] = test['predict_5'] / test['predict_4']
test['predict_5/predict_3_4'] = test['predict_5'] / (test['predict_3'] * test['predict_4'])

test['predict_5/predict_3_rank'] = test['predict_5/predict_3'].rank()
test['predict_5/predict_4_rank'] = test['predict_5/predict_4'].rank()
test['predict_5/predict_3_4_rank'] = test['predict_5/predict_3_4'].rank()

feature = [x for x in train.columns if
           x not in ['zhdh', 'black_flag', 'jyrq', 'jysj', 'predict', 'oof_preds']]
label = 'black_flag'

feat_imp = pd.DataFrame()

feat_imp['name'] = feature
skf = StratifiedKFold(n_splits=nsp, random_state=42, shuffle=True)

for ssr in [1, 10, 100, 1000, 10000]:

    oof_valid = np.zeros(shape=(train.shape[0], 1))
    oof_test = np.zeros(shape=(test.shape[0], nsp))

    # prb_train_add_1 = pd.read_csv('../data/prb_train_add_1.csv')
    # prb_train_add_0 = pd.read_csv('../data/prb_train_add_0.csv')
    # prb_train_add = pd.concat([prb_train_add_1,prb_train_add_0]).sample(frac=1)
    np.random.seed(42 + i * ssr)
    # np.random.shuffle(feature)
    for index, (tr_index, va_index) in enumerate(skf.split(train[feature], train[label])):
        X_train, y_train = train[feature].iloc[tr_index], train[label].iloc[tr_index]
        X_valid, y_valid = train[feature].iloc[va_index], train[label].iloc[va_index]

        # offser_ = np.random.randint(0, 11)
        # print(offser_)
        # if offser_ >= 6:
        # X_train_add , y_train_add = prb_train_add[feature],prb_train_add[label]
        X_train = pd.concat([X_train, X_train, X_train, X_train, X_train]).reset_index(drop=True)
        y_train = pd.concat([y_train, y_train, y_train, y_train, y_train]).reset_index(drop=True)
        # else:
        #     X_train = X_train
        #     y_train = y_train

        train_matrix = xgb.DMatrix(X_train, label=y_train)
        valid_matrix = xgb.DMatrix(X_valid, label=y_valid)
        test_matrix = xgb.DMatrix(test[feature])

        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'gamma': 1.5,
                  'min_child_weight': 1.5,
                  'max_depth': 5,
                  'lambda': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.7,
                  'colsample_bylevel': 0.7,
                  'eta': 0.05,
                  'tree_method': 'exact',
                  'seed': 1024 + index * 42 + ssr * 42,
                  # 'nthread': 8
                  }

        watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

        model = xgb.train(params, train_matrix, num_boost_round=5000, evals=watchlist, verbose_eval=250,
                          early_stopping_rounds=500)
        y_val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
        y_test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)

        oof_valid[va_index] = y_val_pred.reshape(-1, 1)
        oof_test[:, index] = y_test_pred

        feat_imp['imp_{}'.format(index)] = model.get_score()
        feat_imp['imp_{}'.format(index)] = feat_imp['imp_{}'.format(index)] / (
                feat_imp['imp_{}'.format(index)].max() - feat_imp['imp_{}'.format(index)].min())

    # feat_imp['imp'] = feat_imp['imp_0'] + feat_imp['imp_1'] + feat_imp['imp_2'] + feat_imp['imp_3'] + feat_imp['imp_4']

    train['predict'] = list(oof_valid.reshape(1, -1)[0])
    tmp_train = train[[label, 'predict']]

    # 截取数据
    tmp_train = tmp_train.sort_values(['predict'], ascending=False).reset_index(drop=True)
    tmp_train['p'] = 0
    tmp_train.loc[tmp_train.index < train.shape[0] * 0.25, 'p'] = 1

    f1 = f1_score(tmp_train['black_flag'], tmp_train['p'])
    print(f1)
    # if f1 > 0.86:
    test['bz_predict_{}'.format(ssr)] = np.mean(oof_test, axis=1)
    # else:
    #     print(f1)

use_col = []
for xx in test.columns:
    if str(xx).__contains__('bz_predict_'):
        use_col.append(xx)

test['bz_predict'] = test[use_col].mean(axis=1)
submit = test[['zhdh', 'bz_predict']]
submit = submit.sort_values(['bz_predict'], ascending=False).reset_index(drop=True)
submit['black_flag'] = 0
submit.loc[submit.index < submit.shape[0] * 0.25, 'black_flag'] = 1
submit[['zhdh', 'black_flag']].to_csv('../submit/xgb_{}.csv'.format(len(use_col)), index=False)
submit[['zhdh', 'bz_predict']].to_csv('../submit/xgb_bz_predict_{}.csv'.format(len(use_col)), index=False)

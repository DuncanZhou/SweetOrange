#!/usr/bin/python
#-*-coding:utf-8-*-
'''@Date:18-10-13'''
'''@Time:下午10:02'''
'''@author: Duncan'''

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import lightgbm as LGB
import xgboost as XGB
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


xgb_param = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 5,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.05,                  # 如同学习率
    'seed': 2018,
    'nthread': 16,                  # cpu 线程数
}


lgb_param = {'num_leaves':31, 'num_trees':100, 'objective':'binary','metric':{'auc'},'learning_rate':0.05, \
             'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'boosting_type':'gbdt','nthread': 16}

# stacking方法
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

# n折交叉验证
def N_Split(_train,_label,n=5):
    data = []
    skf = StratifiedKFold(n_splits=n)
    for train,test in skf.split(_train,_label):
        Train_x = _train.loc[train]
        Test_x = _train.loc[test]
        Train_y = _label.loc[train]
        Test_y = _label.loc[test]
        data.append([Train_x,Train_y,Test_x,Test_y])
    return data

# 数值型归一化
def Normalize(_train,_test):
    len_t = len(_train)
    data = pd.concat((_train,_test))
    # numeric_cols = [col for col in data.columns if data[col].dtypes == np.float or data[col].dtypes == np.int]
    numeric_cols = _train.columns
    min_max_scaler = MinMaxScaler()
    data[numeric_cols] = min_max_scaler.fit_transform(data[numeric_cols])
    return data.iloc[:len_t],data.iloc[len_t:]

# 模型预测
def Predict(_train,features,_label,_test,model='lgb'):
    num_round = 200
    pred = None
    # lgb
    if model == 'lgb':
        Train_data = LGB.Dataset(_train[features],label=_label)
        bst = LGB.train(lgb_param,Train_data,num_round)
        # 输出feature_importance
        columns = pd.DataFrame({
            'column': _train[features].columns,
            'importance': bst.feature_importance(),
        }).sort_values(by='importance',ascending=False)
        columns.to_csv("/data/ymzhou/SweetOrange/data/features_importance.csv",index=False)
        pred = bst.predict(_test[features])
    # xgb
    elif model == 'xgb':
        dtrain = XGB.DMatrix(_train[features],label=_label)
        dtest = XGB.DMatrix(_test[features])
        bst = XGB.train(xgb_param,dtrain,num_round)
        pred = bst.predict(dtest)

    # logistic
    elif model == 'logistic':
        # 数值型需要归一化
        _train,_test = Normalize(_train[features],_test[features])
        clf = LogisticRegression(random_state=2018, solver='lbfgs',multi_class='multinomial').fit(_train,_label)
        pred = clf.predict_proba(_test)[:,1]
    return pred

# 本题评分标准
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


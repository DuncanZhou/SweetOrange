#!/usr/bin/python
#-*-coding:utf-8-*-
'''@Date:18-10-30'''
'''@Time:下午3:33'''
'''@author: Duncan'''

import pandas as pd
import numpy as np
import Features as _F
import Models as _M
import PreProcess as _P
import feather
from sklearn.metrics import roc_auc_score
import os



pri_id = "UID"

data_path = '/data/ymzhou/SweetOrange/'
train_path = data_path + "train/"
test_path = data_path + "test/"
_train = pd.DataFrame()
_test = pd.DataFrame()

# trans_train_info = pd.read_csv(train_path + "transaction_TRAIN.csv")
trans_train_info = pd.read_csv(train_path + "transaction_train_new.csv")
# trans_test_info = pd.read_csv(test_path + "transaction_round1.csv")
trans_test_info = pd.read_csv(test_path + "test_transaction_round2.csv")

# op_train_info = pd.read_csv(train_path + "operation_TRAIN.csv")
op_train_info = pd.read_csv(train_path + "operation_train_new.csv")
# op_test_info = pd.read_csv(test_path + "operation_round1.csv")
op_test_info = pd.read_csv(test_path + "test_operation_round2.csv")

_train[pri_id] = pd.concat((op_train_info[pri_id],trans_train_info[pri_id])).unique()
_test[pri_id] = pd.concat((op_test_info[pri_id],trans_test_info[pri_id])).unique()

# label = pd.read_csv(train_path + "tag_TRAIN.csv")
label = pd.read_csv(train_path + "tag_train_new.csv")

op_info = pd.concat((op_train_info,op_test_info))

trans_info = pd.concat((trans_train_info,trans_test_info))



# 处理操作数据
def ProcessOperation(df):

    data = pd.DataFrame()
    data[pri_id] = pd.concat((_train[pri_id],_test[pri_id]))
    # day处理 (分为上 中 下旬)
    temp = _F.MonthCount(df,pri_id)
    data = pd.merge(data,temp,on=pri_id,how='left')

    # mode
    temp = _F.CatRowsToCols(df,pri_id,'mode','os')
    data = pd.merge(data,temp,on=pri_id,how='left')

    # success
    temp = _F.CatRowsToCols(df,pri_id,'success','os')
    data = pd.merge(data,temp,on=pri_id,how='left')

    # version
    temp = _F.CatRowsToCols(df,pri_id,'version','os')
    data = pd.merge(data,temp,on=pri_id,how='left')

    # time
    df['day_period'] = df['time'].apply(_F.TimeInterval)
    temp = _F.CatRowsToCols(df,pri_id,'day_period','os')
    data = pd.merge(data,temp,on=pri_id,how='left')

    cols = ['device2','ip1','ip2','mac1','mac2','device_code1','device_code2','device_code3']

    for col in cols:
        # device2 (用户有多少不同型号的设备)
        temp = _F.GetCount(df,pri_id,col,'os')
        data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计地理位置次数
    temp = _F.CountWS(df)
    data = pd.merge(data,temp,on=pri_id,how='left')
    #
    # # 危险设备（安卓）
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','device_code1','time',500)
    # data = pd.merge(data,temp,on=pri_id,how='left')


    # 危险设备（安卓）
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','device_code2','time',500)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    #
    # # 危险设备（苹果）
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','device_code3','time',200)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险地理位置
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','geo_code','time',500)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险操作类型
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','mode','time',10000)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险mac1地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','mac1','time',600)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险mac2地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','mac2','time',1000)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险ip1地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','ip1','time',1000)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险ip1_sub地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','ip1_sub','time',1000)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险ip2_sub地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','ip2_sub','time',50)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计每个用户最常出现的经纬度
    # temp = _F.PositionWS(df)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 地理位置聚类

    data = data.fillna(0)
    return data

# 处理交易数据
def ProcessTrans(df):
    data = pd.DataFrame()
    data[pri_id] = pd.concat((_train[pri_id],_test[pri_id]))

    # channel 统计次数
    temp = _F.CatRowsToCols(df,pri_id,'channel','day')
    data = pd.merge(data,temp,on=pri_id,how='left')

    # day
    temp = _F.MonthCount(df,pri_id)
    data = pd.merge(data,temp,on=pri_id,how='left')

    # time
    df['day_period'] = df['time'].apply(_F.TimeInterval)
    temp = _F.CatRowsToCols(df,pri_id,'day_period','day')
    data = pd.merge(data,temp,on=pri_id,how='left')

    # trans_amt (交易金额)(最大值，最小值，平均值)
    # temp = _F.GetValMaxMin(df,pri_id,'trans_amt')
    # data = pd.merge(data,temp,on=pri_id,how='left')
    # temp = _F.GetValAvg(df,pri_id,'trans_amt')
    # data = pd.merge(data,temp,on=pri_id,how='left')
    # temp = _F.GetValSum(df,pri_id,'trans_amt')
    # data = pd.merge(data,temp,on=pri_id,how='left')


    # 计算不同的次数
    cols = ['device2','ip1','mac1','device_code1','device_code2','device_code3','amt_src1','amt_src2','merchant','trans_type1','trans_type2','acc_id1','market_type','market_code']
    for col in cols:
        temp = _F.GetCount(df,pri_id,col,'day')
        data = pd.merge(data,temp,on=pri_id,how='left')

    # 脱敏后的余额(最大值，最小值，平均值)
    # temp = _F.GetValMaxMin(df,pri_id,'bal')
    # data = pd.merge(data,temp,on=pri_id,how='left')
    # temp = _F.GetValAvg(df,pri_id,'bal')
    # data = pd.merge(data,temp,on=pri_id,how='left')
    # temp = _F.GetValSum(df,pri_id,'bal')
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 危险设备（安卓）
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','device_code1','time',100)
    # data = pd.merge(data,temp,on=pri_id,how='left')
    #
    # # 危险设备（安卓）
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','device_code2','time',100)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 危险设备（苹果）
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','device_code3','time',80)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险地理位置
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','geo_code','time',200)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险商家
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','merchant','time',1000)
    # data = pd.merge(data,temp,on=pri_id,how='left')
    #
    # # 统计地理位置次数
    temp = _F.CountWS(df)
    data = pd.merge(data,temp,on=pri_id,how='left')
    #
    # 统计危险mac1地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','mac1','time',600)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险ip1地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','ip1','time',1000)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险ip1_sub地址
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','ip1_sub','time',1000)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    # 统计危险交易账户
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','acc_id1','time',200)
    # data = pd.merge(data,temp,on=pri_id,how='left')
    #
    # # 统计危险转出账户
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','acc_id2','time',100)
    # data = pd.merge(data,temp,on=pri_id,how='left')
    #
    # # 统计危险转入账户
    # temp = _F.CountDangerous(df,label,pri_id,'Tag','acc_id3','time',100)
    # data = pd.merge(data,temp,on=pri_id,how='left')


    # 统计每个用户最常出现的经纬度
    # temp = _F.PositionWS(df)
    # data = pd.merge(data,temp,on=pri_id,how='left')

    data = data.fillna(0)
    return data

# 处理操作和交易数据，处理列名
def Base_Process(encode_type='LabelEncode'):
    if os.path.exists(data_path + "data/data.feather"):
        df = feather.read_dataframe(data_path + "data/data.feather")
        return df

    data = pd.DataFrame()
    data[pri_id] = pd.concat((_train[pri_id],_test[pri_id]))
  
    # 这几列的值要按pri_id相加

    cols = ['device2','ip1','mac1','device_code1','device_code2','device_code3']
    cols = [col+"_count" for col in cols]
    op = ProcessOperation(op_info)
    trans = ProcessTrans(trans_info)

    temp = pd.DataFrame()
    temp[cols] = op[cols] + trans[cols]


    c_cols = ['device_code1','device_code2','device_code3','geo_code','mac1','ip1','ip1_sub']
    c_cols = [col+"_d_label" for col in c_cols]

    temp_c = pd.DataFrame()
    temp_c[c_cols] = op[c_cols] + trans[c_cols]


    # 对op和trans列名重新编码,防止重复名称
    _Prep = _P.Process()

    cols.append(pri_id)
    op = _Prep.RenameColumns(op,cols,'op')
    trans = _Prep.RenameColumns(trans,cols,'trans')


    cols.remove(pri_id)
    
    op_cols = [col for col in op.columns if col not in cols and col not in c_cols]
    tran_cols = [col for col in trans.columns if col not in cols and col not in c_cols]

    # op_cols = [col for col in op.columns if col not in cols]
    # tran_cols = [col for col in trans.columns if col not in cols]


  
    data = pd.merge(data,op[op_cols],on=pri_id,how='left')

    data = pd.merge(data,trans[tran_cols],on=pri_id,how='left')

    # 连接temp
    data = pd.concat((data,temp),axis=1)
    data = pd.concat((data,temp_c),axis=1)


    # 加入label
    data = pd.merge(data,label,on=pri_id,how='left')
    data.rename(columns={'Tag':'y'},inplace=True)

    # 编码
    _Prep = _P.Process()
    data =_Prep.CatColConvert(data,pri_id,encode_type)

    data = data.fillna(0)

    # 持久化
    feather.write_dataframe(data,data_path + "data/data.feather")
    return data

# 聚类地址
def _F_Clsuter_Geo():
    if os.path.exists(data_path + "data/_F_geo.feather"):
        df = feather.read_dataframe(data_path + "data/_F_geo.feather")
        return df
    # 合并 operation和transaction的uid，geo_code
    geo_info = pd.concat((op_info[[pri_id,'geo_code']],trans_info[[pri_id,'geo_code']]))
    geo_info['pos'] = geo_info['geo_code'].apply(_F.Decode)

    temp = geo_info[geo_info['pos'] != -1]

    res = [x for x in temp['pos'].values]
    X = np.asarray(res)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, random_state=2018).fit(X)
    temp['cluster_id'] = kmeans.labels_

    t = temp.groupby(['UID','cluster_id'])['pos'].count().reset_index().rename(columns={'pos':'cluster_count'})
    c = pd.pivot_table(t,index='UID',columns='cluster_id',values='cluster_count').fillna(0).reset_index()
    # 重命名列
    _Prep = _P.Process()
    c = _Prep.RenameColumns(c,[pri_id],'cluster')
    # 持久化
    feather.write_dataframe(c,data_path + "data/_F_geo.feather")
    return c

# 加入用户的省份 城市 区
def _F_GeoCode(encode_type="LabelEncode",n=3):
    if os.path.exists(data_path + "data/_F_geo_code.feather"):
        df = feather.read_dataframe(data_path + "data/_F_geo_code.feather")
        return df
    # 取每个用户经常活跃的topN geo_code
    geo_info = pd.concat((op_info[[pri_id,'geo_code','day']],trans_info[[pri_id,'geo_code','day']]))
    temp = _F.TopNGeo_code(geo_info,pri_id,'day',n)
    # 编码
    _Prep = _P.Process()
    temp =_Prep.CatColConvert(temp,pri_id,encode_type)
    # 持久化
    feather.write_dataframe(temp,data_path + "data/_F_geo_code.feather")
    return temp

# 加入TopN的nunique
def _F_nunique(n=3):
    if os.path.exists(data_path + "data/_F_nunique.feather"):
        df = feather.read_dataframe(data_path + "data/_F_nunique.feather")
        return df

    temp = pd.DataFrame()
    temp[pri_id] = pd.concat((_train[pri_id],_test[pri_id]))

    # 增加天和月份的片
    df = pd.concat((op_info[[pri_id,'day']],trans_info[[pri_id,'day']]))
    month = _F.Day2Month(df,'day')
    temp = pd.merge(temp,_F.TopN_col_distinct_count(month[[pri_id,'month','day']],pri_id,'month','day',n),on=pri_id,how='left')

    df = pd.concat((op_info[[pri_id,'time','day']],trans_info[[pri_id,'time','day']]))
    df['day_period'] = df['time'].apply(_F.TimeInterval)
    temp = pd.merge(temp,_F.TopN_col_distinct_count(df[[pri_id,'day_period','day']],pri_id,'day_period','day',n))


    for col in ['geo_code','ip1','ip1_sub','mac1','merchant','ip2','ip2_sub','mode','mac2','os','channel','trans_type1','trans_type2','code1','code2','market_code','market_type','device_code1','device_code2','device_code3','wifi']:
        if col in ['merchant','channel','trans_type1','trans_type2','code1','code2','market_code','market_type']:
            df = trans_info[[pri_id,col,'day']]
        elif col in ['ip2','ip2_sub','mode','mac2','os','wifi']:
            df = op_info[[pri_id,col,'day']]
        else:
            df = pd.concat((op_info[[pri_id,col,'day']],trans_info[[pri_id,col,'day']]))
        temp = pd.merge(temp,_F.TopN_col_distinct_count(df,pri_id,col,'day',n),on=pri_id,how='left')

    feather.write_dataframe(temp,data_path + "data/_F_nunique.feather")
    return temp


def _F_nunique_ratio(n=3):

    if os.path.exists(data_path + "data/_F_nunique_ratio.feather"):
        df = feather.read_dataframe(data_path + "data/_F_nunique_ratio.feather")
        return df

    temp = pd.DataFrame()
    temp[pri_id] = pd.concat((_train[pri_id],_test[pri_id]))


    # 增加天和月份的片
    df = pd.concat((op_info[[pri_id,'day']],trans_info[[pri_id,'day']]))
    month = _F.Day2Month(df,'day')
    temp = pd.merge(temp,_F.TopN_col_distinct_ratio(month[[pri_id,'month','day']],pri_id,'month','day',n),on=pri_id,how='left')

    df = pd.concat((op_info[[pri_id,'time','day']],trans_info[[pri_id,'time','day']]))
    df['day_period'] = df['time'].apply(_F.TimeInterval)
    temp = pd.merge(temp,_F.TopN_col_distinct_ratio(df[[pri_id,'day_period','day']],pri_id,'day_period','day',n))

    # topN ratio
    for col in ['geo_code','ip1','ip1_sub','mac1','merchant','ip2','ip2_sub','mode','mac2','os','channel','trans_type1','trans_type2','code1','code2','market_code','market_type','device_code1','device_code2','device_code3','wifi']:
        if col in ['merchant','channel','trans_type1','trans_type2','code1','code2','market_code','market_type']:
            df = trans_info[[pri_id,col,'day']]
        elif col in ['ip2','ip2_sub','mode','mac2','os','wifi']:
            df = op_info[[pri_id,col,'day']]
        else:
            df = pd.concat((op_info[[pri_id,col,'day']],trans_info[[pri_id,col,'day']]))
        temp = pd.merge(temp,_F.TopN_col_distinct_ratio(df,pri_id,col,'day',n),on=pri_id,how='left')

    feather.write_dataframe(temp,data_path + "data/_F_nunique_ratio.feather")
    return temp

# market_code
def _F_market():

    if os.path.exists(data_path + "data/_F_market_ratio.feather"):
        df = feather.read_dataframe(data_path + "data/_F_market_ratio.feather")
        return df

    # 计算每个用户消费次数以及营销活动占比
    temp = trans_info.groupby('UID')['day'].count().reset_index().rename(columns={'day':'trans_count'})
    temp = temp.merge(trans_info.groupby('UID')['market_code'].agg({'market_count':'count'}),on='UID',how='left')
    temp['market_ratio'] = temp['market_count'] / temp['trans_count']

    feather.write_dataframe(temp,data_path + "data/_F_market_ratio.feather")
    return temp

# 加入危险label占比
def _F_LabelCount():
    if os.path.exists(data_path + "data/_F_d_label_count.feather"):
        df = feather.read_dataframe(data_path + "data/_F_d_label_count.feather")
        return df

    temp = pd.DataFrame()
    temp[pri_id] = pd.concat((_train[pri_id],_test[pri_id]))

    # trans_type1
    temp = temp.merge(_F.DangerousLabel(trans_info,pri_id,'trans_type1','ced62357ad496957',0.8),on=pri_id,how='left')

    # trans_type2
    temp = temp.merge(_F.DangerousLabel(trans_info,pri_id,'trans_type2',104,0.8),on=pri_id,how='left')

    # amt_src1
    temp = temp.merge(_F.DangerousLabel(trans_info,pri_id,'amt_src1','c4ec9622cf5c6e55',0.8),on=pri_id,how='left')

    # amt_src2
    temp = temp.merge(_F.DangerousLabel(trans_info,pri_id,'amt_src2','c4ec9622cf5c6e55',0.8),on=pri_id,how='left')
    feather.write_dataframe(temp,data_path + "data/_F_d_label_count.feather")

    return temp


# 验证结果
def Metric(model='lgb',encode_type="LabelEncode"):
    df = Base_Process(encode_type)
    # 加一个地理位置聚类
    df = pd.merge(df,_F_Clsuter_Geo(),on=pri_id,how='left')


    # 加一个用户活跃TopN省份 市 区
    temp = _F_GeoCode(n=1)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 加入distinct的统计
    temp = _F_nunique(3)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 加入ratio的统计
    temp = _F_nunique_ratio(3)
    df = pd.merge(df,temp,on=pri_id,how='left')
    
    # 加入market count 和 参与活动率
    temp = _F_market()
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 填充缺失值
    df = df.fillna(0)

    # 构造训练集
    df = pd.merge(_train,df,on=pri_id,how='left')


    cols = [col for col in df.columns if col != 'y']
    Train = df[cols]
    Label = df['y']

    # 加入nn的特征
    # _mlp_features = pd.read_csv(data_path + "data/_F_mlp_features.csv")
    # print(_mlp_features.shape)
    # print(Train.shape)
    # Train = pd.merge(Train,_mlp_features,on=pri_id,how='left')
    data = _M.N_Split(Train,Label,n=5)
    auc = 0
    score = 0
    for _d in data:
        _train_x = _d[0]
        _train_y = _d[1]
        _test_x = _d[2]
        _test_y = _d[3]
        features = [col for col in _train_x.columns if col != pri_id and col != 'y']
        # 计算auc和本题评分标准
        pred = _M.Predict(_train_x,features,_train_y,_test_x,model)
        _t_auc = roc_auc_score(_test_y,pred)
        auc += _t_auc
        print("auc is %.4f" % _t_auc)
        # 本题评价标准
        _t_score = _M.tpr_weight_funtion(_test_y,pred)
        score += _t_score
        print("score is %.4f" % _t_score)
    auc /= 5
    score /= 5
    print("avg auc is %.4f" % auc)
    print("avg score is %.4f" % score)


# 添加神经网络预测方法
def NeuralNetwork(encode_type="LabelEncode"):
    df = Base_Process(encode_type)
    # 加一个地理位置聚类
    df = pd.merge(df,_F_Clsuter_Geo(),on=pri_id,how='left')


    # 加一个用户活跃TopN省份 市 区
    temp = _F_GeoCode(n=1)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 加入distinct的统计
    temp = _F_nunique(3)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 加入ratio的统计
    temp = _F_nunique_ratio(3)
    df = pd.merge(df,temp,on=pri_id,how='left')

    _Train = pd.merge(_train,df,on=pri_id,how='left').fillna(0)
    _Test = pd.merge(_test,df,on=pri_id,how='left').fillna(0)
    features = [col for col in _Train.columns if col != pri_id and col != 'y']

    _Label = _Train['y']

    # 数据输入和结构构造
    from keras.models import Sequential
    model = Sequential()
    from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
    from keras import backend as K
    import tensorflow as tf
    import itertools

    shape = _Train.shape
    # 卷积层
    # model.add(Conv2D(64, (3,3), activation='relu', input_shape = (shape[0],shape[1],1)))
    # # 池化层
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # # 全连接层 (设置输出层的维度)
    # model.add(Dense(256, activation='relu'))
    # # dropout层
    # model.add(Dropout(0.5))
    # # 最后全连接层，输出概率
    # model.add(Dense(1, activation='sigmoid'))

    # MLP
    # print(shape)
    model.add(Dense(64, input_dim=402, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # 编译(后面要换成自己定义的评价函数)

    # 本题评分标准
    def tpr_weight_funtion(y_true,y_pred):


        # batch_size, n_elems = y_pred.shape[0],y_pred.shape[1]
        # idxs = list(itertools.permutations(range(n_elems)))
        # permutations = tf.gather(y_pred, idxs, axis=-1)  # Shape=(batch_size, n_permutations, n_elems)

        d = pd.DataFrame()
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # d['prob'] = permutations.eval(session=sess)

        d['prob'] = list(K.eval(y_pred))
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

    def AUC(y_true,y_pred):
        not_y_pred = np.logical_not(y_pred)
        y_int1 = y_true*y_pred
        y_int0 = np.logical_not(y_true)*not_y_pred
        TP = np.sum(y_pred*y_int1)
        FP = np.sum(y_pred)-TP
        TN = np.sum(not_y_pred*y_int0)
        FN = np.sum(not_y_pred)-TN
        TPR = np.float(TP)/(TP+FN)
        FPR = np.float(FP)/(FP+TN)
        return ((1+TPR-FPR)/2)

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=[AUC])

    # 训练 (batch_size 每次迭代选择的样本数)

    res = pd.DataFrame()
    res[pri_id] = _Test[pri_id]
    _K_Train = pd.DataFrame()
    _KTrain = pd.DataFrame()
    _KTrain[pri_id] = _Train[pri_id]
    # 需要将输入归一化
    _Train,_Test = _M.Normalize(_Train[features],_Test[features])

    from sklearn.model_selection import StratifiedKFold
    # 将_Train分成5份，5折之后求平均
    skf = StratifiedKFold(n_splits=5)
    pred = np.zeros((_Test.shape[0],1))

    for train,test in skf.split(_Train,_Label):

        model.fit(_Train.iloc[train],_Label.iloc[train],epochs=50,batch_size=128)
        # 连接剩下的一折和test
        temp = model.predict(_Test)
        pred += np.asarray(temp)
        _K_T = pd.DataFrame()
        _K_T[pri_id] = _KTrain.iloc[test][pri_id]
        _K_T['mlp'] = model.predict(_Train.iloc[test])
        _K_Train = pd.concat((_K_Train,_K_T))

    pred /= 5
    # 全连接的输出
    res['mlp'] = pred
    res = pd.concat((_K_Train,res))
    res.to_csv(data_path + "data/_F_mlp_features.csv",index=False)

# 用NN编码op和trans
def Encoder(target_col,encode_dim=5):
    # data_path = "./"
    # mode和trans_type1，merchant
    if target_col in ['mode']:
        data = op_info
    if target_col in ['trans_typd1','merchant']:
        data = trans_info
    keys,matrix = _F.simple_countVectorizer(data,data_path,pri_id,target_col)
    res = _F.AutoEncoder(matrix,encode_dim)
    temp = pd.DataFrame()
    temp[pri_id] = keys
    for i in range(encode_dim):
        temp[target_col + "_encode_" + str(i)] = res[:,i]
    temp.to_csv(data_path + "data/_F_encode_%s.csv" % target_col,index=False)
    return temp


# 序列编码
def SeqEncoder(df,pri_id,col,timesteps=5,encode_dim=5):
    mode = df[[pri_id,col,'day','time']]
    # 获取每个用户的操作序列，按时间
    # 按照天数降序排列，按照时间升序排列，然后按照UID聚集
    mode.sort_values(by=['day','time'],ascending=[False,True],inplace=True)
    temp = mode.groupby(pri_id)[col].apply(list).reset_index().rename(columns={col:'%s_seq' % col})
    temp['%s_seq_len' % col] = temp['%s_seq' % col].apply(lambda x : len(x))
    ob_len = int(temp['%s_seq_len' % col].max() * 3.0 / 5.0)
    # 改为timesteps的倍数
    while ob_len % timesteps != 0:
        ob_len += 1
    # 不足长度的序列进行填充(选取长度为最长长度的5/3)
    def Padding(x):

        if len(x) < ob_len:
            # 填充(Nan)
            x += ['Nan' for i in range(ob_len - len(x))]
        else:
            # 截取
            x = x[:ob_len]
        return x
    temp['%s_seq' % col] = temp['%s_seq' % col].apply(Padding)
    # 编码
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    mode['encoder'] = le.fit_transform(mode[col])
    # 建立字典
    seq_dic = {}
    encoder = mode[[col,'encoder']].drop_duplicates()
    for key,val in zip(encoder[col],encoder['encoder']):
        seq_dic[key] = val
    seq_dic['Nan'] = mode['encoder'].max() + 1
    # 对序列进行转化
    temp['%s_seq' % col] = temp['%s_seq' % col].apply(lambda x : [seq_dic[i] for i in x])
    matrix = np.asarray([i for i in temp['%s_seq' % col]])

    # 全连接层编码
    # res = _F.AutoEncoder(matrix,encode_dim)

    # LSTM编码
    res = _F.AutoEncoderLSTM(matrix,5,timesteps)
    # 保存结果
    results = pd.DataFrame()
    results[pri_id] = temp[pri_id]
    for i in range(encode_dim):
        results[col + "_encode_" + str(i)] = res[:,i]
    results.to_csv(data_path + "data/_F_seq_encode_%s.csv" % col,index=False)
    return results


# 拼接后序列编码
def ConcatSeq(df,pri_id,cols):
    for i in range(len(cols),1,len(cols)):
        df[cols[0]] += "_" + df[cols[i]]
    df.rename(columns={cols[0]:"%s_target_seq" % cols[0]},inplace=True)
    SeqEncoder(df,pri_id,"%s_target_seq" % cols[0],5)

# 预测
def Predict(model='lgb',encode_type="LabelEncode"):
    df = Base_Process(encode_type)
    # 加一个地理位置聚类
    df = pd.merge(df,_F_Clsuter_Geo(),on=pri_id,how='left')


    # 加一个用户活跃TopN省份 市 区
    temp = _F_GeoCode(n=1)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 加入distinct的统计
    temp = _F_nunique(3)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 加入ratio的统计
    temp = _F_nunique_ratio(3)
    df = pd.merge(df,temp,on=pri_id,how='left')

    # 填充缺失值
    df = df.fillna(0)
    _Train = pd.merge(_train,df,on=pri_id,how='left')
    _Test = pd.merge(_test,df,on=pri_id,how='left')

    features = [col for col in _Train.columns if col != pri_id and col != 'y']

    _Label = _Train['y']
    res = pd.DataFrame()
    res[pri_id] = _Test[pri_id]
    pred = _M.Predict(_Train,features,_Label,_Test,model)
    print(pred)
    res['Tag'] = pred
    res.to_csv("resutls.csv",index=False)

# data = Process()
Metric("lgb",'LabelEncode')
Predict()

# 用神经网络预测值作为lgb的特征
# NeuralNetwork()

# 对特征列（count）使用神经网络进行编码，然后编码作为lgb的输入特征
# Encoder('mode')

# 对特征列使用神经网络进行序列编码，然后编码作为lgb的输入特征
# SeqEncoder(op_info,pri_id,'mode',5,5)

# 对merchant, trans_type1, amt_src1拼接做序列编码
# ConcatSeq(trans_info,pri_id,['merchant','trans_type1','amt_src1'])
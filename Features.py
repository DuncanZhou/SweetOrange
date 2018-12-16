#!/usr/bin/python
#-*-coding:utf-8-*-
'''@Date:18-10-13'''
'''@Time:下午10:01'''
'''@author: Duncan'''
import pandas as pd
import math
from math import radians, cos, sin, asin, sqrt
import geohash
import PreProcess as _Prep
from keras.layers import Dense,Input,LSTM,RepeatVector

from keras.models import Model
import Models as _M
import os
import scipy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import gc
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential


# 设置种子
np.random.seed(2018)

# 聚合统计 类别特征列相应数量
def GetCategroicalCount(df,pri_id,col):
    temp = df.groupby(pri_id)[col].count().reset_index().rename(columns={col:"%s_count" % col})
    return temp

# 聚合统计 数值列均值
def GetValAvg(df,pri_id,col):
    temp = df.groupby(pri_id)[col].mean().reset_index().rename(columns={col:"%s_avg" % col})
    return temp

# 聚合统计 数值列方差
def GetValVar(df,pri_id,col):
    temp = df.groupby(pri_id)[col].var().reset_index().rename(columns={col:"%s_var" % col})
    return temp

# 聚合统计 数值列最大值和最小值
def GetValMaxMin(df,pri_id,col):
    temp1 = df.groupby(pri_id)[col].max().reset_index().rename(columns={col:"%s_max" % col})
    temp2 = df.groupby(pri_id)[col].min().reset_index().rename(columns={col:"%s_min" % col})
    temp = pd.merge(temp1,temp2,on=pri_id)
    return temp

# 某列求和
def GetValSum(df,pri_id,col):
    temp = df.groupby(pri_id)[col].sum().reset_index().rename(columns={col:"%s_sum" % col})
    return temp

# 某列为空统计
def GetValNaCount(df,pri_id,col,other_col):
    temp = df[df[col].isna()]
    temp = temp.groupby(pri_id)[other_col].count().reset_index().rename(columns={other_col:"%s_Nacount" % col})
    return temp

# 特征列转化（行转列）
def CatRowsToCols(df,pri_id,col,other_col):
    temp = df.groupby([pri_id,col])[other_col].count().reset_index().rename(columns={other_col:"count"})
    temp = pd.DataFrame(temp.pivot_table(index=pri_id,columns=col,values="count").reset_index())
    _P = _Prep.Process()
    temp = _P.RenameColumns(temp,[pri_id],col)
    return temp

# day切分为上 中  下旬
def Day2Month(df,col):
    def Convert(x):
        if x < 10:
            return 0
        elif x < 20:
            return 1
        else:
            return 2
    df['month'] = df[col].apply(Convert)
    return df

# 统计月份时间
def MonthCount(df,pri_id):
    temp = Day2Month(df,'day')
    temp = CatRowsToCols(temp,pri_id,'month','day')
    return temp

# 操作时间进行切片 （一天 切片 按不同的时间粒度 （1-4h））
def TimeInterval(x,period=2):
    # 时间片粒度为 1h
    x = str(x).split(":")[0]
    return math.ceil(int(x) / period)

# 统计每个主键对应某个参数不同的个数
def GetCount(df,pri_id,target_col,other_col):
    temp = df.groupby([pri_id,target_col])[other_col].count().reset_index()
    temp = temp.groupby(pri_id)[target_col].count().reset_index().rename(columns={target_col:'%s_count' % target_col})
    return temp

# 计算每个用户有多少个不同的经纬度
def CountWS(df):
    temp = df[['UID','geo_code','day']].groupby(['UID','geo_code'])['day'].count().reset_index().rename(columns={"day":"times"})
    temp = temp.groupby('UID')['geo_code'].nunique().reset_index().rename(columns={'geo_code':'geo_times'})
    return temp

# 统计每个用户最经常出现的经纬度
def PositionWS(df):
    temp = df[['UID','geo_code','day']].groupby(['UID','geo_code'])['day'].count().reset_index().rename(columns={"day":"geo_times"})
    idx = temp.groupby('UID')['geo_times'].idxmax()
    temp = temp.iloc[idx][['UID','geo_code']]
    temp['pos'] = temp['geo_code'].apply(Decode)
    temp['w'] = temp['pos'].apply(lambda x : x[0])
    temp['s'] = temp['pos'].apply(lambda x : x[1])
    return temp[['UID','w','s']]

# 经纬度计算
def haversine(p1, p2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1 = p1[0]
    lat1 = p1[1]
    lon2 = p2[0]
    lat2 = p2[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

# 计算距离
def Decode(x):
    try:
        return list(geohash.decode(x))
    except:
        return -1

def GeoRCode(location):
    # 传入字符串，经纬度用，分割
    import requests
    amap_web_key = '44658b956283c80965a20de0d75c7b39'
    pos_rencode_url = 'https://restapi.amap.com/v3/geocode/regeo'
    params = {"key":amap_web_key,'location':location,'output':'json'}
    response = requests.get(pos_rencode_url,params)
    data = response.json()
    return data['regeocode']['addressComponent']['country'],data['regeocode']['addressComponent']['province'],data['regeocode']['addressComponent']['city'],data['regeocode']['addressComponent']['district']

# 统计某列危险值，并对原dataframe做count,并定义筛选阈值
def CountDangerous(df,tags,pri_id,label,col,other_col,threshold):
    df = pd.merge(df,tags,on=pri_id,how='left')
    temp = df[df[label] == 1].groupby([col,label])[other_col].count().reset_index().rename(columns={'time':"%s_count" % col})
    temp = temp[temp["%s_count" % col] >= threshold]
    d_vals = temp[col].values
    df['%s_d_label' % col] = df[col].apply(lambda x : 1 if x in d_vals else 0)
    temp = df.groupby(pri_id)['%s_d_label' % col].sum().reset_index()
    return temp

# # 取每个用户经常活跃的topN geo_code
def TopNGeo_code(df,pri_id,other_col,n=3):
    geo_code = pd.read_csv('geo_code.csv')
    temp = df.groupby([pri_id,'geo_code'])[other_col].count().reset_index().rename(columns={other_col:'geo_times_count'})
    temp['row_number'] = temp['geo_times_count'].groupby(temp['UID']).rank(ascending=False,method='first')
    temp = temp[temp['row_number'] <= 3]
    res = pd.DataFrame()
    res['UID'] = df['UID'].unique()

    for i in range(1,n+1):
        temp['top%d_geo' % i] = temp[temp['row_number'] == i]['geo_code']
        temp['top%d_geo_count' % i] = temp[temp['row_number'] == i]['geo_times_count']

    # merge
    for i in range(1,n+1):
        res = pd.merge(res,temp[~temp['top%d_geo_count' % i].isnull()][['UID','top%d_geo' % i,'top%d_geo_count' % i]],on='UID',how='left')

    for i in range(1,n+1):
        res.rename(columns={'top%d_geo' % i:'geo_code'},inplace=True)

        res = pd.merge(res,geo_code[['geo_code','country','prov','city','district']],on='geo_code',how='left')
        cols = ['geo_code','country','prov','city','district']
        new_cols = {}
        for col in cols:
            new_cols[col] = ("top%d" % i) + "_" + col
        res.rename(columns=new_cols,inplace=True)
    # 填充缺失值
    for i in range(1,n+1):
        res['top%d_geo_code' % i] = res['top%d_geo_code' % i].fillna('-1')
        res['top%d_country' % i] = res['top%d_country' % i].fillna('-1')
        res['top%d_prov' % i] = res['top%d_prov' % i].fillna('-1')
        res['top%d_city' % i] = res['top%d_city' % i].fillna('-1')
        res['top%d_district' % i] = res['top%d_district' % i].fillna('-1')
        res['top%d_geo_count' % i] = res['top%d_geo_count' % i].fillna(-1)
    # 只加省 市 区，不加count数量
    cols = [col for col in res.columns if col.find("count") == -1]
    return res[cols]

# 统计每个目标列的nuique，然后取pri_id的前top3的
def TopN_col_distinct_count(df,pri_id,col,other_col,n=3):
    temp = df.groupby([pri_id,col])[other_col].count().reset_index().rename(columns={other_col:'%s_times_count' % col})
    temp['row_number'] = temp['%s_times_count' % col].groupby(temp[pri_id]).rank(ascending=False,method='first')
    temp = temp[temp['row_number'] <= n]
    # 统计nunique
    count = df.groupby(col)[pri_id].nunique().reset_index().rename(columns={pri_id:'%s_count_distinct' % col})
    temp = pd.merge(temp[[pri_id,col,'row_number']],count,on=col,how='left')

    res = pd.DataFrame()
    res[pri_id] = df[pri_id].unique()

    for i in range(1,n+1):
        #     temp['top%d_geo' % i] = temp[temp['row_number'] == i]['geo_code']
        temp['top%d_%s_count' % (i,col)] = temp[temp['row_number'] == i]['%s_count_distinct' % col]
    # merge
    for i in range(1,n+1):
        res = pd.merge(res,temp[~temp['top%d_%s_count' % (i,col)].isnull()][[pri_id,'top%d_%s_count' % (i,col)]],on=pri_id,how='left')

    res = res.fillna(0)
    return res

# 统计每个目标列的nuique，然后取pri_id的前top3的
def TopN_col_distinct_ratio(df,pri_id,col,other_col,n=3):
    temp = df.groupby([pri_id,col])[other_col].count().reset_index().rename(columns={other_col:'%s_times_count' % col})
    temp['row_number'] = temp['%s_times_count' % col].groupby(temp[pri_id]).rank(ascending=False,method='first')
    temp = temp[temp['row_number'] <= n]
    # 统计nunique
    count = df.groupby(col)[pri_id].agg({'%s_count_distinct' % col : "nunique",'%s_count' % col : "count"})
    count['%s_distinct_ratio' % col] = count['%s_count_distinct' % col] / count['%s_count' % col]

    temp = pd.merge(temp[[pri_id,col,'row_number']],count,on=col,how='left')

    res = pd.DataFrame()
    res[pri_id] = df[pri_id].unique()

    for i in range(1,n+1):
        #     temp['top%d_geo' % i] = temp[temp['row_number'] == i]['geo_code']
        temp['top%d_%s_distinct_ratio' % (i,col)] = temp[temp['row_number'] == i]['%s_distinct_ratio' % col]
    # merge
    for i in range(1,n+1):
        res = pd.merge(res,temp[~temp['top%d_%s_distinct_ratio' % (i,col)].isnull()][[pri_id,'top%d_%s_distinct_ratio' % (i,col)]],on=pri_id,how='left')

    res = res.fillna(0)
    return res

# 相关性较高的特征列的特征值筛选
def DangerousLabel(df,pri_id,col,val,threshold):
    temp = df.groupby(pri_id)[col].agg({'%s_count' % col:'count'})
    # temp
    temp = temp.merge(df[df[col] == val].groupby(pri_id)[col].count().reset_index().rename(columns={col:'d_%s' % col}),on='UID',how='left')
    # temp = temp.merge(label,on=pri_id,how='left')
    temp['d_%s_label' % col] = temp['d_%s' % col].div(temp['%s_count' % col]).apply(lambda x : 1 if x > threshold else 0)
    temp[temp['d_%s_label' % col] == 1]
    return temp[[pri_id,'d_%s_label' % col]]


# nn自编码输出编码层
def AutoEncoder(data,output_dim=5):

    '''

    :param data: 数据
    :param output_dim: 压缩到多少维
    :return: 返回encoder层的结果
    '''

    # 归一化
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data.A)
    input_dim = data.shape[1]

    # 占位
    input_data = Input(shape=(input_dim,))
    autoencoder = Sequential()
    #
    autoencoder.add(Dense(output_dim,input_shape=(input_dim,),activation='relu'))
    autoencoder.add(Dense(input_dim,activation='sigmoid'))

    # encoder model
    encoder_layer = autoencoder.layers[0]
    encoder = Model(input_data, encoder_layer(input_data))
    print(encoder.summary())

    # 编译，训练
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(data, data,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True)
    # 输出编码层结果
    encoder = encoder.predict(data)
    return encoder


# lstm自编码器
def AutoEncoderLSTM(data,timesteps=5,output_dim=5):

    # LSTM的输入一定是三维的
    # 归一化
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    # 转换成3维的
    data = np.asarray(data)

    data = data.reshape(data.shape[0],timesteps,-1)
    print(data.shape)
    latent_dim = output_dim
    input_dim = data.shape[-1]
    inputs = Input(batch_shape=(None,timesteps,input_dim))


    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    print(encoder.summary())
    # 编译，训练

    sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    print("编译完成")
    sequence_autoencoder.fit(data, data,
                    epochs=50,
                    batch_size=128,
                    shuffle=True)
    # 输出编码层结果
    encoder = encoder.predict(data)
    return encoder

def simple_countVectorizer(df, dataPath, cate1, cate2, min_df=2):
    if not os.path.exists(os.path.join(dataPath, 'cache/')):
        os.mkdir(os.path.join(dataPath, 'cache/'))
    sentence_file = os.path.join(dataPath,
                                 'cache/%s_%s_mindf_%d_Simple_CountVectorizer_vector.npz' % (cate1, cate2, min_df))
    cate1s_file = os.path.join(dataPath,
                               'cache/%s_%s_mindf_%d_Simple_CountVectorizer_cate1s.npz' % (cate1, cate2, min_df))
    if (not os.path.exists(sentence_file)) or (not os.path.exists(cate1s_file)):
        # 保证文件内容的是一致的 不能有错位
        if os.path.exists(sentence_file):
            os.remove(sentence_file)
        if os.path.exists(cate1s_file):
            os.remove(cate1s_file)

        mapping = {}
        for sample in df[[cate1, cate2]].astype(str).values:
            mapping.setdefault(sample[0], []).append(sample[1])
        cate1s = list(mapping.keys())
        cate2_as_sentence = [' '.join(mapping[cate]) for cate in cate1s]
        del mapping;
        gc.collect()

        cate2_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',
                                          min_df=min_df).fit_transform(cate2_as_sentence)
        scipy.sparse.save_npz(sentence_file, cate2_as_matrix)
        np.savez(cate1s_file, cate1s=cate1s)
        return cate1s, cate2_as_matrix
    else:
        cate2_as_matrix = scipy.sparse.load_npz(sentence_file)
        cate1s = np.load(cate1s_file)['cate1s']
        return list(cate1s), cate2_as_matrix

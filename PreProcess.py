#!/usr/bin/python
#-*-coding:utf-8-*-
'''@Date:18-10-13'''
'''@Time:下午10:01'''
'''@author: Duncan'''

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import datetime

class Process:
    def __init__(self, naT=0.9):
        '''
        :param naT: 缺失值去除比例（默认去除控制超过0.9比例的）
        '''
        self.naT = naT

    # 去除空值比例大于naT的列
    def remove_na(self, df):
        cols = df.columns
        to_remove = [col for col in cols if len(df[~df[col].isna()]) >= len(df[col]) * self.naT]
        df.drop(columns=to_remove, inplace=True)
        return df

    # 转换特征列
    def CatColConvert(self,df,pri_id,type='LabelEncode'):
        cat_cols = [col for col in df.columns if df[col].dtypes == object and col != pri_id]
        if type == 'LabelEncode':
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col].astype(str))
        elif type == 'OneHotEncode':
            enc = OneHotEncoder()
            for col in cat_cols:
                df[col] = enc.fit_transform(df[col].astype(str))
        return df

    # 计算天数
    def days(self,x):
        return x.days

    # 转换字符串年月 到 日期
    def YMToTime(self,x):
        return pd.to_datetime(x,format="%Y年%m月").strftime("%Y/%m")

    def ConvertYM(self,df,col):
        df[col] = df[col].apply(self.YMToTime())
        df['%s_days' % col] = pd.to_datetime(datetime.date.today().strftime("%Y/%m/%d")) - pd.to_datetime(df[col])
        df['%s_days' % col] = df['%s_days' % col].apply(self.days)
        df.drop(columns={col},inplace=True)

    # 对列名重新编码
    def RenameColumns(self,df,exclue_columns,param):
        cols = [col for col in df.columns if col not in exclue_columns]
        new_cols = {}

        for col in cols:
            new_cols[col] = param + "_" + str(col)
        df.rename(columns=new_cols,inplace=True)
        return df
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import random
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBRegressor

import sys
sys.path.append("./my/")
from my import myUtils

def model():
    warnings.filterwarnings("ignore")
    
    def read_data(data,group):
        if group == 1:
            data = data[data['毒品数量'] == 0]
            data = data[data['判刑月份'] < 36]
        elif group == 2:
            data = data[data['毒品数量'] > 0]
            data = data[data['毒品数量'] < 7]
            data = data[data['判刑月份'] < 36]
        elif group == 3:
            data = data[data['毒品数量'] >= 7]
            data = data[data['毒品数量'] < 10]
            data = data[data['判刑月份'] >= 36]
            data = data[data['判刑月份'] < 84]
        elif group == 4:
            data = data[data['毒品数量'] >= 10]
            data = data[data['毒品数量'] < 50]
            data = data[data['判刑月份'] >= 84]
        else:
            raise Exception()
        return data
    
    def train(num,X_train,y_train,X_test,y_test):
        if num == 1:
            model = tree.DecisionTreeRegressor()
        elif num == 2:
            model = svm.SVR()
        elif num == 3:
            model = LinearRegression()
        elif num == 4:
            model = neighbors.KNeighborsRegressor(n_neighbors=11)
        elif num == 5:
            model = ensemble.RandomForestRegressor(n_estimators=100)
        elif num == 6:
            model = ensemble.AdaBoostRegressor(n_estimators=100)
        elif num == 7:
            model = ensemble.GradientBoostingRegressor(n_estimators=100)
        elif num == 8:
            model = ensemble.BaggingRegressor()
        elif num == 9:
            model = ExtraTreeRegressor()
        model.fit(X_train, y_train)
        pred=model.predict(X_test)
        return rmse(np.array(y_test), np.array(pred)),r_squared(np.array(y_test),np.array(pred))
    
    def rmse(pre, tar):
        '''它这里的pre是标签。'''
        return np.sqrt(((pre - tar) ** 2).mean())

    def r_squared(pre,tar):
        return 1-(((pre - tar) ** 2).mean()/np.var(pre))

    def mean(list):
        return sum(list) / len(list)
    
    t = pd.read_csv('./dataset/all_feature.csv', sep=',',index_col=0)
    
    for group in range(1,5):
        data=read_data(t,group=group)
        
        periodList = []
        for record in data.values:
            periodList.append(record[-1])
        print(min(periodList))
        print(max(periodList))
        print(sum(periodList)/len(periodList))
        myUtils.write_log("###第%d组：min(%d) max(%d) mean(%f)"%(group,min(periodList),max(periodList),sum(periodList)/len(periodList)),log_path="./features.log")
        
        tmp_data=pd.DataFrame({"id":list(data.index)},index=data.index)
        
        train_data=data[tmp_data["id"].apply(lambda x:x[:5]=="train")]
        test_data=data[tmp_data["id"].apply(lambda x:x[:4]=="test")]
        
        # train
        train_yArr = train_data['判刑月份'].values
        deletedItems = ['姓名', '毒品种类', '毒品数量', '判刑月份']
        for di in deletedItems:
            train_data = train_data.drop([di], axis=1)
        train_xArr = train_data.values
        train_xArr = np.mat(train_xArr)
        train_yArr = np.mat(train_yArr)
        train_yArr = train_yArr.reshape(-1, 1)
        print(train_xArr.shape)
        print(train_yArr.shape)
        # test
        test_yArr = test_data['判刑月份'].values
        deletedItems = ['姓名', '毒品种类', '毒品数量', '判刑月份']
        for di in deletedItems:
            test_data = test_data.drop([di], axis=1)
        test_xArr = test_data.values
        test_xArr = np.mat(test_xArr)
        test_yArr = np.mat(test_yArr)
        test_yArr = test_yArr.reshape(-1, 1)
        print(test_xArr.shape)
        print(test_yArr.shape)
        # data
        X_train=train_xArr
        X_test=test_xArr
        y_train=train_yArr
        y_test=test_yArr
        # fit
        for i in range(1,10):
            res_rmse,res_r2=train(i,X_train,y_train,X_test,y_test)
            log='第%d组,第%d个算法,train(%d),test(%d):rmse %.3f,r2 %.3f'%(group,i,X_train.shape[0],X_test.shape[0],res_rmse,res_r2)
            print(log)
            myUtils.write_log(log,log_path="./features.log")
        
        # xgboost回归
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train')]
        xgb_pars = {'min_child_weight': 50, 'eta': 0.01, 'colsample_bytree': 0.4, 'max_depth': 10,
                    'subsample': 0.8, 'lambda': 1., 'booster': 'gbtree', 'silent': 1, 'tree_method':'gpu_hist', 'max_bin': 16, 'gpu_id':0,
                    'eval_metric': 'rmse', 'objective': 'reg:linear'}
        model = xgb.train(xgb_pars, dtrain, 5000, watchlist, early_stopping_rounds=200,maximize=False, verbose_eval=20)

        pred=model.predict(dtest)
        res_rmse=rmse(np.array(y_test), np.array(pred))
        res_r2=r_squared(np.array(y_test), np.array(pred))
        log='第%d组,第10个算法,train(%d),test(%d):rmse %.3f,r2 %.3f'%(group,X_train.shape[0],X_test.shape[0],res_rmse,res_r2)
        print(log)
        myUtils.write_log(log,log_path="./features.log")

if __name__ == "__main__":
    model()
    pass
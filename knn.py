import warnings
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

def knn(path):
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
    
    def del_useless(data,del_columns=[]):
        return data.drop(del_columns,axis=1)
    
    def get_labels(data,columns_name):
        return data[columns_name].values
    
    def rmse(pre, tar):
        '''它这里的pre是标签。'''
        return np.sqrt(((pre - tar) ** 2).mean())
    
    def r_squared(pre,tar):
        return 1-(((pre - tar) ** 2).mean()/np.var(pre))
    
    def choose_model(choose,n_neighbors=15):
        if choose==1:
            model_name=neighbors.KNeighborsRegressor.__name__
            model=neighbors.KNeighborsRegressor(n_neighbors)
        elif choose==2:
            model_name=LinearRegression.__name__
            model=LinearRegression()
        else:
            raise Exception()
        return model_name,model
    
    whole_data = pd.read_csv(path,index_col=0)
    
    for group in range(1,5):
        print("处理第 %d 组数据"%group)
        
        data=read_data(whole_data,group=group)
        tmp_data=pd.DataFrame({"id":list(data.index)},index=data.index)
        
        # train
        train=data[tmp_data["id"].apply(lambda x:x[:5]=="train")]
        test=data[tmp_data["id"].apply(lambda x:x[:4]=="test")]
        
        # train labels
        train_yArr = get_labels(train,"判刑月份")
        # del useless columns of data
        train=del_useless(train,['姓名', '毒品数量', '判刑月份'])
        # variables
        train_xArr = train.values
        # convert type list to type np.mat
        train_xMat = np.mat(train_xArr)
        train_yMat = np.mat(train_yArr)
        train_yMat = train_yMat.reshape(-1, 1)
        
        # test labels
        test_yArr = get_labels(test,"判刑月份")
        test=del_useless(test,['姓名', '毒品数量', '判刑月份'])
        test_xArr = test.values
        test_xMat = np.mat(test_xArr)
        test_yMat = np.mat(test_yArr)
        test_yMat = test_yMat.reshape(-1,1)
        
        # knn model
        n_neighbors=15
        model_name,model=choose_model(choose=1,n_neighbors=n_neighbors)
        
        # data
        X_train=train_xMat
        X_test=test_xMat
        y_train=train_yMat
        y_test=test_yMat
        # train
        model.fit(X_train, y_train)
        # test
        X_predict=model.predict(X_test)
        # loss
        rmse_loss=rmse(np.array(y_test), np.array(X_predict))
        r2_loss=r_squared(np.array(y_test),np.array(X_predict))
        
        log="%s: model(%s) group(%d) train(%d) test(%d) n_neighbors(%d) rmse(%.3f) r2(%.3f)"%(str(path),model_name,group,X_train.shape[0],X_test.shape[0],n_neighbors,rmse_loss,r2_loss)
        print(log)
        with open("./results.log","a",encoding="utf-8") as f:
            f.write(log+"\n")
    return


if __name__ == "__main__":
    knn("./bert/data/nofuzzy.averpool.csv")
    knn("./bert/data/fuzzy.averpool.csv")
    knn("./xlnet/data/nofuzzy.averpool.csv")
    knn("./xlnet/data/fuzzy.averpool.csv")
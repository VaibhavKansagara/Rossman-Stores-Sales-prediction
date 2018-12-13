# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

# loading packages
# basic + dates 
import numpy as np
import pandas as pd
from pandas import datetime
import pickle

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
    #%matplotlib inline

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

class rossmann(object):
    def __init__(self, train, test,store):
        self.train = train
        self.test = test
        self.store=store
        self.train_store=None
        self.test_store=None
        self.close_store=None
        self.result = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.store_ind = None
        self.Y_pred = None
        self.__rforest=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=20, min_samples_split=10, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, n_jobs=4, verbose=0)
        #self.__xgb=
    def convert_date_train(self):
        self.train['Year']=self.train.index.year
        self.train['Month']=self.train.index.month
        self.train['Day']=self.train.index.day
        self.train['WeekOfYear']=self.train.index.weekofyear
    def convert_date_test(self):
        self.test['Year']=self.test.index.year
        self.test['Month']=self.test.index.month
        self.test['Day']=self.test.index.day
        self.test['WeekOfYear']=self.test.index.weekofyear
    def remove_zero_sales_train(self):
        self.train=self.train[(self.train.Sales!=0) & (self.train.Open!=0)]
    def remove_closed_store_test(self):
        test=self.test
        test=test[test.Open!=0]
        self.close_store=test["Id"][test.Open==0]
        self.test=test
    def fill_null_store(self):
        store=self.store
        store['CompetitionOpenSinceMonth'].fillna(store.CompetitionOpenSinceMonth.min(),inplace=True)
        store['CompetitionOpenSinceYear'].fillna(store.CompetitionOpenSinceYear.min(),inplace=True)
        store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(),inplace=True)
        store['Promo2SinceWeek'].fillna(0,inplace=True)
        store['Promo2SinceYear'].fillna(0,inplace=True)
        store['PromoInterval'].fillna(0,inplace=True)
        self.store=store
    def merge_with_store(self):
        train=self.train
        test=self.test
        train_store=pd.merge(train,store,how="inner",on="Store")
        test_store=pd.merge(test,store,how="inner",on="Store")
        self.train_store=train_store
        self.test_store=test_store
    def add_competionopen(self):
        train_store=self.train_store
        test_store=self.test_store
        train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + (train_store.Month - train_store.CompetitionOpenSinceMonth)
        test_store['CompetitionOpen'] = 12 * (test_store.Year - test_store.CompetitionOpenSinceYear) + (test_store.Month - test_store.CompetitionOpenSinceMonth)
        self.train_store=train_store
        self.test_store=test_store
    def delete_redundant_feature(self):
        train_store=self.train_store
        test_store=self.test_store
        del train_store['CompetitionOpenSinceYear']
        del train_store['CompetitionOpenSinceMonth']
        del test_store['CompetitionOpenSinceYear']
        del test_store['CompetitionOpenSinceMonth']
        self.train_store=train_store
        self.test_store=test_store
    def convert_to_numeric(self):
        train_store=self.train_store
        test_store=self.test_store
        train_store.StateHoliday = train_store.StateHoliday.map({"0":0,"a": 1, "b": 1, "c": 1})
        test_store.StateHoliday = test_store.StateHoliday.map({"0":0,"a": 1, "b": 1, "c": 1})

        train_store['StateHoliday'] = train_store['StateHoliday'].astype('category')
        train_store['Assortment'] = train_store['Assortment'].astype('category')
        train_store['StoreType'] = train_store['StoreType'].astype('category')
        train_store['PromoInterval']= train_store['PromoInterval'].astype('category')

        test_store['StateHoliday'] = test_store['StateHoliday'].astype('category')
        test_store['Assortment'] = test_store['Assortment'].astype('category')
        test_store['StoreType'] = test_store['StoreType'].astype('category')
        test_store['PromoInterval']= test_store['PromoInterval'].astype('category')

        train_store[['StateHoliday', 'StoreType', 'Assortment']] = train_store[['StateHoliday', 'StoreType', 'Assortment']].apply(lambda x: x.cat.codes)
        test_store[['StateHoliday', 'StoreType', 'Assortment']] = test_store[['StateHoliday', 'StoreType', 'Assortment']].apply(lambda x: x.cat.codes)
    
        train_store=pd.get_dummies(train_store, columns=["Assortment", "StoreType","PromoInterval"], 
                              prefix=["Assortment_", "StoreType_","PromoInteval_"])
        test_store=pd.get_dummies(test_store, columns=["Assortment", "StoreType","PromoInterval"], 
                              prefix=["Assortment_", "StoreType_","PromoInteval_"])

        train_store['Year'] = train_store['Year'].apply(lambda x: int(x))
        train_store['Month'] = train_store['Month'].apply(lambda x: int(x))
        test_store["Year"] = test_store["Year"].apply(lambda x: int(x))
        test_store['Month'] = test_store['Month'].apply(lambda x: int(x))

        train_store.info()
        self.train_store=train_store
        self.test_store=test_store
    def make_x_y(self):
        train_store = self.train_store
        test_store = self.test_store
        self.result=pd.Series()
        self.X_train=train_store.drop(['Sales','Store',"Id"],axis=1)
        self.Y_train = np.log(train_store.Sales)
        self.X_test=test_store.copy()
        self.store_ind=self.X_test["Id"]
        self.X_test.drop(['Id','Store'],axis=1,inplace=True)
        
    def randomforest(self):
        # estimator=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=20, min_samples_split=10, min_samples_leaf=1, 
        #                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, n_jobs=4, verbose=0)
        # Y_train.fillna(Y_train.mean())
        self.__rforest.fit(self.X_train,self.Y_train)
        self.Y_pred=self.__rforest.predict(self.X_test)

    def get_output(self):
        result = self.result
        result=result.append(pd.Series(self.Y_pred,index=self.store_ind))
        result=result.append(pd.Series(0,index=self.close_store))
        result = pd.DataFrame({ "Id": result.index, "Sales": np.exp(result.values)})
        result.to_csv('result_new.csv', index=False)
        #self.result=result
    def execute(self):
        self.convert_date_train()
        self.convert_date_test()
        self.remove_zero_sales_train()
        self.remove_closed_store_test()
        self.fill_null_store()
        self.merge_with_store()
        self.add_competionopen()
        self.delete_redundant_feature()
        self.convert_to_numeric()
        self.make_x_y()
if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv", 
                    parse_dates = True,index_col = 'Date')
    test = pd.read_csv("../input/test.csv",
                    parse_dates = True,index_col = 'Date')

    # additional store data
    store = pd.read_csv("../input/store.csv")
    
    model = rossmann(train,test,store)
    model.execute()
    model.randomforest()
    model.get_output()
    with open('Model_RFR.sav','wb') as f:
        pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)

# print test_store.columns
# for column in train_store.columns:
#     print train_store[column].dtype,
        
    

# Any results you write to the current directory are saved as output.
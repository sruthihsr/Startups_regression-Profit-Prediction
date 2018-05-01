# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:23:52 2018

@author: sruth
"""
#Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

os.chdir(r'C:\Users\sruth\OneDrive\Desktop\ml')
data= pd.read_csv('50_Startups.csv')
print(data.head(10))
data.describe()
data.shape
type(data)
print(data.columns)
print(np.median(data['R&D Spend']))
iv=data.iloc[:,:-1]
dv=data.iloc[:,-1]

# Label Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEnc=LabelEncoder()
iv['State']=LabelEnc.fit_transform(iv['State'])

#One Hot Encoding
ohc=OneHotEncoder(categorical_features=[3])
iv=ohc.fit_transform(iv).toarray()

#Test Train Split at 20% test and 80% train
from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)

#Linear Regresion
from sklearn.linear_model import LinearRegression
lin_Reg= LinearRegression()
lin_Reg.fit(iv_train,dv_train)
lin_Reg.predict(iv_test)


# Making  a comparison data frame
results_train=pd.DataFrame([])
results_train=results_train.append(pd.DataFrame(dv_train),ignore_index=True)
results_train.columns=['ActualProfit']
results_train['LinearPredictions']=lin_Reg.predict(iv_train)

results_test=pd.DataFrame([])
results_test=results_test.append(pd.DataFrame(dv_test),ignore_index=True)
results_test.columns=['ActualProfit']
results_test['LinearPredictions']=lin_Reg.predict(iv_test)

results_train['mae']=results_train['ActualProfit']-results_train['LinearPredictions']
#  Decision Tree

def mae(actual,predicted):
    return round(np.mean(abs(actual-predicted)),2)


#functin mape
def mape(actual,predicted):
    print (round(np.mean((abs(actual-predicted)))/actual*100,2))


def rmse(actual,predicted):
    return round(np.sqrt(np.mean((actual-predicted)**2)),2)

print('For Linear Regression model')
print('Mean Average error for train is:',mae(results_train['ActualProfit'],results_train['LinearPredictions']))
print('Mean Average Percentagr error for train is:',mape(results_train['ActualProfit'],results_train['LinearPredictions']))
print('Mean RMSE for train is:',rmse(results_train['ActualProfit'],results_train['LinearPredictions']))

print('Mean Average error for test is:',mae(results_test['ActualProfit'],results_test['LinearPredictions']))
print('Mean Average Percentagr error for test is:',mape(results_test['ActualProfit'],results_test['LinearPredictions']))
print('Mean RMSE for test is:',rmse(results_test['ActualProfit'],results_test['LinearPredictions']))


#Decisin tree
from sklearn.tree import DecisionTreeRegressor
dt_regressor=DecisionTreeRegressor()
dt_regressor.fit(iv_train,dv_train)



results_train['DT_Predict']=dt_regressor.predict(iv_train)
results_test['DT_Predict']=dt_regressor.predict(iv_test)

print('For DecisionTree model')
print('Mean Average error for test is:',mae(results_test['ActualProfit'],results_test['DT_Predict']))
print('Mean Average Percentagr error for test is:',mape(results_test['ActualProfit'],results_test['DT_Predict']))
print('Mean RMSE for test is:',rmse(results_test['ActualProfit'],results_test['DT_Predict']))

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_regressor=RandomForestRegressor()
rf_regressor.fit(iv_train,dv_train)

results_train['RF_Predict']=rf_regressor.predict(iv_train)
results_test['RF_Predict']=rf_regressor.predict(iv_test)

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas
from sklearn import datasets 
from sklearn.cluster import KMeans 
from sklearn.cluster import Birch
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

from 决策树 import data_train


scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(data_train['Age'].values.reshape(-1, 1))
data_train['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1, 1), age_scale_param)

X = data_train[["Pclass","Sex","SibSp","Parch","Age_scaled"]]
y = data_train['Survived']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=0)  
# 使用KMeans考察线性分类KMeans的预测能力 

y_pred = KMeans(n_clusters=4, random_state=0).fit_predict(X_train) 
print(y_pred)

class chj_data(object):
    def __init__(self,data,target):
        self.data=data
        self.target=target

def chj_load_file(fdata,ftarget):
    res=chj_data(fdata,ftarget)
    return res
print(X_train)
print(X_train["Pclass"])
iris = chj_load_file(X_train,y_pred)
X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(iris.data)
plt.figure(figsize=(12, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.colorbar()
plt.show()

y_Birch = Birch(n_clusters = None).fit_predict(X_train)
iris_Birch = chj_load_file(X_train,y_Birch)
X_tsne_Birch = TSNE(n_components=2,learning_rate=100).fit_transform(iris_Birch.data)
plt.figure(figsize=(12, 6))
plt.scatter(X_tsne_Birch[:, 0], X_tsne_Birch[:, 1], c=iris_Birch.target)
plt.colorbar()
plt.show()


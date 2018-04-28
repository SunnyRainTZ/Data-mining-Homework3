import pandas 
import sys
import os  
import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO
from logistic回归 import set_missing_ages
from logistic回归 import set_cabin_type
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

data_train = pandas.read_csv('./train.csv')

data_train, rfr = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)

data_train.loc[data_train['Sex'] == 'male','Sex'] = 1
data_train.loc[data_train['Sex'] == 'female','Sex'] = 0  
 
dummies_Cabin = pandas.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pandas.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pandas.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pandas.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pandas.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

X = data_train[["Pclass","Sex","SibSp","Parch"]]
y = data_train['Survived']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

#决策树
dt=tree.DecisionTreeClassifier()
dt=dt.fit(X_train,y_train)

# 从sklearn.metrics导入classification_report。
from sklearn.metrics import classification_report
print(dt.score(X_test, y_test))
y_predict = dt.predict(X_test)
print(classification_report(y_predict, y_test, target_names = ['died', 'survived']))

feature_name = ["Pclass","Sex","SibSp","Parch"]
target_name = ["Survived","Died"]
dot_data = StringIO()
tree.export_graphviz(dt,out_file = dot_data,feature_names=feature_name,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("决策树.pdf")


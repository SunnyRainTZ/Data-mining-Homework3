import pandas 
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import tree
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
titanic = pandas.read_csv('./train.csv')  


def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    return df, rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(titanic)
data_train = set_cabin_type(titanic)

#特征因子化
dummies_Cabin = pandas.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pandas.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pandas.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pandas.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pandas.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

#数值数据归一化处理
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)


# 将标称型数据转化为数值型数据
df.loc[df['Sex'] == 'male','Sex'] = 1
df.loc[df['Sex'] == 'female','Sex'] = 0  

X = df[["Age","SibSp","Parch","Fare","Pclass","Sex"]]
y = df['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X_train, y_train)
print(clf)

predictions = clf.predict(X_test)
# 从sklearn.metrics导入classification_report。
from sklearn.metrics import classification_report
# 输出预测准确性。
print(clf.score(X_test, y_test))
print(classification_report(predictions, y_test, target_names = ['died', 'survived']))

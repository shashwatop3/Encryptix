import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier



pd.set_option('display.max_columns', None)
df = pd.read_csv('Churn_Modelling.csv')
y = df['Exited']
df.drop(columns=['Exited'], inplace=True)
print(df.info())
#print(df.describe())
#print(df.isna().sum())

sc = StandardScaler()
for col in df.columns[df.dtypes != 'object']:
    df[col] = sc.fit_transform(df[[col]])
print(df.describe())

le = LabelEncoder()
for col in df.columns[df.dtypes == 'object']:
    df[col] = le.fit_transform(df[col])
print(df.head())
#
# sns.heatmap(df.corr(), annot=True)
# plt.show()\
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
# model_log = LogisticRegression()
# model_log.fit(x_train, y_train)
# print(model_log.score(x_test, y_test))

model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
print(model_rf.score(x_test, y_test))
y_pred = model_rf.predict(x_test)
print(classification_report(y_test, y_pred))

model_xg = XGBClassifier()
model_xg.fit(x_train, y_train)
y_predxg = model_xg.predict(x_test)
print(classification_report(y_test, y_predxg))

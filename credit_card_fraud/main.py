import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBModel
import pickle

df = pd.read_csv("fraudTrain.csv")
dt = pd.read_csv("fraudTest.csv")
# print(df['is_fraud'].value_counts())
# print(dt['is_fraud'].sum()/dt['is_fraud'].count())
y = df['is_fraud']
dt_y = dt['is_fraud']
df = df[['amt', 'job', 'category']]
dt = dt[['amt', 'job', 'category']]
le = LabelEncoder()
# Fit the LabelEncoder on the combined 'job' columns of the training and test data
le.fit(pd.concat([df['job'], dt['job']]))

# Transform the 'job' columns
df['job'] = le.transform(df['job'])
dt['job'] = le.transform(dt['job'])
df['category'] = le.fit_transform(df['category'])
dt['category'] = le.transform(dt['category'])
sc = StandardScaler()
df['amt'] = sc.fit_transform(df[['amt']])
dt['amt'] = sc.transform(dt[['amt']])

X_train, X_cv, y_train, y_cv = train_test_split(df, y, test_size=0.3, random_state=42)
X_train, y_train = RandomOverSampler(sampling_strategy='minority').fit_resample(X_train, y_train)
# model = LogisticRegression(max_iter=100)
# model.fit(X_train, y_train)
# class_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
# model = XGBModel(scale_pos_weight=class_weight)
model = RandomForestClassifier(verbose=2) # best
model.fit(X_train, y_train)
y_cv_pred = model.predict(X_cv)
# y_cv_pred = (y_cv_pred > 0.5).astype(int)
y_pred = model.predict(dt)
print(classification_report(y_cv, y_cv_pred))
print(f1_score(y_cv, y_cv_pred))
print(f1_score(dt_y, y_pred))
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

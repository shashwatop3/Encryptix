import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

with open("encoded_data.pkl", 'rb') as file:
    df = pickle.load(file)

df['encoded'] = df['encoded'].apply(lambda x: np.array(x))
df['encoded_labels'] = df['encoded_labels'].apply(lambda x: np.array(x))
X = np.stack(df['encoded'].tolist())
le = LabelEncoder()
y = le.fit_transform(df['genre'])
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
# print(x_train[0].shape)

# model = LogisticRegression(solver='lbfgs', max_iter=500, penalty='l2', C=0.1, multi_class='multinomial')
# model.fit(x_train, y_train)
# report_train = classification_report(y_train, model.predict(x_train))
# report = classification_report(y_test, model.predict(x_test))
# print(report, report_train)
# predict = model.predict(x_test[:5])
# print(le.inverse_transform(predict), '>>', le.inverse_transform(y_test[:5]))

# joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
with open("encoded_data_test.pkl", 'rb') as file:
    dt = pickle.load(file)
dt['encoded'] = dt['encoded'].apply(lambda x: np.array(x))
X_test = np.stack(dt['encoded'].tolist())
Y_test = le.transform(dt['genre'])

report = classification_report(Y_test, model.predict(X_test))
print(report)
# Output:
#
#               precision    recall  f1-score   support
#
#            0       0.38      0.17      0.23      2629
#            1       0.58      0.19      0.29      1180
#            2       0.37      0.08      0.14      1550
#            3       0.49      0.09      0.15       996
#            4       0.40      0.01      0.02       529
#            5       0.47      0.50      0.49     14893
#            6       0.30      0.05      0.08      1010
#            7       0.65      0.81      0.72     26192
#            8       0.51      0.72      0.60     27225
#            9       0.36      0.10      0.15      1567
#           10       0.45      0.06      0.10       645
#           11       0.85      0.40      0.54       387
#           12       0.28      0.01      0.03       486
#           13       0.56      0.43      0.49      4408
#           14       0.60      0.37      0.46      1462
#           15       0.56      0.03      0.06       553
#           16       0.66      0.05      0.10       637
#           17       0.50      0.02      0.04       362
#           18       0.44      0.22      0.29      1767
#           19       0.35      0.10      0.15      1344
#           20       0.50      0.20      0.29      1293
#           21       0.38      0.31      0.34     10145
#           22       0.58      0.20      0.30       863
#           23       0.63      0.18      0.28       782
#           24       0.32      0.14      0.20      3181
#           25       0.70      0.08      0.14       264
#           26       0.82      0.58      0.68      2064
#
#     accuracy                           0.54    108414
#    macro avg       0.51      0.23      0.27    108414
# weighted avg       0.52      0.54      0.50    108414

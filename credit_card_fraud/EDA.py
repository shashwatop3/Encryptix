import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("fraudTrain.csv")
# lst_cols = ["merchant", "category", "gender", "state", "city", "job"]
# for col in lst_cols:
#     sns.histplot(data=df, x=col, hue="is_fraud")
#     plt.show()
# print(df.isna().sum())

sns.histplot(data=df, x="merchant", hue="is_fraud")
plt.show()




# Data columns (total 23 columns):
#  #   Column                 Non-Null Count    Dtype
# ---  ------                 --------------    -----
#  0   Unnamed: 0             1296675 non-null  int64
#  1   trans_date_trans_time  1296675 non-null  object
#  2   cc_num                 1296675 non-null  int64
#  3   merchant               1296675 non-null  object
#  4   category               1296675 non-null  object
#  5   amt                    1296675 non-null  float64
#  6   first                  1296675 non-null  object
#  7   last                   1296675 non-null  object
#  8   gender                 1296675 non-null  object
#  9   street                 1296675 non-null  object
#  10  city                   1296675 non-null  object
#  11  state                  1296675 non-null  object
#  12  zip                    1296675 non-null  int64
#  13  lat                    1296675 non-null  float64
#  14  long                   1296675 non-null  float64
#  15  city_pop               1296675 non-null  int64
#  16  job                    1296675 non-null  object
#  17  dob                    1296675 non-null  object
#  18  trans_num              1296675 non-null  object
#  19  unix_time              1296675 non-null  int64
#  20  merch_lat              1296675 non-null  float64
#  21  merch_long             1296675 non-null  float64
#  22  is_fraud               1296675 non-null  int64
# dtypes: float64(5), int64(6), object(12)
# memory usage: 227.5+ MB
# None

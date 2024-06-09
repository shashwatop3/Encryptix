import numpy as np
import pandas as pd
import pickle
def encode(review):
    word_index=[]
    for word in word_lst['word']:
        word_index.append(review.count(word))
    return word_index
def encode_labels(genre):
    return np.argwhere(labels==genre)
word_lst = pd.read_csv("word_vec.csv")
# removing first 30 row as it is very common and after 2500 as it is very rare
word_lst = word_lst[30:1500]
lst_rev = []
genre=[]
with open("train_data.txt", 'r') as file:
    reviews = file.readlines()
    for review in reviews:
        line = review.replace(",", "").replace(".", '').replace("?", "").strip().split(":::")
        lst_rev.append(line[3].split())
        genre.append(line[2])
df = pd.DataFrame({'review':lst_rev,'genre':genre})
labels = df['genre'].unique()

df['encoded'] = df['review'].apply(lambda x: encode(x))
df['encoded_labels'] = df['genre'].apply(lambda x: encode_labels(x))
with open("encoded_data.pkl", 'wb') as file:
    pickle.dump(df, file)
with open("labels.pkl", 'wb') as file:
    pickle.dump(labels, file)
print(df.shape)

# test data set
with open("test_data_solution.txt", 'r') as file:
    reviews = file.readlines()
    for review in reviews:
        line = review.replace(",", "").replace(".", '').replace("?", "").strip().split(":::")
        lst_rev.append(line[3].split())
        genre.append(line[2])
dt = pd.DataFrame({'review':lst_rev,'genre':genre})

dt['encoded'] = dt['review'].apply(lambda x: encode(x))
dt['encoded_labels'] = dt['genre'].apply(lambda x: encode_labels(x))
with open("encoded_data_test.pkl", 'wb') as file:
    pickle.dump(dt, file)

print(dt.shape)



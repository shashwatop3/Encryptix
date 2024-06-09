import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

with open("word_index.json", "r") as file:
    word_index = json.load(file)
word_vec = pd.DataFrame(word_index.items(), columns=['word', 'index'])
word_vec.set_index(keys="word", inplace=True)
word_vec.drop(columns=['index'], inplace=True)
word_vec['count'] = 0
print(word_vec.head())
lst_rev = []
genre = []
with open("train_data.txt", 'r') as file:
    reviews = file.readlines()
    for review in reviews:
        line = review.replace(",", "").replace(".", '').replace("?", "").strip().split(":::")
        lst_rev.append(line[3])
        genre.append(line[2])
df = pd.DataFrame({'review': lst_rev, 'genre': genre})
for review in lst_rev[:25000]:
    words = review.split(" ")
    for word in words:
        if word in word_vec.index:
            word_vec.loc[word] += 1
word_vec.sort_values(by='count', ascending=False, inplace=True)
word_vec.to_csv("word_vec.csv")
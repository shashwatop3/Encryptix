import pandas as pd
import json
lst_rev = []
genre=[]
with open("train_data.txt", 'r') as file:
    reviews = file.readlines()
    for review in reviews:
        line = review.replace(",", "").replace(".", '').replace("?", "").strip().split(":::")
        lst_rev.append(line[3])
        genre.append(line[2])
df = pd.DataFrame({'review':lst_rev,'genre':genre})
print(df.head())
print(df['genre'].value_counts())
word_index = {}
i = 0
# for line in lst_rev:
#     words = line.split(" ")
#     for word in words:
#         if word not in word_index:
#             word_index[word] = i
#             i+=1
# with open("word_index.json", 'w') as json_file:
#     json.dump(word_index, json_file)
labels = df['genre'].unique()
print(labels)




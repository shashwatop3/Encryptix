
# Importing necessary libraries
import numpy as np # For linear algebra
import pandas as pd # For data manipulation
from sklearn.model_selection import train_test_split # For splitting the data into training and testing sets
from xgboost import XGBModel # For using XGBoost model
from sklearn.ensemble import RandomForestClassifier # For using Random Forest Classifier
from sklearn.metrics import classification_report, f1_score, accuracy_score # For model evaluation

# Reading the data from a CSV file
df = pd.read_csv("spam.csv", encoding='latin-1')

# Preprocessing the messages in the dataset
msgs = []
for line in df.v2:
    # Removing punctuation and converting to lower case
    line = line.replace(',', '').replace('?', '').replace('.', '').replace('!', '').lower().strip()
    # Splitting the line into words and appending to the list
    msgs.append(line.split())

# Printing the first 3 messages after preprocessing
print(msgs[:3])

# Converting the spam/ham labels to binary format
msg_type = []
for typ in df.v1:
    if(typ == 'spam'):
        msg_type.append(1)
    else:
        msg_type.append(0)

# Converting the list to a numpy array
msg_type = np.array(msg_type)

# Counting the number of spam messages
msg_type.sum()

# Counting the total number of messages
len(msg_type)

# Creating a list of unique words in the messages
word_index=[]
for words in msgs:
    for word in words:
        word_index.append(word)
word_index = set(word_index)
word_index = pd.DataFrame(word_index, columns=['word'])

# Creating separate lists for spam and ham messages
spam = [msgs[i] for i in range(len(df)) if df.v1[i] == 'spam']
ham = [msgs[i] for i in range(len(df)) if df.v1[i] == 'ham']

# Printing the first message
print(msgs[:1])

# Counting the frequency of each word in spam messages
word_count_spam = {}
for msg in spam:
    for word in msg:
        word_count_spam[word] = word_count_spam.get(word, 0)+1

# Counting the frequency of each word in ham messages
word_count_ham = {}
for msg in ham:
    for word in msg:
        word_count_ham[word] = word_count_ham.get(word, 0)+1

# Counting the frequency of each word in all messages
word_count = {}
for msg in msgs:
    for word in msg:
        word_count[word] = word_count.get(word, 0)+1

# Converting the dictionaries to dataframes
word_count_spam = pd.DataFrame(word_count_spam.items(), columns=['word', 'count'])
word_count_ham = pd.DataFrame(word_count_ham.items(), columns=['word', 'count'])
word_count = pd.DataFrame(word_count.items(), columns = ['word', 'count'])

# Sorting the dataframe by count in descending order
word_count.sort_values(by='count', inplace=True, ascending=False)

# Printing the top 5 words
print(word_count[:5])

# Printing the 10th to 40th words in spam and ham messages
print(word_count_spam[10:40], word_count_ham[10:40])

# Selecting the top 1500 words as features for the model
word_index = word_count['word'][:1500]

# Printing the top 5 words
print(word_index[:5])

# Function to convert a message to a vector
def encoder(msg):
    word_vec = []
    for word in word_index:
        word_vec.append(msg.count(word))
    return word_vec

# Converting all messages to vectors
x = []
for lst in msgs:
    x.append(encoder(lst))
x = np.array(x)
y = msg_type

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Training a Random Forest Classifier
forest_model = RandomForestClassifier()
forest_model.fit(x, y)

# Predicting the labels for the test set
y_pred = forest_model.predict(x_test)

# Calculating the F1 score
f1_score(y_pred, y_test)

# Printing the accuracy score
print(accuracy_score(y_pred, y_test))

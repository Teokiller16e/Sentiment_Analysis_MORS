# import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import platform
print(platform.python_version())

import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk import pos_tag
from gensim.models import KeyedVectors

import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, concatenate, SpatialDropout1D, Dropout, LSTM, Embedding, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import DistilBertTokenizer, TFDistilBertModel
import contractions


# Functions :
# 1) DeContraction of revies :
def decontracted(phrasis):
    # specific
    phrase = contractions.fix(phrasis)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", phrase)
    phrase = re.sub(r"what's", "what is ", phrase)
    phrase = re.sub(r"\'s", " ", phrase)
    phrase = re.sub(r"\'ve", " have ", phrase)
    phrase = re.sub(r"n't", " not ", phrase)
    phrase = re.sub(r"i'm", "i am ", phrase)
    phrase = re.sub(r"\'re", " are ", phrase)
    phrase = re.sub(r"\'d", " would ", phrase)
    phrase = re.sub(r"\'ll", " will ", phrase)
    phrase = re.sub(r",", " ", phrase)
    phrase = re.sub(r"\.", " ", phrase)
    phrase = re.sub(r"!", " ! ", phrase)
    phrase = re.sub(r"\/", " ", phrase)
    phrase = re.sub(r"\^", " ^ ", phrase)
    phrase = re.sub(r"\+", " + ", phrase)
    phrase = re.sub(r"\-", " - ", phrase)
    phrase = re.sub(r"\=", " = ", phrase)
    phrase = re.sub(r"'", " ", phrase)
    phrase = re.sub(r"(\d+)(k)", r"\g<1>000", phrase)
    phrase = re.sub(r":", " : ", phrase)
    phrase = re.sub(r" e g ", " eg ", phrase)
    phrase = re.sub(r" b g ", " bg ", phrase)
    phrase = re.sub(r" u s ", " american ", phrase)
    phrase = re.sub(r" 0s", "0", phrase)
    phrase = re.sub(r" 9 11 ", "911", phrase)
    phrase = re.sub(r"e - mail", "email", phrase)
    phrase = re.sub(r"j k", "jk", phrase)
    phrase = re.sub(r"\s{2,}", " ", phrase)

    ## Stemming
    # phrase = phrase.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in phrase]
    # phrase = " ".join(stemmed_words)

    return phrase
# 2) Remove punctuations :
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
# 3) Lemmatization  Maybe useless:
def lemmatize_it(sent):
    empty = []
    for word, tag in pos_tag(w_tokenizer.tokenize(sent)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
            empty.append(lemma)
        else:
            lemma = lemmatizer.lemmatize(word, wntag)
            empty.append(lemma)
    return ' '.join(empty)
# 4) Plot Validation Loss:
def plot_loss(history):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'black', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'black', ls='--', linewidth=3.0)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()
# 5) Plot Validation Accuracy :
def plot_accuracy(history):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'black', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'black', ls='--', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()
# 6) Sentence Tokenization
def tokenize_review(review):
    return nltk.sent_tokenize(review)
# 7) Plot the Coefficient Analysis :
def plot_coefficients(top_features):
    #top_features = coef_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], np.abs(top_features['Coefficient']))
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Top 10 Features - Coefficient Analysis')
    plt.show()
# 8) Plot the Coefficient Analysis :
def plot_features(top_features):
    #top_features = feature_importance_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Features - Feature Importance Analysis')
    plt.show()
# 9) Plot the Correlation Analysis:
def plot_correlations(top_features):
    # Plot the top 10 features with the highest absolute correlation coefficients
    #top_10_features = correlation_coeffs.head(10)
    plt.figure(figsize=(10, 8))
    plt.barh(top_features.index, top_features['Absolute_Correlation_Coefficients'])
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficients')
    plt.title('Top 10 Features with Highest Absolute Correlation Coefficients')
    plt.show()
# 10) Plot the statistical analysis of the data:
def plot_statistics(df):
    # Descriptive statistics for 'Ground Service Scores'
    ground_service_stats = df['user_rating'].describe()
    print(ground_service_stats)

    # Histogram for 'Wifi & Connectivity' scores
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='user_rating', bins=10)
    plt.title('Histogram of User rating Scores')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.show()

    # Kernel Density Plot for 'Wifi & Connectivity' scores
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x='user_rating')
    plt.title('Kernel Density Plot of User rating Scores')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.show()

# 11)
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# load the Skytrax dataset into a pandas dataframe
df = pd.read_csv('new_air_reviews.csv')

# Preprocess the text data
# Drop columns that  will not be used :
df.drop(columns=["Unnamed: 0", "name_reviewer", "country_reviewer", "airline_name", "section_being_reviewed", "date_review", "review_title", "verified_trip",
                 "review_count_reviewer", "date_response", "text_response", "Type Of Traveller", "Seat Type", "Route", "Date Flown",
                  "Recommended"], inplace=True)

#df = df.dropna(subset=['Seat Comfort'])
#df = df.dropna(subset=['Cabin Staff Service'])
#df = df.dropna(subset=['Food & Beverages'])
#df = df.dropna(subset=['Inflight Entertainment'])
df = df.dropna(subset=['Ground Service'])
#df = df.dropna(subset=['Wifi & Connectivity'])
#df = df.dropna(subset=['Value For Money'])


#--------------------------------------------------------------------------------------------- Data Analysis ------------------------------------------------------------------------------------------------
#plot_statistics(df)

# See how many items are null or NaN in your dataset's columns :
empty_values = df.isnull().sum()
print(empty_values)
df = df.sample(frac=0.50)

# ----------------------------------------------------------------------------------DATA PRE-PROCESSING --------------------------------------------------------------------------------------------------

# Lowercase the reviews :
df['lower'] = df['review_text'].apply(lambda x: x.lower())

# remove special characters
df['special_chars'] = df['lower'].replace(r"'(http|@)\S+", "")

# Remove digits and words containing digits :
df['non_digits'] = df['special_chars'].apply(lambda x: re.sub('\w*\d\w*', '', x))

# Remove punctuations :
df["rem_punct"] = df['non_digits'].apply(remove_punctuations)

# # Remove stop words :
#stop_words = stopwords.words('english')
#df['stop_words'] = df['rem_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Expanding Contractions in the reviews
df['decontracted'] = df['rem_punct'].apply(lambda x: decontracted(x))

# # Lemmatization & Tokenization :
#df['lem_words'] = df['rem_punct'].apply((lambda x: lemmatize_it(x)))

df.drop(columns=["review_text", "lower", "special_chars", "non_digits", "rem_punct" ], inplace=True)

# ---------------------------------------------------------------- Initializing all possible models that will take part to Multi Output -------------------------------------------------------------------------------------------

# Prepare your data (X_train, X_test, y_train, y_test)
# X_train and X_test contain your input texts, y_train and y_test contain the corresponding regression targets

X = df['decontracted']
Y = df['Ground Service']

print("This is the length of the training data :",len(X))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Get the vocabulary of the tokenizer
vocabulary = tokenizer.get_vocab()

# Create a list to store feature names
feature_names = []

# Iterate over the vocabulary and get the corresponding feature name
for token, token_id in vocabulary.items():
    if token != '[PAD]' and token != '[UNK]':
        decoded_token = tokenizer.decode([token_id])  # Decode token
        feature_names.append(decoded_token)

# Save feature names to a file (e.g., 'feature_names.txt')
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(feature + '\n')

# Load pre-trained BERT model
bert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Tokenize the input texts
X_train_tokens = tokenizer.batch_encode_plus(X_train.tolist(), padding='longest', truncation=True, return_tensors='tf')
X_test_tokens = tokenizer.batch_encode_plus(X_test.tolist(), padding='longest', truncation=True, return_tensors='tf')

# Convert tokenized inputs to TensorFlow tensors
train_input_ids = np.array(X_train_tokens['input_ids'])
train_attention_mask = np.array(X_train_tokens['attention_mask'])
test_input_ids = np.array(X_test_tokens['input_ids'])
test_attention_mask = np.array(X_test_tokens['attention_mask'])

# Define the BERT model architecture:
input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)

bert_output = bert(input_ids, attention_mask=attention_mask)[0]
pooler_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)
output = tf.keras.layers.Dense(1, activation='linear')(pooler_output)

model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])



# Train the model
# with tf.device('/device:GPU:0'):
#     model.fit(
#         [X_train_tokens['input_ids'], X_train_tokens['attention_mask']], y_train,
#         validation_data=([X_test_tokens['input_ids'], X_test_tokens['attention_mask']], y_test),
#         epochs=10, batch_size=16
#     )


# with tf.device('/device:GPU:0'):
#     y_pred = model.predict([X_test_tokens['input_ids'], X_test_tokens['attention_mask']], y_test)


with tf.device('/device:GPU:0'):
  scores = model.evaluate([X_test_tokens['input_ids'], X_test_tokens['attention_mask']], y_test)
print("Model Mean Squared Error: {:.2f}".format(scores[0]))
print("Model Mean Absolute Error: {:.2f}".format(scores[1]))
print("Model R Squared: {:.2f}".format(scores[2]))



# Load the feature names from the file
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Get the model parameters (weights)
model_params = model.get_weights()

# Get the embedding layer weights
embedding_weights = model_params[0]

# Calculate the sum of absolute weights for each feature
embedding_weights_sum = np.sum(np.abs(embedding_weights), axis=1)

# Remove the last two elements from embedding_weights_sum to make lengths equal
print(embedding_weights_sum[:6])
embedding_weights_sum = embedding_weights_sum[:-2]

# Get the vocabulary of the tokenizer
vocabulary = tokenizer.get_vocab()

# Create a list to store feature names
filtered_feature_names = []

# Iterate over the vocabulary and get the corresponding feature name
for token, token_id in vocabulary.items():
    if token != '[PAD]' and token != '[UNK]':
        decoded_token = tokenizer.decode([token_id])  # Decode token
        filtered_feature_names.append(decoded_token)

# Make sure the lengths match
print(len(filtered_feature_names))
print(len(embedding_weights_sum))

# Create a DataFrame to store feature names and importance
feature_importance_df = pd.DataFrame({'Feature': filtered_feature_names, 'Importance': embedding_weights_sum})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Print the top 10 features by importance
print(feature_importance_df.head(20))

# Plot the top 10 features by importance
plot_features(feature_importance_df.head(20))

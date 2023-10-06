import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk import pos_tag
from sklearn.inspection import permutation_importance
import nltk
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from keras.layers import GRU, LSTM
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from keras.layers import Dense, Embedding, Flatten, Conv1D, MaxPooling1D, Input
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, concatenate, SpatialDropout1D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras import backend as K
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
    plt.title('Histogram of Overall User Rating Scores')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.show()

    # Kernel Density Plot for 'Wifi & Connectivity' scores
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x='user_rating')
    plt.title('Kernel Density Plot of Overall User Rating Scores')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.show()
# 11) R squared metric
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load tokenizer and lemmatizer

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# load the Skytrax dataset into a pandas dataframe
df = pd.read_csv('new_air_reviews.csv')

# Preprocess the text data
# Drop columns that  will not be used :
df.drop(columns=["Unnamed: 0", "name_reviewer", "country_reviewer", "airline_name", "section_being_reviewed", "date_review", "review_title", "verified_trip",
                 "review_count_reviewer", "date_response", "text_response", "Type Of Traveller", "Seat Type", "Route", "Date Flown",
                  "Recommended"], inplace=True)

#df = df.dropna(subset=['user_rating'])


#--------------------------------------------------------------------------------------------- Data Analysis ------------------------------------------------------------------------------------------------

plot_statistics(df)

empty_values = df.isnull().sum()
print(empty_values)
#df = df.sample(frac=0.25)

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
stop_words = stopwords.words('english')
df['stop_words'] = df['rem_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Expanding Contractions in the reviews
df['decontracted'] = df['rem_punct'].apply(lambda x: decontracted(x))

# # Lemmatization & Tokenization :
#df['lem_words'] = df['rem_punct'].apply((lambda x: lemmatize_it(x)))

df.drop(columns=["review_text", "lower", "special_chars", "non_digits", "rem_punct" ], inplace=True)

# ---------------------------------------------------------------- Initializing all possible models that will take part to Multi Output -------------------------------------------------------------------------------------------

# Load the GloVe word embeddings
embeddings_index = dict()
word_embeddings = ""
inp = "0"
while ((inp != '1') and  (inp!='2') and  (inp!='3')):

    print("Enter 1 for GloVe pre-trained Word Embeddings \n")
    print("Enter 2 for FastText pre-trained Word Embeddings \n")
    print("Enter 3 for Custom Word2Vec Word Embeddings \n")
    inp = input("Choice :")
    if(inp == '1'):
        word_embeddings = "glove.6B.300d.txt"
    elif (inp == '2'):
        word_embeddings = "cc.en.300.vec"
    elif (inp == '3'):
        word_embeddings = "custom_skytrax_stop_words_300d.txt"
    else:
        print("Not available choice please try again")

with open(word_embeddings, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()

# Prepare the data
X = df['decontracted'].values
y = df['user_rating'].values

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=300)

# Create a weight matrix for words in training docs
embedding_matrix = np.zeros((10000, 300))
for word, index in tokenizer.word_index.items():
    if index > 10000 - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the input shape
input_shape = (300,)
# Define the input tensor
input_tensor = Input(shape=input_shape)
# Add the embedding layer
x = Embedding(10000, 300, weights=[embedding_matrix], input_length=300, trainable=False)(input_tensor)
# Add the LSTM layer
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
# Add a second LSTM layer
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
# Add a dropout layer
x = Dropout(0.5)(x)
# Add a dense layer with ReLU activation
x = Dense(64, activation='relu')(x)
# Add another dropout layer
x = Dropout(0.5)(x)
# Add a dense layer with linear activation
predictions = Dense(1, activation='linear')(x)
# Create the model
model = Model(inputs=input_tensor, outputs=predictions)
# Compile the model with Mean Absolute Error loss and Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r_squared])
# Set up early stopping to prevent overfitting
#early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, file)


with tf.device('/device:GPU:0'):
  model.fit(X_train, y_train, batch_size=32, epochs=10)

with tf.device('/device:GPU:0'):
  scores = model.evaluate(X_test, y_test)
print("Model Mean Squared Error: {:.2f}".format(scores[0]))
print("Model Mean Absolute Error: {:.2f}".format(scores[1]))
print("Model R Squared: {:.2f}".format(scores[2]))


#--------------------------------------------------------------------------------------------- Data Analysis ------------------------------------------------------------------------------------------------------------------

# Get the word index from the tokenizer
word_index = tokenizer.word_index

# Get the embedding layer weights:
embedding_weights = model.layers[1].get_weights()[0]

embedding_weights = np.sum((embedding_weights), axis=1)
# Create a list to store feature names and importance
feature_names = []

# Iterate over the word index and get the corresponding feature name
for word, index in word_index.items():
    if index <= 10000:
        feature_names.append(word)

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': embedding_weights})
coef_df['AbsoluteCoefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('AbsoluteCoefficient', ascending=False)
print(coef_df.head(10))
plot_coefficients(coef_df.head(10))

#----------------------------------------------------------------------------------------- Feature Importance --------------------------------------------------------------------------------
# Calculate the importance as the sum of absolute values of weights for each feature :

importance = np.abs(embedding_weights) / np.sum(np.abs(embedding_weights))

# Create a DataFrame to store feature names and their corresponding importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot the top 10 features by importance
print(feature_importance_df.head(10))
plot_features(feature_importance_df.head(10))

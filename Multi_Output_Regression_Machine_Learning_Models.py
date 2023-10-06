import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection as md
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import re
import string
import matplotlib.pyplot as plt
import nltk
import nltk.tokenize as tkn
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import Ridge
import tensorflow as tf
from nltk.stem.snowball import SnowballStemmer
import contractions
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import MultiTaskElasticNet

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
    sns.histplot(data=df, x='user_rating')
    plt.title('Histogram of user_rating Scores')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.show()

    # Kernel Density Plot for 'Wifi & Connectivity' scores
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x='user_rating')
    plt.title('Kernel Density Plot of user_rating Scores')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.show()

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

#df = df.dropna(subset=['Seat Comfort'])
#df = df.dropna(subset=['Cabin Staff Service'])
df = df.dropna(subset=['Food & Beverages'])
df = df.dropna(subset=['Inflight Entertainment'])
df = df.dropna(subset=['Ground Service'])
df = df.dropna(subset=['Wifi & Connectivity'])
#df = df.dropna(subset=['Value For Money'])


#--------------------------------------------------------------------------------------------- Data Analysis ------------------------------------------------------------------------------------------------
#plot_statistics(df)

# See how many items are null or NaN in your dataset's columns :
empty_values = df.isnull().sum()
print(empty_values)
df = df.sample(frac=0.3)
print(df.shape)

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
# Split the data into features (X) and target scores (Y)
X = df['decontracted']
Y = df[['Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 'Inflight Entertainment',
        'Value For Money', 'Wifi & Connectivity', 'user_rating', 'RecommendationInt']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Initialize the Ridge regressor
#model = svm.SVR(kernel="linear")
model = svm.SVR(kernel='linear')
# Create the multi-output regressor using Ridge regressor
multi_output_ridge = MultiOutputRegressor(model)

# Train the model
multi_output_ridge.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = multi_output_ridge.predict(X_test_tfidf)

# Calculate the metrics for each target score
mse_scores = []
mae_scores = []
r2_scores = []

for i in range(len(Y.columns)):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    print("Metrics for", Y.columns[i])
    print("MSE:", mse)
    print("MAE:", mae)
    print("R-squared:", r2)
    print()

# Print overall metrics
print("Overall Metrics")
print("Mean Squared Error:", np.mean(mse_scores))
print("Mean Absolute Error:", np.mean(mae_scores))
print("R-squared:", np.mean(r2_scores))

# ------------------------------------------------------------------------------ Data Analysis ------------------------------------------------------------------------------------------
# Perform coefficient analysis
coefficient_analysis = []
for i, target_score in enumerate(Y.columns):
    coef = multi_output_ridge.estimators_[i].coef_
    features = vectorizer.get_feature_names_out()
    coefficient_analysis.extend(zip(features, coef))

# Create a dataframe for coefficient analysis
coef_df = pd.DataFrame(coefficient_analysis, columns=['Feature', 'Coefficient'])
coefficients = coef_df['Coefficient']
feature_names = coef_df['Feature']

# Sort the dataframe by absolute coefficient values
coef_df['AbsoluteCoefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('AbsoluteCoefficient', ascending=False)

# Print the top 10 features based on coefficient values
print(coef_df.head(10))

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Feature Importance Analysis :
feature_importance = np.abs(coefficients) / np.sum(np.abs(coefficients))
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sort the dataframe by importance values in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Print the top 10 features with the highest importance values
print(feature_importance_df.head(10))

plot_coefficients(coef_df.head(10))
plot_features(feature_importance_df.head(10))
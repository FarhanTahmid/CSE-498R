
# loading neccesary libraries

import nltk
nltk.download('wordnet')
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF
import pyLDAvis
import numpy as np
import time
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
#from google.colab import drive
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The CSV file is empty")
        return data
    except FileNotFoundError:
        raise FileNotFoundError("The specified file does not exist")
    except Exception as e:
        raise Exception(f"An unknown error occurred: {e}")

def report_missing_values(data):
    """ Checks for missing values"""
    missing_values = data.isnull().sum()
    print("Missing values in each column:")
    for column, missing_value_count in missing_values.items():
        print(f"{column}: {missing_value_count}")

def tokenise_text(data):
    """
    Tokenise the text in the clean_text column
    """
    try:
        # label changed to a float when I imported it, so changing it back
        data['oh_label'] = data['oh_label'].astype(int)
        data['clean_text'] = data['clean_text'].astype(str)

        # Begin by tokenizing the words
        # also lowercase all words for consistency
        data['tokens'] = data['clean_text'].apply(lambda x: [word.lower() for word in x.split()]) #ensures all text is lowercase
        print("Tokenisation successful")
        return data
    except Exception as e:
        print(f"Tokenisation error: {e}")
        return None

def lemmatize_text(data):
    """
    Lemmatises the tesxt data
    """
    try:
        lemm = WordNetLemmatizer() #using the inbuilt lemmatisation function

    # Lemmatize all words
        data['lemmatized'] = data['tokens'].apply(lambda x: [lemm.lemmatize(word) for word in x])
        print("Lemmatisation successful")
        return data
    except Exception as e:
        print(f"An error occurred during lemmatisation: {e}")
        return data

def word_frequency_analysis(data):
    """
    calculates the most common words in the data
    """
    words_1 = data[data.oh_label == 1]['lemmatized']
    words_0 = data[data.oh_label == 0]['lemmatized']

    _1_words = Counter(word for words in words_1 for word in str(words).split())
    _0_words = Counter(word for words in words_0 for word in str(words).split())

    print("Most common words for oh_label = 1:")
    print(_1_words.most_common(50))

    print("Most common words for oh_label = 0:")
    print(_0_words.most_common(50))

# Function to remove numbers from a list of words
def remove_numbers(word_list):
    """
    removees any numbers from the text
    """
    return [word for word in word_list if not bool(re.search(r'\d', word))]




# Function to remove URLs from a list of words
def remove_urls(word_list):
    """
    Removes any URLs from the text
    """
    return [word for word in word_list if not (word.startswith('http') or word.startswith('www'))]

def bag_of_words(data, column_name, max_features=5000):
    """
    Creates a Bag of Words representation of the text data
    """
    # Joining the lemmatised words to form a string, as CountVectorizer requires string input
    data['string_lemmatized'] = data[column_name].apply(' '.join)

    # Creates the CountVectorizer and fits it to the data
    vectorizer = CountVectorizer(max_features=max_features)
    X_bag_words = vectorizer.fit_transform(data['string_lemmatized'])

    # Converts to a DataFrame
    df_bag_words = pd.DataFrame(X_bag_words.toarray(), columns=vectorizer.get_feature_names_out())

    return df_bag_words , vectorizer


def tfidf(data, column_name, max_features=5000):
    """
    Creates a TF-IDF representation of the text data
    """
    try:
        # Joining the lemmatized words to form a string since TfidfVectorizer requires string input
        data['string_lemmatized'] = data[column_name].apply(' '.join)

        # Create the TfidfVectorizer and fit it to the data
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        X_TFIDF = tfidf_vectorizer.fit_transform(data['string_lemmatized'])

        # Convert the result to a DataFrame
        df_TFIDF = pd.DataFrame(X_TFIDF.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        return df_TFIDF, tfidf_vectorizer
    except Exception as e:
        print(f"An error occurred during TF-IDF vectorization: {e}")
        return None


def visualise_topics(data, vectorizer, lda_model):
    """
    Visualiseing topics using pyLDAvis

    """

    # calculate the topic-term distribution
    topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, None]

    # calculate the document-topic distribution
    doc_topic_dists = lda_model.transform(data)

    # document lengths
    doc_lengths = data.sum(axis=1).tolist()

    # frequencies of vocab and terms
    vocab = vectorizer.get_feature_names_out()
    term_frequency = data.sum(axis=0).values.ravel()

    # Visualisation
    vis = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency
    )

    return vis

def assign_topic_to_docs(doc_topic_dists):
    """Assigning the most likely topic to each document"""
    return np.argmax(doc_topic_dists, axis=1)

def calculate_oh_label_proportion(data, assigned_topics):
    """Calculates the proportion of comments with oh_label=1 for each topic.."""
    topic_oh_label_props = {}
    for topic in np.unique(assigned_topics):
        indices = np.where(assigned_topics == topic)
        proportion = data.iloc[indices]['oh_label'].mean()
        topic_oh_label_props[topic] = proportion
    return topic_oh_label_props

# Gets the top n words for each topic
def get_topics(lda_model, vectorizer, top_n=10):
    """top_n words for each topic in the lda"""
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_idx = topic.argsort()[-top_n:][::-1]
        topics.append([feature_names[i] for i in top_features_idx])
    return topics


# Function to get the top N words from each topic
def get_top_words_from_topic(model, feature_names, n_top_words=50):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words

# Function to get contextual keywords for each topic based on hate speech terms

def get_contextual_keywords(texts, hate_terms, assigned_topics, window=5):
    contextual_keywords = {}
    for topic in set(assigned_topics):
        contextual_keywords[topic] = Counter()

    for text, topic in zip(texts, assigned_topics):
        words = text
        for i, word in enumerate(words):
            if word in hate_terms[topic]:
                # Get surrounding words
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                surrounding_words = words[start:i] + words[i + 1:end]

                # Exclude hates speech terms from surrounding_words
                surrounding_words = [w for w in surrounding_words if w not in hate_terms[topic]]

                contextual_keywords[topic].update(surrounding_words)

    return contextual_keywords

def sentiment_analysis(data):
    # Use the Sentiment Intensity Analyzer
    intensity_analyser = SentimentIntensityAnalyzer()

    # Convert the lemmatised words back to strings, needed for VADER to analyse
    data['lemmatized_string'] = data['lemmatized_clean'].apply(' '.join)

    # Apply VADER analysis
    data['sentiment_scores'] = data['lemmatized_string'].apply(lambda x: intensity_analyser.polarity_scores(x)['compound'])

    # Classifying the sentiments
    data['sentiment'] = data['sentiment_scores'].apply(lambda x: 'Positive' if x >= 0.05 else ('Neutral' if x > -0.05 else 'Negative'))

    # Pivot table to analyse the sentiment scores wrt the oh_label
    pivot_table = pd.pivot_table(data, values='sentiment_scores', index=['sentiment'], columns=['oh_label'], aggfunc='count', fill_value=0)

    # Percentage of each sentiment category for each oh_label
    total_counts_per_label = pivot_table.sum(axis=0)
    pivot_table_percentage = (pivot_table / total_counts_per_label) * 100
    pivot_table_percentage = pivot_table_percentage.round(2)

    return data, pivot_table_percentage



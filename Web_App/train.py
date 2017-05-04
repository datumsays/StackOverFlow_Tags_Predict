import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Web_App.settings")

import django
django.setup()

# Load modules
import numpy as np
import pandas as pd

# Load django modules
from prediction.models import Posts
from django.shortcuts import get_object_or_404

# Load Utilities
from operator import itemgetter
from itertools import chain

# Load sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.externals import joblib

# Load Gensim
from gensim import corpora, models, similarities

# Load custom module
from preprocess import *


# VARIABLES
K_TO_CHOOSE = 50
TOP_K_PER_TOPIC = {}

# LOCAL FUNCTIONS

def db_to_series():
    tags = pd.Series(Posts.objects.values_list('tag'))
    content = pd.Series(Posts.objects.values_list('content'))
    return tags, content

def topic_index(series, topic):
    return series.ix[series == topic].index

def corpusCreator(df, labelSeries):
    """ Create a word dictionary
    """
    text = {}
    for topic in labelSeries.unique():
        index = topic_index(labelSeries, topic)
        topicText = []
        for sent in df[index]:
            topicText += sent.split()
        text[topic] = topicText

    return text

def tfidf(bags_of_words):
    """ Fetches the top words in k based on TF-IDF scores.
        Returns a list of words with top K.
    """
    # Fetch id for each word
    idDict = corpora.Dictionary(bags_of_words)

    # Get the reverse key-vaule mapping
    inv_Dict = {v: k for v, k in idDict.items()}

    # Transform tCorpus into vector form
    vCorpus = [idDict.doc2bow(tokens) for tokens in bags_of_words]

    # Fit TFIDF
    tfidf = models.TfidfModel(vCorpus)

    return tfidf, inv_Dict, vCorpus


def tfidf_at_k(tfidf, doc_vector, word_key, k):
    """ Get top k features per topic based on TFIDF """
    top_k_id_scores = sorted(tfidf[doc_vector], key=itemgetter(1), reverse=True)[:k]
    return [word_key[key] for key, word in top_k_id_scores]

if __name__ == "__main__":

    # Create pd series
    tags = db_to_series()[0].map(remove_tuples)
    content = db_to_series()[1].map(remove_tuples)

    # Apply TFIDF
    class_dictionary = corpusCreator(content, tags)
    topics = class_dictionary.keys()
    mod, word_key, vCorpus = tfidf(list(class_dictionary.values()))
    topic_vector = {topic: document for topic, document in zip(tags, content)}

    # Get Features
    topic_k_group = k_number_chooser(tags, K_TO_CHOOSE)
    for tag in tags.unique():
        feats_k_count = topic_k_group[tag]
        TOP_K_PER_TOPIC[tag] = topic_vector[tag][:feats_k_count]

    print(TOP_K_PER_TOPIC)
    feats = ' '.join(TOP_K_PER_TOPIC.values()).split()
    print(feats)
    np.save('features', feats)
    train_bow = GetBowDummies_Array2(content, feats).index_feats_dict()

    # Train Model
    mnb = MultinomialNB()
    mnb.fit(X=train_bow,y=tags)
    joblib.dump(mnb, 'mnb.pkl')

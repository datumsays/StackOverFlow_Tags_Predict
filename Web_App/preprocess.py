
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Web_App.settings")

import django
django.setup()

# Load modules
import numpy as np
import pandas as pd

# Load Utilities
from operator import itemgetter
from itertools import chain
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk, re, string

# Local functions

def remove_tuples(text):
    return text[0]


def k_number_chooser(target_series, k_per_class):
    """ Choose k features per topic based on inverse quanitity
        of document class count and number of k features selected
        per class
    """

    # Get number of vectors
    _len = len(target_series)
    unique_nums = len(target_series.unique())
    total_features = k_per_class * unique_nums

    # Get a pandas groupby with proportions of k features
    frac = target_series.value_counts().map(lambda x: round(1 - float(x) / _len, 2))
    frac_sum = sum(frac)
    new_prop = frac.map(lambda x: round(x / frac_sum * total_features))

    # If the sum of the k features and the series length
    # unequal that randomly add one to one of the class feature
    if sum(new_prop) > total_features:
        plusone = np.random.choice(unique_nums)
        new_prop[plusone] = new_prop[plusone] - (sum(new_prop) - total_features)

    elif sum(new_prop) < total_features:
        plusone = np.random.choice(unique_nums)
        new_prop[plusone] = new_prop[plusone] + (sum(new_prop) - total_features)

    return new_prop.astype('int')


class GetBowDummies_Array2(object):
    """
    Inputs: (1) Series with a text vector (2) Bag of Words for features
    Output: Dataframe with dummy variables indicating whether a feature word is present in a row.

    Examples
    --------

    train = pd.Series(['I dont know','polish dont','fire','healthcare know','healthcare'])
    test  = pd.Series(['I dont know','healthcare know'])
    feats = ['dont','know','healthcare']

    train_bow_dummies = GetBowDummies(train, feats).get_bow_dummies()

    test_bow_dummies = GetBowDummies(test, feats).get_bow_dummies()

    test_bow_dummies
     >> dont  know  healthcare
        0     0     0           0
        1     0     0           0

    """

    # Initialize
    def __init__(self, series, features):
        """
        :param series: A column containing raw text
        :param features: A list of feature words
        """
        features.sort()

        self.series = series
        self.index = self.series.index
        self.features = np.asarray(features)

        # Define dimension
        self.nrows = series.shape[0]
        self.ncols = len(features)
        self.dim = (self.nrows, self.ncols)

    def index_feats_dict(self):
        """
        For every document row, features present in doc
        identified.
        """
        # doc_features_list = []
        zero_matrix = np.zeros(self.dim, np.int)

        for i, doc in enumerate(self.series):
            # Sets for a doc and feature words

            doc_set = set(doc.split())
            feat_set = set(self.features)

            # Shared words between the two sets
            interset_words = np.asarray(list(doc_set.intersection(feat_set)))

            if len(interset_words) != 0:
                ndx = np.searchsorted(self.features, interset_words)
                zero_matrix[i, ndx] = 1
            else:
                continue

        return zero_matrix


# Pre-processing and Cleaning

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    """ Strip out HTML tags """

    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text = soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""


def removePunctuation(x):
    """ Removes punctuation marks """

    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]', r' ', x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("[" + string.punctuation + "]", " ", x)


stops = set(stopwords.words("english"))

def removeStopwords(x):
    """ Removes English stopwords """

    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


from nltk.stem import SnowballStemmer

snow = SnowballStemmer('english')
def stemmer(x):
    return ' '.join([snow.stem(word) for word in x.split()])


def is_digit(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def numeric_replacer(x):
    new_set = []
    for word in x.split():
        if is_digit(word):
            new_set.append('digitstring')
        else:
            new_set.append(word)

    return ' '.join(new_set)


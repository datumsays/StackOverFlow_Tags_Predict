"""
The module includes utility for creating train and test dataframes
with bag of words features.
"""

# Author: Dan Lee <Lee_Daniel2@bah.com>

from operator import itemgetter
from itertools import chain

import pandas as pd
import numpy as np

class GetBowDummies(object):

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
        self.series = series
        self.index  = self.series.index
        self.features = features

        # Define dimension
        self.nrows = series.shape[0]
        self.ncols = len(features)
        self.dim   = (self.nrows, self.ncols)

    def index_feats_dict(self):
        """
        For every document row, features present in doc
        identified.
        """
        doc_features_dict = {}

        for index, doc in zip(self.index, self.series):
            # Sets for a doc and feature words
            doc_set = set(doc.split())
            feat_set = set(self.features)

            # Shared words between the two sets
            interset_words = doc_set.intersection(feat_set)

            # Append to doc_features_dict
            doc_features_dict[index] = list(interset_words)

        return doc_features_dict

    def get_bow_dummies(self):
        """
        Replace 0's with 1 in positions of a bow dataframe
        to indicate that feature words are present in docs
        """

        # Get an np matrix of zeros based on defined dim
        zero_matrix = np.zeros(self.dim, np.int)

        # Create a dataframe containing feature columns and 0's
        zero_df = pd.DataFrame(zero_matrix, columns=self.features)

        # Get a dictionary of index and features per doc
        doc_features_dict = self.index_feats_dict()
        doc_ids = doc_features_dict.keys()
        doc_feats = doc_features_dict.values()

        print(zero_df)
        # For each row in zero_df, indicate 1 for every
        # feature word present in a doc of a dataframe
        for index, feats in zip(doc_ids, doc_feats):
            zero_df.ix[index, feats] = 1

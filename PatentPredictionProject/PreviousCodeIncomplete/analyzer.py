from pandas import DataFrame
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle
import numpy as np


class Analyzer(object):
    def __init__(self, config):
        self.config = config
        self.data_frame = None
        self.data = None
        self.response = None
        self.feature_model = None
        self.feature_matrix = None

    def load_patent_data(self, filename):
        """
        Load raw data from CSV
        :param filename: Name of csv file to extract patents
        :return:
        """
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, filename))
        df.dropna(how='any')

        self.data_frame = df[[x.startswith(self.config.art_units) for x in df.artunit]]

        # Clean data, get rid of entries with less than 100 characters in the abstract, title, and claims
        criterion = self.data_frame['abstract'].map(lambda x: len(str(x)) > 100)
        self.data_frame = self.data_frame[criterion]

        criterion = self.data_frame['title'].map(lambda x: len(str(x)) > 10)
        self.data_frame = self.data_frame[criterion]

        criterion = self.data_frame['claims'].map(lambda x: len(str(x)) > 100)
        self.data_frame = self.data_frame[criterion]

    def extract_data(self, column_name):
        """

        :param column_name: Name list abstract or title or claims
        :return:
        """
        self.data = self.data_frame[column_name].tolist()

        # Assign art unit to class
        self.data_frame['class'] = [0]*self.data_frame.shape[0]
        for art_unit in self.config.art_units:
            self.data_frame.loc[self.data_frame['artunit'].str[:2] == str(art_unit), 'class'] = art_unit

        self.response = self.data_frame['class'].tolist()

    @staticmethod
    def tokenize(text):
        """
        Tokenizer that uses porter stemmer to stemm all works
        :param text:
        :return:
        """
        tokens = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(item) for item in tokens]
        return stemmed

    @staticmethod
    def initialize_model(n_grams):
        """
        Create a new TFIDF model
        :param n_grams: Number of phrases to capture in model
        :return: The initialized TFIDF model
        """
        model = TfidfVectorizer(
            ngram_range=(1, n_grams),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            decode_error='replace',
            tokenizer=Analyzer.tokenize,
            norm='l2',
            min_df=5,
            max_features=24000
        )
        return model

    def train_feature_model(self, n_grams, feature_name):
        """
        Train an existing or new model with data
        :param n_grams: Number of phrases to captures
        :return:
        """

        # Use loaded data to train
        if not self.data:
            raise ReferenceError('Data not loaded into analyzer object')

        # Do not initialize if the tfidf vector exists
        if self.feature_model:
            vec = self.feature_model
        else:
            vec = self.initialize_model(n_grams)

        # Fit data to model
        tfs = vec.fit_transform(self.data)

        # Put the vector back into the object
        self.feature_model = vec

        # Return the feature matrix for these items
        self.feature_matrix = tfs

        # Print stats
        print('Feature matrix', self.feature_matrix.shape)

        # Save
        self.save_model(feature_name)
        self.save_features(feature_name)

    def extract_features(self, n_grams, feature_name):
        """
        Get the feature matrix for a corpus with the saved vectorizer
        :param corpus:
        :return:
        """
        if os.path.isfile(self.config.get_matrix_path(feature_name)):
            self.load_features(feature_name)
        elif os.path.isfile(self.config.get_model_path(feature_name)):
            self.load_model(feature_name)
            self.feature_matrix = self.feature_model.transform(self.data)
            self.save_features(feature_name)
        else:
            self.train_feature_model(n_grams, feature_name)


    def transform(self, data):
        """
        Transform single data entry for web app
        :param data:
        :return:
        """
        return self.feature_model.transform(data)

    def save_model(self, feature_name):
        """
        Save the feature model into a pickled object
        :param feature_name:
        :return:
        """
        pickle.dump(self.feature_model, open(self.config.get_model_path(feature_name), 'wb'), protocol=2)

    def load_model(self, feature_name):
        """
        Load a saved model for additional training or testing
        :param feature_name
        :return:
        """
        self.feature_model = pickle.load(open(self.config.get_model_path(feature_name), 'rb'))

    def save_features(self, feature_name):
        """
        Use pickle to save the feature matrix in a python object
        :param feature_name:
        :return: Nothing
        """
        pickle.dump(self.feature_matrix, open(self.config.get_matrix_path(feature_name), 'wb'), protocol=2)

    def load_features(self, feature_name):
        """

        :param feature_name:
        :return:
        """
        self.feature_matrix = pickle.load(open(self.config.get_matrix_path(feature_name), 'rb'))

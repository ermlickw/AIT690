from config import Config
from analyzer import Analyzer
from classify import Classify
from scipy.sparse import hstack


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = Analyzer(self.config)
        self.classify = Classify(config)

    @staticmethod
    def get_all_column_data(file):
        """
        Combine all column data into a single feature matrix
        :param file:
        :return:
        """
        # Get all the feature matrices
        title_matrix, response_vector = f.analyze_column_data(file, 'title')
        abstract_matrix, response_vector = f.analyze_column_data(file, 'abstract')
        claims_matrix, response_vector = f.analyze_column_data(file, 'claims')

        # Get them all together
        feature_matrix = hstack([title_matrix, abstract_matrix])
        feature_matrix = hstack([feature_matrix, claims_matrix])
        return feature_matrix, response_vector

    def analyze_column_data(self, filename, column_name):
        """
        Create the feature model and matrix for the abstract column
        :param filename:
        :return:
        """
        self.analyzer.load_patent_data(filename)
        self.analyzer.extract_data(column_name)
        n_grams = 1
        self.analyzer.extract_features(n_grams, column_name)
        return self.analyzer.feature_matrix, self.analyzer.response

    def compute_heuristics(self, filename, column_name):
        """
        Figure out what words make up the groups in the shit
        :param filename:
        :return:
        """
        self.analyze_column_data(filename, column_name)
        self.analyzer.heuristics(column_name)

    def full_train(self):
        """
        GET THE CLASSIFIER TRAINED
        :return:
        """
        # self.classify.feature_selection()
        self.classify.classifier_selection()
        # self.classify.optimize_classifier()
        self.classify.train()
        self.classify.save_classifier()

    def evaluate(self, title, abstract, claims):
        """
        Predict group of a single entry
        :param abstract:
        :return:
        """
        self.analyzer.load_model('title')
        title_vector = self.analyzer.transform([title])
        self.analyzer.load_model('abstract')
        abstract_vector = self.analyzer.transform([abstract])
        self.analyzer.load_model('claims')
        claims_vector = self.analyzer.transform([claims])

        feature_vector = hstack([title_vector, abstract_vector])
        feature_vector = hstack([feature_vector, claims_vector])

        return feature_vector

    def predict(self, feature_vector):
        """
        Predict class based on feature vector input
        :param feature_vector:
        :return:
        """
        group = self.classify.predict(feature_vector)
        return group

if __name__ == '__main__':
    config_info = Config()
    f = Factory(config_info)
    file = '2015_2016_Patent_Data_new.csv'
    feature_matrix, response_vector = f.get_all_column_data(file)
    f.classify = Classify(config_info, feature_matrix, response_vector)
    f.full_train()
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
import numpy as np
import dill as pickle


class Classify(object):
    def __init__(self, feature_matrix=None):
        self.classifier = SGDClassifier()
        self.clf_name = 'SGD'
        self.feature_matrix = feature_matrix
        self.classifiers = {
            'Bayes': [MultinomialNB(), {'alpha': np.arange(0.0001, 0.2, 0.0001)}],

            'SGD': [SGDClassifier(n_iter=8, penalty='elasticnet'), {'alpha':  10**-6*np.arange(1, 15, 2),
                                                                    'l1_ratio': np.arange(0.1, 0.3, 0.05)}],

            'Passive Aggressive': [PassiveAggressiveClassifier(loss='hinge'), {}],

            'Perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}],
            'LogisticRegression': [LogisticRegression(solver='lbfgs', multi_class='multinomial'), {}],
            'LDA': [LinearDiscriminantAnalysis(solver='svd'), {}],
            'QDA': [QuadraticDiscriminantAnalysis(), {}],
            # 'MLP': [MLPClassifier(), {}]
        }


    @staticmethod
    def reduce_dimensionality(feature_matrix):
        """

        :param feature_matrix: Dense nxp matrix
        :return: Reduced matrix with q*p features
        """
        pca = PCA(n_components=feature_matrix.shape[1])
        feature_matrix_reduced = pca.fit_transform(feature_matrix)
        return feature_matrix_reduced

    def feature_selection(self):
        """
        Remove zero variance features
        :param feature_matrix:
        :param response_vector:
        :return:
        """
        feature_matrix = self.feature_matrix
        response_vector = self.response
        selected_feature_matrix = SelectKBest(chi2, k=int(0.05*feature_matrix.shape[0])).fit_transform(feature_matrix, response_vector)
        self.feature_matrix = selected_feature_matrix

    def optimize_classifier(self, clf_name=None):
        """
        Optimize a single classifier
        :param classifier:
        :param parameter_grid:
        :param parameter_of_interest:
        :return:
        """
        # TODO: THIS METHOD IS CURRENTLY BROKEN DUE TO THE C
        if not clf_name:
            clf_name = self.clf_name

        cross_val = KFold(len(self.response), n_folds=10, shuffle=True)
        clf = GridSearchCV(self.classifiers[clf_name][0], self.classifiers[clf_name][1], cv=cross_val, n_jobs=4)
        clf.fit(self.feature_matrix, self.response)
        print('Grid Search Completed', clf.best_estimator_, clf.best_score_)
        self.classifier = clf.best_estimator_
        self.results.plot_classifier_optimization(clf_name, clf.grid_scores_, clf.best_params_)

    def classifier_selection(self):
        """
        Select the classifier with the lowest error, create a plot comparing classifiers
        :return:
        """
        best_score = 0.1
        train_sizes = dict()
        train_scores = dict()
        valid_scores = dict()

        clf_names = self.classifiers.keys()
        for clf_name in clf_names:
            clf = self.classifiers[clf_name][0]
            train_sizes[clf_name], train_scores[clf_name], valid_scores[clf_name] = self.evaluate_learning_curve(clf)
            score = np.mean(valid_scores[clf_name][-1])
            print(clf_name, score)
            if score > best_score:
                self.clf_name = clf_name
                self.classifier = clf
                self.classifier.cv_score = score
                best_score = score

        self.results.plot_learning_curves(train_sizes, valid_scores, clf_names)

    def evaluate_learning_curve(self, classifier):
        """
        Evaluate the classifier input with learning curve and cross validation
        :param feature_matrix:
        :param response:
        :return:
        """
        train_sizes = np.arange(100, int(0.9*len(self.response)), 5000)
        cross_val = KFold(len(self.response), n_folds=10, shuffle=True)
        train_sizes, train_scores, valid_scores = learning_curve(classifier, self.feature_matrix, self.response,
                                                                 train_sizes=train_sizes, cv=cross_val,
                                                                 n_jobs=4)

        return train_sizes, train_scores, valid_scores

    def evaluate(self):

        """
        :param feature_matrix:
        :param response:
        :return:
        """
        cross_val = KFold(len(self.response), n_folds=10, shuffle=True)
        scores = cross_validation.cross_val_score(self.classifier, self.feature_matrix, self.response, cv=cross_val)

        test_response = self.predict(self.feature_matrix)
        conf_mat = confusion_matrix(self.response, test_response, self.config.art_units)
        print(conf_mat)
        return np.mean(scores)

    def train(self):
        """
        Train the model with the feature vector and response vector
        :param feature_matrix: blh
        :param response_vector: blh
        :return:
        """

        # Make sure the number of examples is greater than number of predictors
        self.classifier.fit(self.feature_matrix, self.response)
        train_error = self.evaluate()
        return train_error

    def predict(self, test_matrix):
        """

        :param test_matrix:
        :return: Predictions
        """
        predictions = self.classifier.predict(test_matrix)
        return predictions

    def save_classifier(self):
        """
        :param column_name:
        :return:
        """
        # SAVE MODEL
        path = self.config.get_classifier_path(self.clf_name)
        pickle.dump(self.classifier, open(path, 'wb'))

    def load_classifier(self, clf_name):
        """
        :param column_name:
        :return:
        """
        path = self.config.get_classifier_path(clf_name)
        self.classifier = pickle.load(open(path, 'rb'))

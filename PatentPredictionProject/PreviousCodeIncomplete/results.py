import numpy as np
import matplotlib.pyplot as plt


class Results(object):
    def __init__(self, config):
        self.config = config

    def plot_classifier_optimization(self, clf_name, classifier_grid_scores, best_parms):
        """
        Plot the result of the classifier optimization for a single classifier
        :param classifier_grid_scores:
        :param parameter:
        :return:
        """

        for param in best_parms.keys():
            parm_vec1 = [x.parameters['l1_ratio'] for x in classifier_grid_scores if x.parameters[param] == best_parms[param]]
            scores = [x.mean_validation_score for x in classifier_grid_scores]
            plt.plot(parm_vec1, scores, '.')
            plt.title(clf_name)
            plt.show()

            parm_vec2 = [x.parameters['alpha'] for x in classifier_grid_scores if x.parameters[param] == best_parms[param]]
            plt.plot(parm_vec2, scores, '.')
            plt.title(clf_name)
            plt.show()

    def plot_learning_curves(self, train_sizes, valid_scores, classifiers):
        """
        Plot the result from the learning curves of a single classifier
        :param train_sizes:
        :param train_scores:
        :param valid_scores:
        :return:
        """
        for clf in classifiers:
            plt.plot(train_sizes[clf], np.mean(valid_scores[clf], axis=1),  '.-', label=clf)

        plt.legend(loc='best')
        plt.title('Classifier Learning Curve Comparison')
        plt.show()
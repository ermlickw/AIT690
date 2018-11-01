import os
from datetime import date


class Config(object):
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.results_dir = os.path.join(self.base_dir, 'result')
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.art_units = (
            "24", "36",
            "21", "26"
            # "16", "17",
            # "28", "37"
        )
        self.today = date.today().isoformat()

    def get_model_path(self, model_name):
        return os.path.join(self.model_dir, model_name + self.today + '.dill')

    def get_matrix_path(self, feature_name):
        return os.path.join(self.data_dir, feature_name + self.today + '_matrix.dill')

    def get_classifier_path(self, model_name):
        return os.path.join(self.model_dir, model_name + "-" + "".join(self.art_units) + '-' + self.today + '.dill')


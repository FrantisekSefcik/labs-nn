import sys
sys.path.append('../')
from python.shared.layer_model import LayerModelBase, LayerKNNModel

from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model
import numpy as np

from python.shared.utils import get_train_test


class SuspiciousModel:
    def __init__(self, original_model, layers_no, class_names):
        self.layer_models = [
            LayerKNNModel(original_model, layer) for layer in layers_no
        ]
        self._clfs = None
        self.class_names = class_names

    def train(self, x, y, y_pred, train=0.6):
        a_x, b_x = get_train_test(x, train)
        a_y, b_y = get_train_test(y, train)
        _, b_y_pred = get_train_test(y_pred, train)
        self.train_knn_layers(a_x, a_y)
        train_data = self.predict_knn_layers(b_x)

        self._clfs = {}
        for label in range(len(self.class_names)):
            self._clfs[label] = DecisionTreeClassifier(random_state=42,
                                                       max_depth=5)
            self._clfs[label].fit(train_data[b_y_pred == label],
                                  b_y[b_y_pred == label] == label)

    def predict(self, x, pred_label):
        data = self.predict_knn_layers(x)
        return np.invert(self._clfs[pred_label].predict(data))

    def train_knn_layers(self, x, y):
        for l_m in self.layer_models:
            l_m.train_knn(x, y)

    def predict_knn_layers(self, x):
        data = [l_m.predict_knn(x) for l_m in self.layer_models]
        data = np.stack(data, axis=1)
        return data.reshape(
            (data.shape[0], data.shape[1] * data.shape[2]))


class SimilarityCasesModel:
    def __init__(self, original_model: Model, selected_layers):
        self.layer_models = [
            LayerKNNModel(original_model, layer, None) for layer in selected_layers
        ]

    def layers_activations(self, data):
        return [model.layer_activations(data) for model in self.layer_models]

    def train_knns(self, data, labels):
        for model in self.layer_models:
            model.train_knn(data, labels)

    def predict_similar_cases(self, data):
        return [model.predict_knn(data) for model in self.layer_models]
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from python.utils import get_train_test


def flatten_array_elements(arr):
    return np.array([x.flatten() for x in arr])


def get_flatten_activations(activations):
    return [flatten_array_elements(a) for a in activations]


class LayerModel:
    def __init__(self, original_model: Model, selected_layers, class_names):
        self._knns = None
        self._atlas_x = None
        self._atlas_y = None
        self._clfs = None
        self.class_names = class_names
        if selected_layers is None or len(selected_layers) == 0:
            raise ValueError("selected_layers must be speciefied")

        if isinstance(selected_layers[0], str):
            outputs = [layer.output for layer in original_model.layers if layer.name in selected_layers]
        else:
            outputs = [original_model.layers[i].output for i in selected_layers]

        self._model = Model(inputs=original_model.inputs, outputs=outputs)

    def predict_layers_activations(self, x):
        activations = self._model.predict(x)
        return get_flatten_activations(activations)

    def train_knns(self, x, y):
        self._atlas_x = x
        self._atlas_y = y
        layers_activations = self.predict_layers_activations(x)
        self.train_knn_with_activations(layers_activations, y)

    def train_knn_with_activations(self, layers_activations, labels):
        self._knns = []
        for layer_activations in layers_activations:
            neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
            neigh.fit(layer_activations, labels)
            self._knns.append(neigh)

    def predict_knn(self, x):
        layers_activations = self.predict_layers_activations(x)
        data = []
        for i, layer_activations in enumerate(layers_activations):
            data.append(self._knns[i].predict_proba(layer_activations))
        data = np.stack(data, axis=1)
        return data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    def get_neighbours_idxs_on_layers(self, x):
        layers_activations = self.predict_layers_activations(x)
        indxs = []
        for i, layer_activations in enumerate(layers_activations):
            neighbours_idx = self._knns[i].kneighbors(layer_activations)
            indxs.append(neighbours_idx[1])

        return indxs

    def train(self, x, y, y_pred, train=0.6):
        a_x, b_x = get_train_test(x, train)
        a_y, b_y = get_train_test(y, train)
        _, b_y_pred = get_train_test(y_pred, train)
        self.train_knns(a_x, a_y)
        train_data = self.predict_knn(b_x)

        self._clfs = {}

        for label in range(len(self.class_names)):
            self._clfs[label] = DecisionTreeClassifier(random_state=42, max_depth=5)
            self._clfs[label].fit(train_data[b_y_pred==label], b_y[b_y_pred==label] == label)

    def predict(self, x, pred_label):
        x_knn = self.predict_knn(x)
        return np.invert(self._clfs[pred_label].predict(x_knn))








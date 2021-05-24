from tensorflow.keras import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def flatten_array_elements(arr):
    return np.array([x.flatten() for x in arr])


def get_flatten_activations(activations):
    return [flatten_array_elements(a) for a in activations]


class LayerModelBase:

    def __init__(self, original_model: Model, selected_layer):
        self._model = Model(
            inputs=original_model.inputs,
            outputs=original_model.layers[selected_layer].output
        )

    def layer_activations(self, x):
        activations = self._model.predict(x)
        return flatten_array_elements(activations)

    def layer_activations_generator(self, generator, batch_size):
        all_labels = []
        all_activations = None
        for _ in range(round(generator.samples / batch_size)):
            (images, labels) = generator.next()
            activations = self._model.predict(images)
            all_labels.extend(labels)
            if all_activations is None:
                all_activations = activations
            else:
                all_activations = [
                    np.concatenate((a1, a2), axis=0)
                    for a1, a2 in zip(all_activations, activations)
                ]
        return get_flatten_activations(all_activations), all_labels


class LayerKNNModel(LayerModelBase):
    def __init__(self, original_model: Model, selected_layer, knn_model=None):
        super().__init__(original_model, selected_layer)
        self._knn = knn_model if knn_model else KNeighborsClassifier(n_neighbors=5, n_jobs=8)

    def train_knn(self, x, y):
        layer_activations = self.layer_activations(x)
        self.train_knn_with_activations(layer_activations, y)

    def train_knn_with_activations(self, layer_activations, y):
        self._knn.fit(layer_activations, y)

    def predict_knn(self, x):
        layer_activations = self.layer_activations(x)
        return self._knn.predict_proba(layer_activations)

    def get_neighbours_idx(self, x):
        layer_activations = self.layer_activations(x)
        neigh = self._knn.kneighbors(layer_activations)
        return neigh[1]
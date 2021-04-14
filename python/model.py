from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import innvestigate

from python.utils import get_train_test


def flatten_array_elements(arr):
    return np.array([x.flatten() for x in arr])


def get_flatten_activations(activations):
    return [flatten_array_elements(a) for a in activations]


class SuspiciousModel:
    def __init__(self, original_model, layers_no, class_names):
        self.layer_models = [
            LayerModel(original_model, layer) for layer in layers_no
        ]
        for l_m in self.layer_models:
            l_m.initialize_bilrp("lrp.epsilon", {"epsilon": 1})
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


class LayerModel:
    def __init__(self, original_model: Model, layer, knn_model=None):
        self._knn = knn_model if knn_model else KNeighborsClassifier(n_neighbors=5, n_jobs=8)
        self._bi_lrp = None
        self._model = Model(
            inputs=original_model.inputs,
            outputs=original_model.layers[layer].output
        )

    def initialize_bilrp(self, method, params):
        self._bi_lrp = BiLrp(
            innvestigate.create_analyzer(
                method,
                self._model,
                **params))

    def get_layer_activations(self, x):
        activations = self._model.predict(x)
        return flatten_array_elements(activations)

    def predict_layers_activations_generator(self, generator, batch_size):
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

    def train_knn(self, x, y):
        layer_activations = self.get_layer_activations(x)
        self.train_knn_with_activations(layer_activations, y)

    def train_knn_with_activations(self, layer_activations, y):
        self._knn.fit(layer_activations, y)

    def predict_knn(self, x):
        layer_activations = self.get_layer_activations(x)
        return self._knn.predict_proba(layer_activations)

    def get_neighbours_idx(self, x):
        layer_activations = self.get_layer_activations(x)
        neigh = self._knn.kneighbors(layer_activations)
        return neigh[1]

    def common_features(self, img1, img2, top_n):
        layers_activations = self.get_layer_activations(np.array([img1, img2]))
        return layers_activations.mean(0).argsort()[-top_n:][::-1]

    def analyze_pair_bilrp(self, img1, img2, poolstride=1, top_neurons=None):
        selected_neurons = None if top_neurons is None else self.common_features(img1, img2, top_neurons)
        return self._bi_lrp.analyze_pair(img1, img2, poolstride, selected_neurons)


class BiLrp:
    def __init__(self, analyzer_model):
        self._analyzer = analyzer_model
        self.output_shape = analyzer_model._model.output.shape

    def analyze_pair(self, img1, img2, poolstride=1, neuron_selection=None):

        if neuron_selection is None:
            size = (tf.math.reduce_prod(self.output_shape[1:]))
            r1 = np.empty((size, img1.shape[0], img1.shape[1]))
            r2 = np.empty((size, img1.shape[0], img1.shape[1]))
            iterator = range(size)
        else:
            r1 = np.empty((len(neuron_selection), img1.shape[0], img1.shape[1]))
            r2 = np.empty((len(neuron_selection), img1.shape[0], img1.shape[1]))
            iterator = neuron_selection

        for n, i in enumerate(iterator):
            R = self._analyzer.analyze([img1, img2], neuron_selection=int(i))[
                'input_layer']
            r1[n] = R[0].sum(2)
            r2[n] = R[1].sum(2)
        # dot product of Ri . Ri'
        result = np.tensordot(pool(r1, poolstride), pool(r2, poolstride),
                            axes=(0, 0))
        del r1
        del r2
        return result

    # def analyze_images(self, img1, images, poolstride=1, neuron_selection=None):
    #
    #     if neuron_selection is None:
    #         size = (tf.math.reduce_prod(self.output_shape[1:]))
    #         r1 = np.empty((size, img1.shape[0], img1.shape[1]))
    #         r2 = np.empty((size, img1.shape[0], img1.shape[1]))
    #         iterator = range(size)
    #     else:
    #         r1 = np.empty((len(neuron_selection), img1.shape[0], img1.shape[1]))
    #         r2 = np.empty((len(neuron_selection), img1.shape[0], img1.shape[1]))
    #         iterator = neuron_selection
    #
    #     for n, i in enumerate(iterator):
    #         R = self._analyzer.analyze(np.append([img1], images), neuron_selection=int(i))[
    #             'input_layer']
    #         r1[n] = R[0].sum(2)
    #         r2[n] = R[1].sum(2)
    #     # dot product of Ri . Ri'
    #     result = np.tensordot(pool(r1, poolstride), pool(r2, poolstride),
    #                         axes=(0, 0))
    #     del r1
    #     del r2
    #     return result


def pool(x, stride):
    K = [np.squeeze(
        tf.nn.avg_pool2d(np.expand_dims(o, (0, 3)), ksize=stride,
                         strides=stride, padding='VALID')) for o in x]
    return K



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


class BiLRPLayerModel(LayerKNNModel):
    def __init__(self, original_model: Model, selected_layer, analyzer=None, knn_model=None):
        super().__init__(original_model, selected_layer, knn_model)
        self._analyzer = analyzer
        self.output_shape = original_model.output.shape

    def initialize_analyzer(self, method, params):
        self._analyzer = innvestigate.create_analyzer(method, self._model, **params)

    def top_common_activations(self, img1, img2, top_n):
        layers_activations = self.layer_activations(np.array([img1, img2]))
        return layers_activations.mean(0).argsort()[-top_n:][::-1]

    def pair_similarity(self, img1, img2, poolstride=1, top_neurons=None):
        if self._analyzer is None:
            raise Exception("Analyzer is not initialized!")
        selected_neurons = None if top_neurons is None else self.top_common_activations(img1, img2, top_neurons)
        return self.analyze_pair(img1, img2, poolstride, selected_neurons)

    def lrp(self, data):
        if self._analyzer is None:
            raise Exception("Analyzer is not initialized!")
        return self._analyzer.analyze(data)

    def analyze_pair(self, img1, img2, poolstride=1, neuron_selection=None):

        if neuron_selection is None:
            size = (tf.math.reduce_prod(self.output_shape[1:]))
            r1 = np.empty((size, img1.shape[0], img1.shape[1]))
            r2 = np.empty((size, img1.shape[0], img1.shape[1]))
            iterator = range(size)
        else:
            r1 = np.empty((len(neuron_selection), img1.shape[0], img1.shape[1]))
            r2 = np.empty((len(neuron_selection), img1.shape[0], img1.shape[1]))
            iterator = neuron_selection

        for n, i in enumerate(iterator):
            R = self._analyzer.analyze([img1, img2], neuron_selection=int(i))[
                'input_layer']
            r1[n] = R[0].sum(2)
            r2[n] = R[1].sum(2)
        # dot product of Ri . Ri'
        return np.tensordot(pool(r1, poolstride), pool(r2, poolstride),
                            axes=(0, 0))


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
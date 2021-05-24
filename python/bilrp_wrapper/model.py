import sys
sys.path.append('../')
from python.shared.layer_model import LayerKNNModel

from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import innvestigate


def pool(x, stride):
    K = [np.squeeze(
        tf.nn.avg_pool2d(np.expand_dims(o, (0, 3)), ksize=stride,
                         strides=stride, padding='VALID')) for o in x]
    return K


class BiLRPLayerModel(LayerKNNModel):
    def __init__(self, original_model: Model, selected_layer, analyzer_method=None, analyzer_params=None, knn_model=None):
        super().__init__(original_model, selected_layer, knn_model)
        if analyzer_method is not None:
            self.initialize_analyzer(analyzer_method, analyzer_params)
        self.output_shape = original_model.output.shape

    def initialize_analyzer(self, method, params):
        self._analyzer = innvestigate.create_analyzer(method, self._model, **params)

    def top_common_activations(self, img1, img2, top_n):
        layers_activations = self.layer_activations(np.array([img1, img2]))
        return layers_activations.mean(0).argsort()[-top_n:][::-1]

    def max_activations(self, img1, img2, top_n):
        layers_activations = self.layer_activations(np.array([img1, img2]))
        return layers_activations.max(0).argsort()[-top_n:][::-1]

    def pair_similarity(self, img1, img2, poolstride=1, top_neurons=None):
        if self._analyzer is None:
            raise Exception("Analyzer is not initialized!")
        selected_neurons = None if top_neurons is None else self.max_activations(img1, img2, top_neurons)
        return self.analyze_pair(img1, img2, poolstride, selected_neurons)

    def multiple_similarity(self, img, images, poolstride=1, top_neurons=None):
        if self._analyzer is None:
            raise Exception("Analyzer is not initialized!")
        R_array = []
        for image in images:
            selected_neurons = None if top_neurons is None else self.max_activations(img, image, top_neurons)
            R_array.append(self.analyze_pair(img, image, poolstride, selected_neurons))
        return np.array(R_array)

    def lrp(self, data):
        if self._analyzer is None:
            raise Exception("Analyzer is not initialized!")
        return self._analyzer.analyze(data)

    def analyze_pair(self, img1, img2, poolstride=1, neuron_selection=None):

        if neuron_selection is None:
            size = (tf.math.reduce_prod(self.output_shape[1:]))
            iterator = range(size)
        else:
            size = len(neuron_selection)
            iterator = neuron_selection

        r1 = np.empty((size, img1.shape[0], img1.shape[1]))
        r2 = np.empty((size, img1.shape[0], img1.shape[1]))

        for n, i in enumerate(iterator):
            R = self._analyzer.analyze([img1, img2], neuron_selection=int(i))[
                'input_layer']
            r1[n] = R[0].sum(2)
            r2[n] = R[1].sum(2)
        # dot product of Ri . Ri'
        return np.tensordot(pool(r1, poolstride), pool(r2, poolstride),
                            axes=(0, 0))

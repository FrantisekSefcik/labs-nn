from sklearn.neighbors import KNeighborsClassifier


def get_train_test(x, train=0.8):
    div = int(len(x) * train)
    return x[:div], x[div:]


def train_knn_with_activations(layers_activations, labels):
    layers_knn = []
    for layer_activations in layers_activations:
        neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
        neigh.fit(layer_activations, labels)
        layers_knn.append(neigh)
    return layers_knn


def get_neighbours_idxs_on_layers(layers_activations, layers_knn):
    indxs = []
    for i, layer_activations in enumerate(layers_activations):
        neighbours_idx = layers_knn[i].kneighbors(layer_activations)
        indxs.append(neighbours_idx[1])

    return indxs
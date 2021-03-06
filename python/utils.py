from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def get_mask_for_gray(segmentation, threshold=0, exact=None):
    """
    Get mask of tumor as gray image

    :param segmentation: segmentation image
    :param threshold: threshold for tumor region
    :param exact: True if mask is count as equation of threshold
    :return: mask of tumor with shape (height, width, 1)
    """
    if exact:
        return segmentation == exact
    else:
        return segmentation > threshold


def get_mask_of_seg_rgb(segmentation, threshold=0, exact=False):
    """
    Get mask of tumor for rgb image

    :param segmentation: segmentation image
    :param threshold: threshold for tumor region
    :param exact: True if mask is count as equation of threshold
    :return: mask of tumor with shape (height, width, 3)
    """
    height, width = segmentation.shape[0], segmentation.shape[1]
    new_mask = np.zeros((height, width, 3), dtype=bool)
    if exact:
        new_seg = (segmentation == threshold)[:, :, 0]
    else:
        new_seg = (segmentation > threshold)[:, :, 0]
    for ch in range(3):
        new_mask[:, :, ch] = new_seg
    return new_mask


def get_mask_of_brain(image, segmentation=None):
    """
    Get mask of brain from original image as gray image
    mask = brain mask - tumor mask

    :param image: original image
    :param segmentation: tumor segmentation image
    :return: mask of brain without tumor region with shape (height, width, 1)
    """
    if segmentation is not None and image.shape != segmentation.shape:
        raise Exception('Not equal shape of image and segmentation')
    if len(image.shape) == 3:
        image = image.any(axis=np.argmax(np.asarray(image.shape) == 3))

    return image > 0


def get_mask_of_brain_rgb(image, segmentation=None):
    """
    Get mask of brain from original image as rgb image
    mask = brain mask - tumor mask

    :param image: original image
    :param segmentation: tumor segmentation image
    :return: mask of brain without tumor region with shape (height, width, 3)
    """
    if segmentation is not None and image.shape != segmentation.shape:
        raise Exception('Not equal shape of image and segmentation')
    if segmentation is not None:
        mask = image > 0
        mask[segmentation] = False
        return mask
    else:
        return image > 0


def indexes_predicted_groups(y, pred):
    """
    return indexes for each predicted group (TP, TN, FN, FP)
    :param pred: numpy array of predicted class
    :param y: numpy array of true class
    :return: dictionary with arrays of True, False of index
    """
    if type(y) is not np.ndarray:
        y = np.array(y)
    if type(pred) is not np.ndarray:
        pred = np.array(pred)

    result = {'tp': (y == 1) & (pred == 1),
              'tn': (y == 0) & (pred == 0),
              'fn': (y == 1) & (pred == 0),
              'fp': (y == 0) & (pred == 1)}
    return result



def sum_image_channels(image):
    """
    Sum values cross channels in image
    :param image: original image
    :return: ndarray with shape (height, weight, 1)
    """
    return image.sum(axis=np.argmax(np.asarray(image.shape) == 3))


def any_image_channels(image):
    """
    Any value cross channels of image
    :param image: original image
    :return: ndarray with shape (height, weight, 1)
    """
    return image.any(axis=np.argmax(np.asarray(image.shape) == 3))


def scale_image_intensity(image, min=0, max=1):
    """
    Scale image intensities into range min max
    :param image: original image
    :param min: min of range
    :param max: max of range
    :return: scaled image
    """

    return image


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


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
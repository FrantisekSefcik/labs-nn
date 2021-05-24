import numpy as np
import innvestigate
import math


def mask_value(heatmap, image, mask):
    """
    Function to count metric how analysis hit into mask.
    :param heatmap: saliency or activation map obtained from analyzer
    :param image: original image
    :param mask: mask of region where prediction should hit
    :return: score
    """
    if image.shape != mask.shape:
        raise Exception('Not equal shape of image and segmentation')
    if mask.sum():
        brain_mask = get_mask_of_brain_rgb(image, mask)
        tumor_val = heatmap[mask].clip(0).sum()
        brain_val = heatmap[brain_mask].clip(0).sum()
        return tumor_val / (tumor_val + brain_val)
    else:
        return 0


def get_mask_of_brain(image, segmentation=None):
    """
    Get mask of brain from original image as gray image
    mask = brain mask - tumor mask

    :param image: original image
    :param segmentation: tumor segmentation image
    :return: mask of brain without tumor region with shape (height, width, 1)
    """

    if len(image.shape) == 3:
        image = image.any(axis=np.argmax(np.asarray(image.shape) == 3))

    if segmentation is not None:
        mask = image > 0
        mask[segmentation] = False
        return mask
    else:
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


def get_mask_of_seg(segmentation, threshold=0, exact=None):
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


def get_mask_stat(gen, gen_seg, model, batch_size=32):
    analyzer = innvestigate.create_analyzer("lrp.epsilon", model, epsilon=1)
    num_batches = math.ceil(gen.samples / batch_size)
    labels = []
    pred = []
    mask_values = []
    for i, ((images, y), (images_seg, _)) in enumerate(zip(gen, gen_seg)):
        if i >= num_batches:
            break
        prob = model.predict(images)
        analysis = analyzer.analyze(images)["input_layer"]
        mask = [
            mask_value(i_a, i, get_mask_of_seg_rgb(i_s))
            for i, i_a, i_s in zip(images, analysis, images_seg)
        ]
        p = prob.argmax(axis=1)
        pred.extend(p)
        labels.extend(y)
        mask_values.extend(mask)
    return np.array(mask_values), np.array(pred), np.array(labels)

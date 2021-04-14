import sys
sys.path.append('../')
from python.mask_utils import mask_loss_val

import numpy as np
import innvestigate
from tensorflow import keras
import tensorflow as tf
import math


def get_analysis(x, analyzer, labels):
    return np.array([t[int(l)] for l, t in zip(labels, zip(
        analyzer.analyze(x, neuron_selection=0)['input_layer'],
        analyzer.analyze(x, neuron_selection=1)['input_layer']
    ))])
    # return analyzer.analyze(x)['input_layer']


def threshold_to_one(images):
    images[images > 1] = 1
    return images


def analysis_to_segmentation(images):
    a = np.expand_dims(images.sum(3), axis=3)
    a[a < 0] = 0
    return a / a.max()


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    print(axes)
    numerator = 2. * np.sum(y_pred * y_true, axes)
    print(numerator)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    print(denominator)
    return 1 - np.mean((numerator + epsilon) / (
                denominator + epsilon))  # average over classes and batch
    # thanks @mfernezir for catching a bug in an earlier version of this implementation!


def get_lrp_loss(analyzer):
    loss_function = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    def loss(target_y, predicted_y, x, x_seg):
        loss_value = loss_function(target_y, predicted_y)
        # a = analysis_to_segmentation(get_analysis(x, analyzer, target_y))
        # m = threshold_to_one(x_seg)
        # print(a.shape, m.shape)
        # assert a.shape == m.shape
        # mask_value = loss_function(m, a)
        mask_value = mask_loss_val(x, get_analysis(x, analyzer, target_y), x_seg)
        return loss_value / tf.pow(mask_value, 2), mask_value
        # return loss_value, mask_value

    return loss


def step(xs, xs_seg, ys, model, optimizer, loss_function):

    with tf.GradientTape() as tape:
        pred = model(xs)  # Model predictions
        loss, mask = loss_function(ys, pred, xs, xs_seg)
        # values ys with predictions

    gradient = tape.gradient(
        target=loss,
        sources=model.trainable_variables)

    optimizer.apply_gradients(zip(gradient,
                                  model.trainable_variables))
    return loss, mask


def get_val_stat(gen, gen_seg, batch_size, loss_function, model):
    test_loss_avg = tf.keras.metrics.Mean()
    test_mask_avg = tf.keras.metrics.Mean()

    num_batches = math.ceil(gen.samples / batch_size)
    y_true_v = []
    pred_v = []
    for i, ((images, labels), (images_seg, _)) in enumerate(zip(gen, gen_seg)):
        prob = model.predict(images)
        p = prob.argmax(axis=1)
        pred_v.extend(p)
        y_true_v.extend(labels)
        loss, mask = loss_function(labels, prob, images, images_seg)
        test_loss_avg.update_state(loss)
        test_mask_avg.update_state(mask)
        if i + 1 == num_batches:
            break
    return test_loss_avg.result(), test_mask_avg.result(), pred_v, y_true_v


def train(model, train_image_gen, train_seg_gen, val_image_gen, val_seg_gen,
          loss_function, optimizer, epochs, batch_size):
    num_batches = math.ceil(train_image_gen.samples / batch_size)

    train_loss_results = []
    train_mask_results = []
    train_accuracy_results = []

    # Training loop (without shuffling for simplicity)
    for e in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_mask_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_test_accuracy = tf.keras.metrics.Accuracy()

        for i, ((images, labels), (images_seg, _)) in enumerate(
                zip(train_image_gen, train_seg_gen)):
            loss, mask = step(images, images_seg, labels, model, optimizer, loss_function)
            epoch_loss_avg.update_state(loss)  # Add current batch loss
            epoch_mask_avg.update_state(mask)  # Add current batch loss
            epoch_accuracy.update_state(labels, model(images, training=True))
            if i + 1 == num_batches:
                break

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_mask_results.append(epoch_mask_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        l_val, m_val, pred, y = get_val_stat(val_image_gen, val_seg_gen,
                                          batch_size, loss_function, model)
        epoch_test_accuracy.update_state(y, pred)
        print("Epoch: {}, Train loss: {:.3f}, Train mask: {:.3f}, Train accuracy: {:.3f}, Test loss: {:.3f}, Test mask: {:.3f}, Test accuracy: {:.3f}".format(
            e, epoch_loss_avg.result(), epoch_mask_avg.result(), epoch_accuracy.result(), l_val, m_val, epoch_test_accuracy.result()
        ))


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean((numerator + epsilon) / (
                denominator + epsilon))  # average over classes and batch
    # thanks @mfernezir for catching a bug in an earlier version of this implementation!
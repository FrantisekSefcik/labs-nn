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


def get_lrp_loss(analyzer):
    loss_function = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    def loss(target_y, predicted_y, x, x_seg):
        loss_value = loss_function(target_y, predicted_y)
        mask_value = mask_loss_val(x, get_analysis(x, analyzer, target_y), x_seg)
        return loss_value / tf.pow(mask_value, 2), mask_value

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
    num_batches = math.ceil(gen.samples / batch_size)
    y_true_v = []
    pred_v = []
    stats = {
        'loss': [],
        'mask': []
    }
    for i, ((images, labels), (images_seg, _)) in enumerate(zip(gen, gen_seg)):
        if i >= num_batches:
            break
        prob = model.predict(images)
        p = prob.argmax(axis=1)
        pred_v.extend(p)
        y_true_v.extend(labels)
        loss, mask = loss_function(labels, prob, images, images_seg)
        stats['loss'].append(loss)
        stats['mask'].append(mask)
    return np.mean(stats['loss']), np.mean(stats['mask']), pred_v, y_true_v


def train(model, train_image_gen, train_seg_gen, val_image_gen, val_seg_gen,
          loss_function, optimizer, epochs, batch_size):
    num_batches = math.ceil(train_image_gen.samples / batch_size)
    # Training loop (without shuffling for simplicity)
    for e in range(epochs):
        stats = {
            'loss': [],
            'mask': []
        }
        for i, ((images, labels), (images_seg, _)) in enumerate(
                zip(train_image_gen, train_seg_gen)):
            if i >= num_batches:
                break
            loss, mask = step(images, images_seg, labels, model, optimizer, loss_function)
            print("mask:", mask, "loss:", loss)
            stats['loss'].append(loss)
            stats['mask'].append(mask)

        l_val, m_val, _, _ = get_val_stat(val_image_gen, val_seg_gen,
                                          batch_size, loss_function, model)
        print('Epoch:', e, 'Train loss:', np.mean(stats['loss']), 'Train mask:',
              np.mean(stats['mask']), 'Test loss:', l_val, 'Test mask:', m_val)

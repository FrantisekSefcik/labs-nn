import sys
sys.path.append('../')
from python.shared import visualization
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
from python.shared.mask_utils import *




def get_analysis(x, analyzer, labels):
    # return np.array([t[int(l)] for l, t in zip(labels, zip(
    #     analyzer.analyze(x, neuron_selection=0)['input_layer'],
    #     analyzer.analyze(x, neuron_selection=1)['input_layer']
    # ))])
    return analyzer.analyze(x)['input_layer']


def threshold_to_one(images):
    images[images > 1] = 1
    return images


def mask_loss_val(x, x_analysis, x_seg):
    mask_values = [
        mask_value(i_a, i, get_mask_of_seg_rgb(i_s))
        for i, i_a, i_s in zip(x, x_analysis, x_seg)]
    return tf.math.reduce_mean(tf.cast(mask_values, tf.float32))


def analysis_to_segmentation(images):
    a = np.expand_dims(images.sum(3), axis=3)
    a[a < 0] = 0
    return a / a.max()


def get_lrp_loss(analyzer):
    loss_function = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    def loss(target_y, predicted_y, x, x_seg):
        loss_value = loss_function(target_y, predicted_y)
        mask_value = mask_loss_val(x, get_analysis(x, analyzer, target_y), x_seg)
        return loss_value / mask_value, mask_value, loss_value

    return loss


def step(xs, xs_seg, ys, model, optimizer, loss_function):

    with tf.GradientTape() as tape:
        pred = model(xs)  # Model predictions
        loss, mask, org_loss = loss_function(ys, pred, xs, xs_seg)
        # values ys with predictions

    gradient = tape.gradient(
        target=loss,
        sources=model.trainable_variables)

    optimizer.apply_gradients(zip(gradient,
                                  model.trainable_variables))
    return loss, mask, org_loss


def get_val_stat(gen, gen_seg, batch_size, loss_function, model):
    test_loss_avg = tf.keras.metrics.Mean()
    test_org_loss_avg = tf.keras.metrics.Mean()
    test_mask_avg = tf.keras.metrics.Mean()

    num_batches = math.ceil(gen.samples / batch_size)
    y_true_v = []
    pred_v = []
    for i, ((images, labels), (images_seg, _)) in enumerate(zip(gen, gen_seg)):
        prob = model.predict(images)
        p = prob.argmax(axis=1)
        pred_v.extend(p)
        y_true_v.extend(labels)
        loss, mask, org_loss = loss_function(labels, prob, images, images_seg)
        test_loss_avg.update_state(loss)
        test_mask_avg.update_state(mask)
        test_org_loss_avg.update_state(org_loss)
        if i + 1 == num_batches:
            break
    return test_loss_avg.result(), test_mask_avg.result(), test_org_loss_avg.result(), pred_v, y_true_v


def train(model, train_image_gen, train_seg_gen, val_image_gen, val_seg_gen,
          loss_function, optimizer, epochs, batch_size, analyzer, test_images,
          test_images_seg):
    num_batches = math.ceil(train_image_gen.samples / batch_size)
    logs_data = {
        'train_loss_values': [],
        'train_org_loss_values': [],
        'train_mask_values': [],
        'train_accuracy_values': [],
        'test_loss_values': [],
        'test_org_loss_values': [],
        'test_mask_values': [],
        'test_accuracy_values': []
    }
    # Training loop (without shuffling for simplicity)
    for e in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_org_loss_avg = tf.keras.metrics.Mean()
        epoch_mask_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_test_accuracy = tf.keras.metrics.Accuracy()

        for i, ((images, labels), (images_seg, _)) in enumerate(
                zip(train_image_gen, train_seg_gen)):
            loss, mask, org_loss = step(images, images_seg, labels, model, optimizer, loss_function)
            epoch_loss_avg.update_state(loss)  # Add current batch loss
            epoch_org_loss_avg.update_state(org_loss)  # Add current batch loss
            epoch_mask_avg.update_state(mask)  # Add current batch loss
            epoch_accuracy.update_state(labels, model(images, training=True))
            if i + 1 == num_batches:
                break

        # Logging
        l_val, m_val, l_org_val, pred, y = get_val_stat(val_image_gen, val_seg_gen, batch_size, loss_function, model)
        epoch_test_accuracy.update_state(y, pred)
        print("Epoch: {}, Train loss: {:.3f}, Train orgloss: {:.3f},Train mask: {:.3f}, Train accuracy: {:.3f}\n          Test loss: {:.3f}, Test org loss: {:.3f}, Test mask: {:.3f}, Test accuracy: {:.3f}".format(
            e, epoch_loss_avg.result(), epoch_org_loss_avg.result(), epoch_mask_avg.result(), epoch_accuracy.result(), l_val, l_org_val,  m_val, epoch_test_accuracy.result()
        ))
        logs_data['train_loss_values'].append(epoch_loss_avg.result().numpy().astype(float))
        logs_data['train_org_loss_values'].append(epoch_org_loss_avg.result().numpy().astype(float))
        logs_data['train_mask_values'].append(epoch_mask_avg.result().numpy().astype(float))
        logs_data['train_accuracy_values'].append(epoch_accuracy.result().numpy().astype(float))
        logs_data['test_loss_values'].append(l_val.numpy().astype(float))
        logs_data['test_org_loss_values'].append(l_org_val.numpy().astype(float))
        logs_data['test_mask_values'].append(m_val.numpy().astype(float))
        logs_data['test_accuracy_values'].append(epoch_test_accuracy.result().numpy().astype(float))

        # Step visualization
        a = analyzer.analyze(test_images)['input_layer']
        mask = [
            mask_value(i_a, i, get_mask_of_seg_rgb(i_s))
            for i, i_a, i_s in zip(test_images, a, test_images_seg)]
        fig, axs = plt.subplots(1, 5, figsize=(15, 20))
        for x in range(5):
            visualization.plot_lrp(a[x], axs[x])
            axs[x].set_title('Mask value: {:.2f}'.format(mask[x]))
        plt.show()

    return logs_data
import matplotlib.pyplot as plt
import numpy as np

COLOR_PALETTE = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def plot_layers_activations(prediction, y_pred, y_true):
    x = np.arange(10)  # the label locations
    width = 0.35  # the width of the bars
    fig, axs = plt.subplots(2, 5, figsize=(20, 5))
    axs = axs.flatten()

    for i in range(10):
        if any(true_predictions_of_ground_truth(y_pred, y_true, i)):
            axs[i].bar(x - width / 2,
                       prediction[
                           true_predictions_of_ground_truth(y_pred, y_true, i)].mean(
                           axis=0),
                       width,
                       label='true')
        if any(false_predictions_of_ground_truth(y_pred, y_true, i)):
            axs[i].bar(x + width / 2,
                       prediction[
                           false_predictions_of_ground_truth(y_pred, y_true, i)].mean(
                           axis=0),
                       width,
                       label='false')
        axs[i].set_title(str(i))
        axs[i].legend()
    plt.show()


def plot_layers_activations_for_prediction(prediction, y_pred, y_true):
    x = np.arange(10)  # the label locations
    width = 0.35  # the width of the bars
    fig, axs = plt.subplots(2, 5, figsize=(20, 5))
    axs = axs.flatten()

    for i in range(10):
        if any(true_predictions_of_predicted(y_pred, y_true, i)):
            axs[i].bar(x - width / 2,
                       prediction[
                           true_predictions_of_predicted(y_pred, y_true, i)].mean(
                           axis=0),
                       width,
                       label='true')
        if any(false_predictions_of_predicted(y_pred, y_true, i)):
            axs[i].bar(x + width / 2,
                       prediction[
                           false_predictions_of_predicted(y_pred, y_true, i)].mean(
                           axis=0),
                       width,
                       label='false')
        axs[i].set_title(str(i))
        axs[i].legend()
    plt.show()


def plot_suspicions_stats(y_susp, y_true, pred_label, class_names):
    print("Analysis of cases predicted as", class_names[pred_label])
    misclassifications = np.sum(y_true != pred_label)
    num_detect_suspitions = np.sum(np.invert(y_susp))
    num_correct_suspitions = np.sum(y_true[np.invert(y_susp)] != pred_label)
    #   normalize values
    p_misclassifications = misclassifications / len(y_true)
    p_detect_suspitions = num_detect_suspitions / len(y_true)
    p_correct_suspitions = num_correct_suspitions / num_detect_suspitions
    p_correct_suspitions_from_all = num_correct_suspitions / len(y_true)
    # print("% of missclassification: {:.2}%".format(p_misclassifications))
    # print("% detected suspicions:   {:.2}%".format(p_detect_suspitions))
    # print(
    #     "% correctly detected suspicions: {:.2}%".format(p_correct_suspitions))
    # print("Rate of correctly detected missclassifications: {:.2}%".format(
    #     p_correct_suspitions_from_all))
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 10))
    fig, ax = plt.subplots(figsize=(10, 3))
    data = [np.sum(y_true == i) for i in range(10)]
    plt.bar(range(10), data, color=colors)
    plt.xticks(range(10), class_names)
    plt.show()

    explode = (0.1, 0)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    axs[0].pie([p_misclassifications, 1 - p_misclassifications], explode=explode,
               labels=["false", "true"], autopct='%1.1f%%', shadow=True, colors=["green", "orange"],
               startangle=90)
    axs[0].title.set_text('Classification ({} cases)'.format(len(y_true)))
    axs[1].pie([p_detect_suspitions, 1 - p_detect_suspitions], explode=explode,
               labels=["detected as suspicious", "detected as not suspicious"],
               autopct='%1.1f%%', shadow=True, startangle=90)
    axs[1].title.set_text('Suspicious classification ({} cases)'.format(len(y_true)))
    axs[2].pie([p_correct_suspitions, 1 - p_correct_suspitions], explode=explode,
               labels=["true suspicions", "false suspicions"], colors=["cornflowerblue", "orangered"],
               autopct='%1.1f%%', shadow=True, startangle=90)
    axs[2].title.set_text('Suspicious classification ({} cases)'.format(num_detect_suspitions))
    axs[3].pie(
        [p_correct_suspitions_from_all, 1 - p_correct_suspitions_from_all], explode=explode,
        labels=["true suspicions", "rest of cases"], autopct='%1.1f%%',
        shadow=True, startangle=90)
    axs[3].title.set_text('Suspicious classification ({} cases)'.format(len(y_true)))
    plt.show()


def visualize_path(layers_idx, atlas_images, atlas_labels, image, label, pred, class_names):
    print('original:', str(class_names[label]), '  predicted:', str(class_names[pred]))
    plt.imshow(image, cmap='gray')
    plt.show()

    fig, axs = plt.subplots(len(layers_idx), 5,
                            figsize=(10, 2 * len(layers_idx)))
    for y, layer_idx in enumerate(layers_idx):
        for x, idx in enumerate(layer_idx):
            axs[y][x].imshow(atlas_images[idx], cmap='gray')
            axs[y][x].set_title(class_names[atlas_labels[idx]])
            axs[y][x].set_axis_off()

    plt.show()


def visualize_path_lrp(layers_idx, atlas_images, atlas_labels, image, label,
                       pred, analyzer, class_names):
    print('original:', str(class_names[label]), ' predicted:', str(class_names[pred]))
    analyzed_image = analyzer.analyze([image])
    analyzed_image = analyzed_image['input_layer']
    analyzed_image = analyzed_image.sum(
        axis=np.argmax(np.asarray(analyzed_image.shape) == 3))
    analyzed_image /= np.max(np.abs(analyzed_image))
    plt.imshow(analyzed_image[0], cmap="seismic", clim=(-1, 1))
    plt.show()

    fig, axs = plt.subplots(len(layers_idx), 5,
                            figsize=(10, 2 * len(layers_idx)))
    for y, layer_idx in enumerate(layers_idx):
        for x, idx in enumerate(layer_idx):
            analyzed_image = analyzer.analyze([atlas_images[idx]])
            analyzed_image = analyzed_image['input_layer']
            analyzed_image = analyzed_image.sum(
                axis=np.argmax(np.asarray(analyzed_image.shape) == 3))
            analyzed_image /= np.max(np.abs(analyzed_image))
            axs[y][x].imshow(analyzed_image[0], cmap="seismic", clim=(-1, 1))
            axs[y][x].set_title(str(class_names[atlas_labels[idx]]))
            axs[y][x].set_axis_off()

    plt.show()


def true_predictions_of_ground_truth(y_pred, y_true, label):
    return np.logical_and(y_true == label, y_true == y_pred)


def false_predictions_of_ground_truth(y_pred, y_true, label):
    return np.logical_and(y_true == label, y_true != y_pred)


def true_predictions_of_predicted(y_pred, y_true, label):
    return np.logical_and(y_pred == label, y_true == y_pred)


def false_predictions_of_predicted(y_pred, y_true, label):
    return np.logical_and(y_pred == label, y_true != y_pred)
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
    data = [np.sum(y_true == i) for i in range(len(class_names))]
    plt.bar(range(len(class_names)), data, color=colors)
    plt.xticks(range(len(class_names)), class_names)
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


def visualize_image(image, pred, label, gs=None):
    ax = gs.subgridspec(1, 1).subplots()
    ax.imshow(image)
    ax.set_axis_off()


def visualize_path(layers_idx, atlas_images, atlas_labels, class_names, gs=None):
    if gs is None:
        fig, axs = plt.subplots(len(layers_idx), 5,
                                figsize=(10, 2 * len(layers_idx)))
    else:
        axs = gs.subgridspec(len(layers_idx), 5, wspace=0, hspace=0).subplots()

    for y, layer_idx in enumerate(layers_idx):
        for x, idx in enumerate(layer_idx):
            axs[y][x].imshow(atlas_images[idx], cmap='gray')
            axs[y][x].set_title(class_names[int(atlas_labels[idx])])
            axs[y][x].set_axis_off()

    if gs is None:
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


def clip(R, clim1, clim2):
    delta = list(np.array(clim2) - np.array(clim1))
    R = R / (np.mean(R ** 4) ** 0.25)  # normalization
    R = R - np.clip(R, clim1[0], clim1[1])  # sparsification
    R = np.clip(R, delta[0], delta[1]) / delta[1]  # thresholding
    return R


def get_alpha(x, p=1):
    x = x ** p
    return x


"""
Plot relevances betweeen images by BiLRP

"""

def plot_relevances_full(R, x1, x2, stride, curvefac=1., gs=None, title=None):
    indices = np.indices(R.shape)
    inds_all = [(i, R[i[0], i[1], i[2], i[3]]) for i in
                indices.reshape((4, np.prod(indices.shape[1:]))).T]

    fname = None
    clip_func = lambda x: get_alpha(
        clip(x, clim1=[-0.25, 0.25], clim2=[-13, 13]), p=2)
    plot_relevances(inds_all, x1, x2, clip_func, stride, fname, curvefac, gs, title)


def plot_relevances(c, x1, x2, clip_func, stride, fname=None, curvefac=1., gs=None, title=None):
    h, w, channels = x1.shape if len(x1.shape) == 3 else list(x1.shape) + [1]
    wgap, hpad = int(0.05 * w), int(0.6 * w)
    if gs is not None:
        ax = gs.subgridspec(1, 1, wspace=0, hspace=0).subplots()
    else:
        fig, ax = plt.subplots(figsize=(5, 10))
    # plt.ylim(-hpad, h + hpad - 1)
    # plt.xlim(0, w * 2 + wgap - 1)

    mid = np.ones([h, wgap, channels])
    X = np.concatenate([x1.reshape(h, w, channels).squeeze(), mid,
                        x2.reshape(h, w, channels).squeeze()], axis=1)[::-1]
    plt.imshow(X, cmap='gray', vmin=-1, vmax=1)
    plt.title(title)
    if len(stride) == 2:
        stridex = stride[0]
        stridey = stride[1]
    else:
        stridex = stridey = stride[0]

    relevance_array = np.array([i[1] for i in c])
    indices = [i[0] for i in c]

    alphas = clip_func(relevance_array)
    inds_plotted = []

    for indx, alpha, s in zip(indices, alphas, relevance_array):
        i, j, k, l = indx[0], indx[1], indx[2], indx[3]

        if alpha > 0.:
            xm = int(w / 2) + 6
            xa = stridey * j + (stridey / 2 - 0.5) - xm
            xb = stridey * l + (stridey / 2 - 0.5) - xm + w + wgap
            ya = h - (stridex * i + (stridex / 2 - 0.5))
            yb = h - (stridex * k + (stridex / 2 - 0.5))
            ym = (0.8 * (ya + yb) - curvefac * int(h / 6))
            ya -= ym
            yb -= ym
            lin = np.linspace(0, 1, 25)
            plt.plot(xa * lin + xb * (1 - lin) + xm,
                     ya * lin ** 2 + yb * (1 - lin) ** 2 + ym,
                     color='red' if s > 0 else 'blue', alpha=alpha)

        inds_plotted.append(((i, j, k, l), s))

    plt.axis('off')

    # if fname:
    #     plt.savefig(fname, dpi=200)
    # else:
    #     plt.show()
    # plt.close()


def plot_lrp(img, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img, cmap='seismic', clim=(-1, 1))
    ax.set_title(title if title else "")
    ax.axis('off')


def plot_rgb(img, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img)
    ax.set_title(title if title else "")
    ax.axis('off')


def plot_gray(img, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img, cmap='gray')
    ax.set_title(title if title else "")
    ax.axis('off')
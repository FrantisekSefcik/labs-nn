import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

COLOR_PALETTE = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


def plot_lrp(img, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    analyzed_image = img.sum(axis=np.argmax(np.asarray(img.shape) == 3))
    analyzed_image /= np.max(np.abs(analyzed_image))

    ax.imshow(analyzed_image, cmap='seismic', clim=(-1, 1))
    ax.set_title(title if title else "", fontsize=15)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def plot_pair_lrp(x1, x2, analyzer, gs=None, title=None):
    analysis = analyzer.analyze(np.array(x1, x2))
    a1 = analysis['input_layer'][0]
    a2 = analysis['input_layer'][1]

    if gs is not None:
        axs = gs.subgridspec(1, 2, wspace=0, hspace=0).subplots()
    else:
        fig, axs = plt.subplots(figsize=(15, 20))
    plot_lrp(a1, axs[0])
    plot_lrp(a2, axs[1])


def plot_rgb(img, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img)
    ax.set_title(title if title else "", fontsize=15)
    ax.axis('off')


def plot_gray(img, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img, cmap='gray')
    ax.set_title(title if title else "", fontsize=15)
    ax.axis('off')


def plot_rgb_images(images_arr, num_imgs=5, titles=None):
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 20))
    axes = axes.flatten()
    for i, (img, ax) in enumerate(zip(images_arr, axes)):
        ax.imshow(img)
        ax.axis('off')
        if titles:
            txt_top = [l + '\n' for l in titles[i]]
            ax.set_title(''.join(txt_top),
                         horizontalalignment='left')
    plt.tight_layout()
    plt.show()


def plot_gray_images(images_arr, num_imgs=5):
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = img.reshape((img.shape[0], img.shape[1]))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_analysis_images(images_arr, num_imgs=5):
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = img.reshape((img.shape[0], img.shape[1]))
        ax.imshow(img, cmap='seismic', clim=(-1, 1))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_rgb_decomposition(rgb_image, cmap='gray', clim=None):
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip([r, g, b], axes):
        ax.imshow(img, cmap=cmap, clim=clim)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_model_performance(y, pred):
    """
    Plot confusion matrix and classification report.
    :param y: array with true values
    :param pred: array with predicted values
    """
    cm = confusion_matrix(y, pred)
    print("Model performance:")
    print(classification_report(y, pred, target_names=['LGG', 'HGG']))
    fig, ax = plt.subplots(figsize=(3, 3))
    plot_confusion_matrix(cm, ax=ax, labels=['LGG', 'HGG'])
    plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues,
                          ax=None, labels=[]):

    ax.matshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    ax.set_xticklabels([''] + labels, fontsize=15)
    ax.set_yticklabels([''] + labels, fontsize=15)
    ax.set_ylabel('True label', fontsize=15)
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.text(0, 0, cm[0][0], color='black', ha='center', va='center', fontsize=17)
    ax.text(1, 0, cm[0][1], color='black', ha='center', va='center', fontsize=17)
    ax.text(0, 1, cm[1][0], color='black', ha='center', va='center', fontsize=17)
    ax.text(1, 1, cm[1][1], color='black', ha='center', va='center', fontsize=17)


def visualize_single_image(image, pred, label, class_names, gs=None):
    if gs is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = gs.subgridspec(1, 1).subplots()
    plot_rgb(image, ax, "true label: " + class_names[int(label)] + "\nprediction: " + class_names[int(pred)])


def visualize_single_lrp(image, pred, label, analyzer, class_names, gs=None):
    ax = gs.subgridspec(1, 1).subplots()
    analyzed_image = analyzer.analyze([image])
    analyzed_image = analyzed_image['input_layer']
    plot_lrp(analyzed_image[0], ax, "true label: " + class_names[int(label)] + "\nprediction: " + class_names[int(pred)])


def visualize_path(layers_idx, atlas_images, atlas_labels, class_names, gs=None):
    if gs is None:
        fig, axs = plt.subplots(len(layers_idx), 5,
                                figsize=(10, 2 * len(layers_idx)))
    else:
        axs = gs.subgridspec(len(layers_idx), 5, wspace=0, hspace=0).subplots()

    for y, layer_idx in enumerate(layers_idx):
        for x, idx in enumerate(layer_idx):
            plot_rgb(atlas_images[idx], axs[y][x], class_names[int(atlas_labels[idx])])

    if gs is None:
        plt.show()


def visualize_path_lrp(layers_idx, atlas_images, atlas_labels, analyzer, class_names, gs):
    if gs is None:
        fig, axs = plt.subplots(len(layers_idx), 5,
                                figsize=(10, 2 * len(layers_idx)))
    else:
        axs = gs.subgridspec(len(layers_idx), 5, wspace=0, hspace=0).subplots()

    for y, layer_idx in enumerate(layers_idx):
        for x, idx in enumerate(layer_idx):
            analyzed_image = analyzer.analyze([atlas_images[idx]])
            analyzed_image = analyzed_image['input_layer']
            plot_lrp(analyzed_image[0], axs[y][x], str(class_names[int(atlas_labels[idx])]))
    if gs is None:
        plt.show()


def plot_image_grid(images,
                    grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    figsize=None,
                    dpi=224):
    n_rows = len(grid)
    n_cols = len(grid[0]) + 1
    if figsize is None:
        figsize = (n_cols, n_rows + 1)

    plt.clf()
    plt.rc("font", family="sans-serif")

    plt.figure(figsize=figsize)
    for r in range(n_rows):
        ax = plt.subplot2grid(shape=[n_rows + 1, n_cols], loc=[r + 1, 0])
        ax.imshow(images[r])
        ax.set_xticks([])
        ax.set_yticks([])

        # row labels

        if row_labels_left != []:
            txt_left = [l + '\n' for l in row_labels_left[r]]
            ax.set_ylabel(
                ''.join(txt_left),
                rotation=0,
                verticalalignment='center',
                horizontalalignment='right',
            )

        for c in range(0, n_cols - 1):
            ax = plt.subplot2grid(shape=[n_rows + 1, n_cols + 1],
                                  loc=[r + 1, c + 1])
            if grid[r][c] is not None:
                ax.imshow(grid[r][c], interpolation='none')
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

            if c == n_cols - 2:
                if row_labels_right != []:
                    txt_right = [l + '\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(
                        ''.join(txt_right),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='left'
                    )

    if file_name is None:
        plt.show()
    else:
        print('Saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi)
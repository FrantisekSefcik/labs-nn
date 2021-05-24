import numpy as np
import matplotlib.pyplot as plt


def clip(R, clim1, clim2):
    delta = list(np.array(clim2) - np.array(clim1))
    R = R / (np.mean(R ** 4) ** 0.25)  # normalization
    R = R - np.clip(R, clim1[0], clim1[1])  # sparsification
    R = np.clip(R, delta[0], delta[1]) / delta[1]  # thresholding
    return R


def get_alpha(x, p=1):
    x = x ** p
    return x


def visualize_bilrp_path(layers_idx, bilrp_relevances, target_image, atlas_images, atlas_labels, class_names, poolstride, gs=None):

    if gs is None:
        fig = plt.figure(constrained_layout=True, figsize=(4 * len(layers_idx), 10))
        gs = fig.add_gridspec(len(layers_idx), 5)

    for y, layer_idx in enumerate(layers_idx):
        for x, idx in enumerate(layer_idx):
            plot_pair_similar_features(bilrp_relevances[y][x], target_image, atlas_images[idx],
                                       poolstride, gs=gs[y,x], title=class_names[int(atlas_labels[idx])])


def plot_pair_similar_features(R, x1, x2, stride, curvefac=1., gs=None, title=None):
    """
        Plot relevances between image pair

        :param R: N-dimensional array of relevances from BiLRP
        :param x1: image
        :param x2: image
        :param stride: Pooling stride. Example: [8]
        :param curvefac: Level of curve bend
        :param gs: matplotlib gridspec
        :param title: Title of plot
        """
    indices = np.indices(R.shape)
    inds_all = [(i, R[i[0], i[1], i[2], i[3]]) for i in
                indices.reshape((4, np.prod(indices.shape[1:]))).T]

    fname = None
    clip_func = lambda x: get_alpha(clip(x, clim1=[-0.25, 0.25], clim2=[-13, 13]), p=3)
    plot_relevances(inds_all, x1, x2, clip_func, stride, fname, curvefac, gs, title)


def plot_relevances(c, x1, x2, clip_func, stride, fname=None, curvefac=1.,
                    gs=None, title=None):
    h, w, channels = x1.shape if len(x1.shape) == 3 else list(x1.shape) + [1]
    wgap, hpad = int(0.05 * w), int(0.6 * w)
    if gs is not None:
        ax = gs.subgridspec(1, 1, wspace=0, hspace=0).subplots()
    else:
        fig, ax = plt.subplots(figsize=(15, 20))
    plt.ylim(0, h)
    plt.xlim(0, w * 2 + wgap - 1)

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
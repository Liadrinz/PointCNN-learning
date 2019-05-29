import io
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


def tsne_visualize(data_train, label_train):
    colors = []
    for color in range(0x0000ff, 0xff0000, 417785):
        rgb = hex(color)[2:]
        while len(rgb) < 6:
            rgb = '0' + rgb
        colors.append('#' + rgb.upper())
    flatten_data = np.reshape(data_train, [-1, data_train.shape[1] * data_train.shape[2]])

    tsne = manifold.TSNE(n_components=2, random_state=501)
    X_tsne = tsne.fit_transform(flatten_data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    X_norm_list = [[str(val) for val in row] for row in X_norm]
    with open('tsne.json', 'w') as f:
        f.write(json.dumps(X_norm_list))

    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.scatter(
            X_norm[i, 0],
            X_norm[i, 1],
            label_train[i],
            color=colors[label_train[i]])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def reduce_dim(data):
    flatten_data = np.reshape(data, [-1, data.shape[1] * data.shape[2]])
    tsne = manifold.TSNE(n_components=2, random_state=501)
    X_tsne = tsne.fit_transform(flatten_data)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm

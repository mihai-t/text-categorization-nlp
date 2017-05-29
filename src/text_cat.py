import os

import matplotlib.pyplot as plot
import numpy as np
from sklearn import decomposition
from sklearn.svm import SVC

from analysis import DocumentAnalysis, WordIterator
from utils import list_files, PROJECT_PATH

WORD2VEC_ANALYSER = DocumentAnalysis(model_file_name=os.path.join(PROJECT_PATH, "models", "all.model"))


def create_plot(matrix, classes, svm, name):
    pca = decomposition.PCA(n_components=2)
    principal_component = pca.fit_transform(matrix)
    plot.figure()
    labels = set(classes)
    colors = ["red", "blue", "cyan", "magenta"]

    multiples = np.arange(-0.005, 0.005, 0.0001)
    first = multiples[:, np.newaxis] * pca.components_[0, :]
    second = multiples[:, np.newaxis] * pca.components_[1, :]
    grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
    flat_grid = grid.reshape(-1, matrix[0].shape[0])

    z = svm.predict(flat_grid)
    z = z.reshape(grid.shape[:-1])
    z = np.vectorize(lambda x: 1 if x == "romanian_news" else 0)(z)
    plot.contourf(multiples, multiples, z, cmap=plot.cm.Paired)
    plot.axis('off')

    for i, l in enumerate(labels):
        plot.scatter(principal_component[np.where(np.array(classes) == l), 0],
                     principal_component[np.where(np.array(classes) == l), 1], label=l, color=colors[i % len(colors)],
                     edgecolors="black")

    plot.legend(loc='best', numpoints=1)
    plot.title(name)
    plot.grid()

    if not os.path.exists(os.path.join(PROJECT_PATH, "pca")):
        os.mkdir(os.path.join(PROJECT_PATH, "pca"))
    plot.savefig(os.path.join(PROJECT_PATH, "pca", name + '_pca.png'), dpi=125)


def compute_document_vector(file_name, analyser=WORD2VEC_ANALYSER):
    vectors = None
    count = 0
    for _ in WordIterator(file_name):
        count += 1
    for word in WordIterator(file_name):
        w = analyser.vector(word)
        if w is None:
            continue

        if vectors is None:
            vectors = analyser.vector(word) / count
        else:
            vectors += analyser.vector(word) / count

    return vectors / count


def build_training_set(analyser=WORD2VEC_ANALYSER):
    X = []
    Y = []

    for file in list_files(os.path.join(PROJECT_PATH, "text", "romanian_prose")):
        v = compute_document_vector(file, analyser)

        X.append(v)
        Y.append("prose")

    for file in list_files(os.path.join(PROJECT_PATH, "text", "romanian_news")):
        v = compute_document_vector(file, analyser)

        X.append(v)
        Y.append("romanian_news")

    return X, Y


def test(svm):
    X = [compute_document_vector(os.path.join(PROJECT_PATH, "unlabeled", "institut.txt")),
         compute_document_vector(os.path.join(PROJECT_PATH, "unlabeled", "vizita.txt"))]
    Y = ["institut", "vizita"]
    print(Y[0] + " predicted as: " + svm.predict(X[0].reshape(1, -1))[0])
    print(Y[1] + " predicted as: " + svm.predict(X[1].reshape(1, -1))[0])

    return X, Y


KERNELS = ["linear", "poly", "sigmoid", "rbf"]
if __name__ == "__main__":
    X, Y = build_training_set()
    for k in KERNELS:
        svm = SVC(kernel=k)
        svm.fit(X, Y)

        X_test, Y_test = test(svm)

        create_plot(X + X_test, Y + Y_test, svm, k)

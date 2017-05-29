import os

import matplotlib.pyplot as plot
import numpy as np
from sklearn import decomposition
from sklearn.svm import SVC

from analysis import DocumentAnalysis, WordIterator
from utils import list_files, PROJECT_PATH

WORD2VEC_ANALYSER = DocumentAnalysis(model_file_name=os.path.join(PROJECT_PATH, "models", "all.model"))


def create_plot(matrix, classes, svm, name):
    """
    Given a matrix of samples and their correct classes, plots the data on a 2d plot by performing a PCA analysis.
    Furthermore, plots the separating hyperplane computed by a SVM classifier
    :param matrix: Labeled points in the hyperplane
    :param classes: List of correct classes
    :param svm: Trained model
    :param name: name of the plot
    :return:
    """
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
    """
    Computes the document vector of a given sample using the word2vec model
    The document vector is defined as the mean of all the word vectors of the text sample
    :param file_name: given text sample
    :param analyser: trained word2vec model
    :return: the average vector of the document
    """
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
    """
    Given a word2vec analyser, compute the document vectors of the labeled samples
    :param analyser: given word2vec model
    :return: annotated data set
    """
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
    """
    Classify unlabeled samples using the trained svm model
    :param svm: given trained model
    :return: document vectors of the classified documents
    """
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

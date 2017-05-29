import operator
import os

import matplotlib.pyplot as plot
import numpy as np
from sklearn import decomposition

from analysis import DocumentAnalysis
from utils import list_files, corpus_to_model, PROJECT_PATH, corpus_to_pca


def sanitize_label(word):
    """
    Sanitize some romanian characters with ascii
    :param word: given word
    :return: sanitized word
    """
    return word.replace("ă", "a").replace("ş", "s").replace("ţ", "t")


def do_pca(word_dictionary, name, labels=None):
    """
    Create a PCA plot over a list of word vectors
    :param word_dictionary: dictionary <K=word, V=vector>
    :param name: name of the plot
    :param labels: words to be highlighted on the plot
    """
    pca = decomposition.PCA(n_components=2)

    words = list(word_dictionary.keys())
    if labels is None:
        labels = DocumentAnalysis.filter_words(words)

    labels = set(labels)

    matrix = np.array(list(word_dictionary.values()))
    principal_component = pca.fit_transform(matrix)

    plot.figure()
    plot.scatter(principal_component[:, 0], principal_component[:, 1])

    p = [(50, 50), (-50, -50), (50, -50), (-50, 50)]
    for (i, word) in enumerate(labels):
        x = principal_component[words.index(word), 0]
        y = principal_component[words.index(word), 1]

        plot.annotate(
            sanitize_label(word),
            xy=(x, y), xytext=p[i % len(p)],
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    title = os.path.basename(os.path.normpath(name)).replace("_pca.png", "")
    plot.grid()
    if not os.path.exists(os.path.join(PROJECT_PATH, "pca")):
        os.mkdir(os.path.join(PROJECT_PATH, "pca"))
    plot.savefig(name, dpi=125)


if __name__ == "__main__":
    """
    Create PCA plots in order to visualize the vector models
    of all the available samples of prose
    """
    for file in list_files(os.path.join(PROJECT_PATH, "text", "romanian_prose")):
        print("Analysing " + file)
        analyser = DocumentAnalysis(model_file_name=corpus_to_model(file))

        similarities = analyser.similarities(analyser.filter_words(analyser.words()))
        similarities = sorted(similarities.items(), key=operator.itemgetter(1))

        print("Top 10 similarities")
        for s in reversed(similarities[-10:]):
            print(s)

        dictionary = {}
        for word in analyser.words():
            dictionary[word] = analyser.vector(word)

        do_pca(dictionary, corpus_to_pca(file),
               labels=[similarities[-1][0][0], similarities[-1][0][1], similarities[-2][0][0], similarities[-2][0][1]])



import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(FILE_PATH, os.path.pardir))


def list_files(name):
    for f in os.listdir(name):
        yield os.path.join(name, f)

    yield name  # all files


def corpus_to_model(file_name):
    name = os.path.basename(os.path.normpath(file_name)).replace(".txt", "") + ".model"
    return os.path.join(PROJECT_PATH, "models", name)


def corpus_to_pca(file_name):
    name = os.path.basename(os.path.normpath(file_name)).replace(".txt", "") + "_pca.png"
    return os.path.join(PROJECT_PATH, "pca", name)

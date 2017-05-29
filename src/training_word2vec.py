import os

from analysis import DocumentAnalysis
from utils import list_files, corpus_to_model, PROJECT_PATH


def create_model(file_name, model_name):
    print("Analysing " + file_name)
    analyser = DocumentAnalysis(corpus_name=file_name, iter=500, window=20, min_count=5)

    if not os.path.exists(os.path.join(PROJECT_PATH, "models")):
        os.mkdir(os.path.join(PROJECT_PATH, "models"))
    analyser.save(model_name)


if __name__ == "__main__":
    for file in list_files(os.path.join(PROJECT_PATH, "text", "romanian_prose")):
        create_model(file, model_name=corpus_to_model(file))

    create_model(os.path.join(PROJECT_PATH, "text"), model_name=os.path.join(PROJECT_PATH, "models", "all.model"))

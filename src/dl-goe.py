import os

from analysis import DocumentAnalysis
from utils import PROJECT_PATH

WORDS = ["tanti", "miţa", "mam", "mare", "mamiţa", "goe", "conductorul"]

if __name__ == "__main__":
    """
    Perform similarities analysis on "D-l Goe"
    """
    analyser = DocumentAnalysis(model_file_name=os.path.join(PROJECT_PATH, "models", "dlgoe.model"))
    print("Odd one out from " + str(WORDS) + ": " + analyser.does_not_match(WORDS))
    for w in WORDS:
        print(w + "~" + str(analyser.most_similar(w)))

import os

from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer

from utils import PROJECT_PATH

CONJUNCTIONS_PREPOSITIONS = ["în", "să", "cu", "de", "pe", "dar", "s", "n", "cu", "a", "o", "şi", "la", "se", "nu",
                             "da", "l", "îşi", "i", "se", "îl", "iar", "te", "cum", "ce", "d", "ta", "am", "lui", "e",
                             "are", "care", "aşa", "apoi", "cine", "ceva", "zice", "trebuie",
                             "dacă", "mai", "îl", "un", "își", "din", "că", "lu", "mi", "este", "când", "mă"]


class SentenceIterator:
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        tokenizer = RegexpTokenizer(r'\w+')
        if os.path.isdir(self.file_name):
            for path, dirs, files in os.walk(self.file_name):
                for f in files:
                    file = os.path.join(path, f)
                    if os.path.isdir(file):
                        continue

                    for line in open(file, encoding="utf8"):
                        yield [s.lower() for s in tokenizer.tokenize(line)]
        else:
            for line in open(self.file_name, encoding="utf8"):
                yield [s.lower() for s in tokenizer.tokenize(line)]


class WordIterator:
    def __init__(self, file_name):
        self.sentence_iterator = SentenceIterator(file_name)

    def __iter__(self):
        for sentence in self.sentence_iterator:
            for word in sentence:
                yield word


class DocumentAnalysis:
    def __init__(self, model_file_name=None, corpus_name=None, size=150, window=15, min_count=5, iter=20):
        if model_file_name is not None:
            self.load(model_file_name)
        else:
            self.iterator = SentenceIterator(corpus_name)
            self._model = self.__word2vec_model(size, window, min_count, iter)

    def __word2vec_model(self, size, window, min_count=5, iter=20):
        return Word2Vec(sentences=self.iterator,
                        size=size, window=window, min_count=min_count, workers=8, iter=iter)

    def words(self):
        """
        :return: list of words in the corpus
        """
        return list(self._model.wv.vocab.keys())

    @staticmethod
    def filter_words(words):
        return [x for x in words if x not in CONJUNCTIONS_PREPOSITIONS]

    def similarities(self, words):
        similarities = {}
        for i in range(0, len(words)):
            for j in range(i + 1, len(words)):
                similarities[(words[i], words[j])] = self._model.similarity(words[i], words[j])

        return similarities

    def vector(self, word):
        if word in self._model.wv:
            return self._model.wv[word]
        else:
            return None

    def does_not_match(self, words):
        """
        Given a list of words, find the odd one out
        :param words: list of words
        :return: odd one
        """
        return self._model.doesnt_match(words)

    def most_similar(self, word):
        return self._model.most_similar(word)

    def save(self, model_file_name):
        self._model.save(model_file_name)

    def load(self, model_file_name):
        self._model = Word2Vec.load(model_file_name)


if __name__ == "__main__":
    for s in SentenceIterator(os.path.join(PROJECT_PATH, "text", "romanian_prose", "dlgoe.txt")):
        print(s)

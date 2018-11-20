import numpy as np


class DictionaryBuilder:
    def __init__(self, datafile):
        self.datafile = datafile

    def build_dictionary(self):
        """
            build a dictionary contains all words except punctuations in datafile,
            and mark the word frequency of each word.
            build a doc_list contains all documents without date, POS tags
            and punctuations.
        """
        word_dict = dict()
        doc_list = []
        with open(self.datafile, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip().split("  ")[1:]
                if len(line) == 0:
                    continue
                doc = []
                for word_pos in line:
                    word = word_pos.split("/")[0]
                    pos = word_pos.split("/")[1]
                    if pos == "w":
                        continue
                    doc.append(word)
                    word_dict[word] = word_dict.get(word, 0) + 1
                if len(doc) == 0:
                    continue
                doc_list.append(doc)
        return word_dict, doc_list

    def max_word_frequency(self):
        word_dict, _ = self.build_dictionary()
        b = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
        return b


if __name__ == '__main__':
    db = DictionaryBuilder("data/199801_clear_1.txt")
    # test build_dictionary
    word_dict, doc_list = db.build_dictionary()

    # print(doc_list[:5])
    # print(db.max_word_frequency())

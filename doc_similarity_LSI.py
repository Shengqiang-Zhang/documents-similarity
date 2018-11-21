import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")

from gensim import corpora, similarities, models
from dictionary_builder import DictionaryBuilder
import numpy as np

DOCUMENT_FILE = "data/199801_clear_1.txt"
DOCUMENT_FILE_TEST = "data/small_data_for_test.txt"  # small datafile for testing
GENSIM_TEST_FILE = "data/gensim_testfile.txt"  # small file for testing gensim


class LSIModel:
    def __init__(self, datafile: str):
        self._word_dict, self._doc_list = DictionaryBuilder(datafile).build_dictionary()
        self._dictionary = corpora.Dictionary(self._doc_list)
        self._doc_bow_vec = self.doc_bow_vec()
        self._doc_tf_idf_vec = self.doc_tf_idf_vec()

    def print_doc_attr(self):
        print("doc num: ", len(self._doc_list))

    def doc_bow_vec(self):
        _doc_bow_vec = [self._dictionary.doc2bow(doc) for doc in self._doc_list]
        return _doc_bow_vec

    def doc_tf_idf_vec(self):
        tf_idf_model = models.TfidfModel(self._doc_bow_vec)
        _doc_tf_idf_vec = tf_idf_model[self._doc_bow_vec]
        return _doc_tf_idf_vec

    def similarity_vec_tf_idf(self):
        similarity_vec_tf_idf = similarities.Similarity("Similarity-tfidf-index",
                                                        corpus=self._doc_tf_idf_vec,
                                                        num_features=len(self._dictionary))
        return similarity_vec_tf_idf

    def similarity_vec_lsi(self):
        lsi = models.LsiModel(self._doc_tf_idf_vec)
        doc_lsi = lsi[self._doc_tf_idf_vec]
        similarity_vec_lsi = similarities.Similarity("Similarity-LSI-index",
                                                     corpus=doc_lsi,
                                                     num_features=len(self._dictionary))
        for i in doc_lsi:
            print(i)
        for sim in similarity_vec_lsi:
            print(sim)
        return similarity_vec_lsi

    def save_vec_to_file(self, filename: str):
        similarity_vec = self.similarity_vec_lsi()
        vec = []
        for sim in similarity_vec:
            vec.append(sim)
        numpy_vec = np.array(vec)
        np.savetxt(filename, numpy_vec)



if __name__ == '__main__':
    lsi_model = LSIModel(DOCUMENT_FILE)
    lsi_model.print_doc_attr()
    # print(lsi_model._doc_list)
    # print(lsi_model._dictionary)

    # test similarity_vec_tf_idf
    # similarity_vec = lsi_model.similarity_vec()
    # print(similarity_vec)

    # test similarity_vec_lsi
    # doc_similarity_vec = lsi_model.similarity_vec_lsi()
    # print(doc_similarity_vec)

    # test save_model_to_file
    lsi_model.save_vec_to_file("similarity_vec_lsi.txt")

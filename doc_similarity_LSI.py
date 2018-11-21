import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")

from gensim import corpora, similarities, models
from dictionary_builder import DictionaryBuilder

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

    def similarity_vec(self):
        similarity = similarities.Similarity("Similarity-tfidf-index",
                                             corpus=self._doc_tf_idf_vec,
                                             num_features=len(self._dictionary))
        for idx, sim in enumerate(similarity):
            print(idx, sim)
        return similarity

    def test_similarity(self):
        test_data = ["十五大", "会议", "精神"]
        test_bow_vec = self._dictionary.doc2bow(test_data)
        similarity_vec = self.similarity_vec()
        similarity_vec.num_best = 2
        tf_idf_model = models.TfidfModel(self._doc_bow_vec)
        test_tf_idf_vec = tf_idf_model[test_bow_vec]
        print(test_tf_idf_vec)
        print(similarity_vec[test_tf_idf_vec])
        # print(similarity_vec)

    def lsi_model(self):
        lsi = models.LsiModel(self._doc_tf_idf_vec)
        doc_lsi = lsi[self._doc_tf_idf_vec]
        similarity_vec = similarities.Similarity("Similarity-LSI-index",
                                                 corpus=doc_lsi,
                                                 num_features=len(self._dictionary))
        for idx, sim in enumerate(similarity_vec):
            print(idx, sim)


if __name__ == '__main__':
    lsi_model = LSIModel(DOCUMENT_FILE_TEST)
    lsi_model.print_doc_attr()
    # print(lsi_model._doc_list)
    # print(lsi_model._dictionary)

    # test similarity_vec
    # similarity_vec = lsi_model.similarity_vec()
    # print(similarity_vec)

    # test lsi_model
    # lsi_model.lsi_model()

    # test test_similarity
    lsi_model.test_similarity()

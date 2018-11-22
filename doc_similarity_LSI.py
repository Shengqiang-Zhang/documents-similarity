import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")

from gensim import corpora, similarities, models
from dictionary_builder import DictionaryBuilder
import numpy as np

DOCUMENT_FILE = "data/199801_clear_1.txt"
DOCUMENT_FILE_TEST = "data/small_data_for_test.txt"  # small datafile for testing
GENSIM_TEST_FILE = "data/gensim_testfile.txt"  # small file for testing gensim


class DocSimilarityLSI:
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
        return lsi, similarity_vec_lsi

    def save_model_to_file(self, lsi_model, similarity_model, lsi_file: str, sim_file: str):
        lsi_model.save(lsi_file)
        similarity_model.save(sim_file)

    def analyze_result_model(self, doc_id: int, *lsi_model_file, similarity_file, load_model_from_file=False):
        if load_model_from_file:
            lsi_model = models.LsiModel.load(*lsi_model_file)
            similarity_vec_lsi = similarities.Similarity.load(similarity_file)
        else:
            lsi_model = models.LsiModel(self._doc_tf_idf_vec)
            doc_lsi = lsi_model[self._doc_tf_idf_vec]
            similarity_vec_lsi = similarities.Similarity("Similarity-LSI-index",
                                                         corpus=doc_lsi,
                                                         num_features=len(self._dictionary))
        doc_lsi_vec = lsi_model[self._doc_tf_idf_vec[doc_id]]
        doc_sim_list = similarity_vec_lsi[doc_lsi_vec]
        sim_list = np.argsort(-doc_sim_list)
        print("the base doc:")
        print(self._doc_list[doc_id])
        print("three mostly similar doc:")
        for idx, i in enumerate(sim_list[:4]):
            print(idx, self._doc_list[i])


if __name__ == '__main__':
    doc_sim = DocSimilarityLSI(DOCUMENT_FILE)
    # doc_sim.print_doc_attr()

    # lsi_model, doc_similarity_vec = doc_sim.similarity_vec_lsi()
    # doc_sim.save_model_to_file(lsi_model, doc_similarity_vec, "lsi.model", "sim.model")
    doc_sim.analyze_result_model(29, "lsi.model", similarity_file="sim.model", load_model_from_file=True)

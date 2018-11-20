from dictionary_builder import DictionaryBuilder
import numpy as np
import time

DOCUMENT_FILE = "data/199801_clear_1.txt"
DOCUMENT_FILE_TEST = "data/small_data_for_test.txt"  # datafile for testing


class DocSimilarityCalculator:
    def __init__(self, datafile):
        self._word_dict, self._doc_list = DictionaryBuilder(datafile).build_dictionary()
        self._num_doc = len(self._doc_list)
        # self._max_doc_len = self.max_doc_length()
        self.doc_frequency = self.cal_doc_frequency()
        self.word_frequency_list = self.word_frequency_list()
        self.all_docs_vector = self.build_doc_vector()
        self.doc_vec_l2norm = self.save_doc_vec_l2norm()

    def print_doc_attr(self):
        print("num_doc: ", self._num_doc)
        print("doc_frequency: ", self.doc_frequency)
        print("doc_list:", self._doc_list)

    def word_frequency_list(self):
        word_frequency_list = []
        for doc in self._doc_list:
            doc_word_frequency = self.cal_word_frequency_in_one_doc(doc)
            word_frequency_list.append(doc_word_frequency)
        return word_frequency_list

    def build_doc_vector(self):
        """use dict structure to save the doc vector"""
        all_docs_vector = []
        for doc_idx, doc in enumerate(self._doc_list):
            doc_vector_dict = dict()
            for word in doc:
                word_tf_idf = self.cal_tf_idf(word, self.word_frequency_list[doc_idx])
                doc_vector_dict[word] = word_tf_idf
            all_docs_vector.append(doc_vector_dict)
        return all_docs_vector

    def save_doc_vec_l2norm(self):
        """
        to improve the efficiency and avoid repetitive computation,
        we can save l2_norm of each doc vector which will be used in
        cosine distance computation.
        """
        _all_docs_vector = self.build_doc_vector()
        all_docs_vector_l2norm = []
        for doc_vec in _all_docs_vector:
            l2_norm = sum([i * i for i in doc_vec.values()]) ** 0.5
            all_docs_vector_l2norm.append(l2_norm)
        return all_docs_vector_l2norm

    def cosine_distance(self, doc1_id: int, doc2_id: int):
        """calculate cosine distance between two docs"""
        doc1_vec, doc2_vec = self.all_docs_vector[doc1_id], self.all_docs_vector[doc2_id]
        if len(doc1_vec) <= len(doc2_vec):
            base_vec = doc1_vec
            cmp_vec = doc2_vec
        else:
            base_vec = doc2_vec
            cmp_vec = doc1_vec
        inner_product = 0
        for word in base_vec:
            if word in cmp_vec:
                inner_product += base_vec[word] * cmp_vec[word]
        vec_norm_product = self.doc_vec_l2norm[doc1_id] * self.doc_vec_l2norm[doc2_id]
        distance = 0
        try:
            distance = inner_product / vec_norm_product
        except ZeroDivisionError as e:
            print("doc1_id = ", doc1_id, doc1_vec, self._doc_list[doc1_id])
            print("doc2_id = ", doc2_id, doc2_vec, self._doc_list[doc2_id])
            print(e)
        return distance

    def cal_all_docs_similarity(self):
        time_start = time.time()
        doc_similarity_vec = np.zeros((self._num_doc, self._num_doc), dtype=np.float32)
        for i in range(self._num_doc):
            doc_similarity_vec[i, i] = 1
        for i in range(self._num_doc):
            if i % 100 == 0:
                print("processed %d, time %f" % (i, time.time() - time_start))
            for j in range(i):
                doc_similarity = self.cosine_distance(i, j)
                doc_similarity_vec[i][j] = doc_similarity_vec[j][i] = doc_similarity
        return doc_similarity_vec

    def cal_tf_idf(self, word, word_frequency):
        if word not in word_frequency.keys():
            return 0
        word_tf = 1 + np.log(word_frequency[word])
        word_idf = np.log(self._num_doc / self.doc_frequency[word])
        word_tf_idf = word_tf * word_idf
        return word_tf_idf

    def cal_word_frequency_in_one_doc(self, doc):
        word_frequency = dict()
        for word in doc:
            word_frequency[word] = word_frequency.get(word, 0) + 1
        return word_frequency

    def cal_doc_frequency(self):
        word_doc_frequency = dict()
        for doc in self._doc_list:
            doc_set = set(doc)
            for word in doc_set:
                word_doc_frequency[word] = word_doc_frequency.get(word, 0) + 1
        sort_list = sorted(word_doc_frequency.items(), key=lambda d: d[1], reverse=True)
        word_doc_frequency_dict = dict(sort_list)
        return word_doc_frequency_dict


if __name__ == '__main__':
    doc_similarity = DocSimilarityCalculator(DOCUMENT_FILE)
    # doc_similarity.print_doc_attr()

    # print(doc_similarity._num_doc)

    # test cal_doc_frequency
    # print(doc_similarity.cal_doc_frequency())

    # test build_doc_vector
    # doc_vec = doc_similarity.build_doc_vector()
    # print(doc_vec)

    # test cal_doc_similarity
    # doc_similarity_vec = doc_similarity.cal_doc_similarity()
    # print(doc_similarity_vec)

    # test build_doc_vector
    # all_docs_vector_ = doc_similarity.build_doc_vector()
    # print(all_docs_vector_)

    # test cosine distance
    # cosine_distance = doc_similarity.cosine_distance(3, 1)
    # print(cosine_distance)

    # test cal_all_docs_similarity
    all_docs_similarity = doc_similarity.cal_all_docs_similarity()
    print(all_docs_similarity)

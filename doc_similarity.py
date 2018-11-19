from dictionary_builder import DictionaryBuilder
import numpy as np
import time

DOCUMENT_FILE = "data/199801_clear_1.txt"
DOCUMENT_FILE_TEST = "data/small_data_for_test.txt"  # datafile for testing


def cosine_distance(vec1, vec2):
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 0.5 + 0.5 * cos


class DocSimilarityCalculator:
    def __init__(self, datafile):
        self._word_dict, self._doc_list = DictionaryBuilder(datafile).build_dictionary()
        self._num_doc = len(self._doc_list)
        # self._max_doc_len = self.max_doc_length()
        self.doc_frequency = self.cal_doc_frequency()
        self.word_frequency_list = self.word_frequency_list()

    # def print_doc_attr(self):
    #     print("num_doc: ", self._num_doc, " max_doc_len: ", self._max_doc_len)
        # print("doc_frequency: ", self.doc_frequency)
        # print("doc_list:", self._doc_list)

    def word_frequency_list(self):
        word_frequency_list = []
        for doc in self._doc_list:
            doc_word_frequency = self.cal_word_frequency_in_one_doc(doc)
            word_frequency_list.append(doc_word_frequency)
        return word_frequency_list

    def cal_two_docs_similarity(self, doc1_id, doc2_id):
        doc1_word_frequency = self.word_frequency_list[doc1_id]
        doc2_word_frequency = self.word_frequency_list[doc2_id]
        union_wordlist = set(doc1_word_frequency.keys()).union(set(doc2_word_frequency.keys()))

        doc1_vec = np.zeros(len(union_wordlist), dtype=np.float32)
        doc2_vec = np.zeros(len(union_wordlist), dtype=np.float32)
        for idx, word in enumerate(union_wordlist):
            word_doc1_tf_idf = self.cal_tf_idf(word, doc1_word_frequency)
            word_doc2_tf_idf = self.cal_tf_idf(word, doc2_word_frequency)

            doc1_vec[idx] = word_doc1_tf_idf
            doc2_vec[idx] = word_doc2_tf_idf
        doc_similarity = cosine_distance(doc1_vec, doc2_vec)
        # print(doc1)
        # print(doc2)
        # print(union_wordlist)
        # print(doc1_vec)
        # print(doc2_vec)
        return doc_similarity

    def cal_all_docs_similarity(self):
        time_start = time.time()
        doc_similarity_vec = np.zeros((self._num_doc, self._num_doc), dtype=np.float32)
        for i in range(self._num_doc):
            doc_similarity_vec[i, i] = 1
        for i in range(self._num_doc):
            if i % 100 == 0:
                print("processed %d, time %f" % (i, time.time() - time_start))
            for j in range(i):
                doc_similarity = self.cal_two_docs_similarity(i, j)
                doc_similarity_vec[i][j] = doc_similarity_vec[j][i] = doc_similarity
        return doc_similarity_vec

    # def build_doc_vector(self):
    #     doc_list_vec = np.zeros((self._num_doc, self._max_doc_len), dtype=np.float32)
    #     doc_count = 0
    #     for doc in self._doc_list:
    #         doc_vec = np.zeros((1, self._max_doc_len), dtype=np.float32)
    #         word_frequency = self.cal_word_frequency_in_one_doc(doc)
    #         word_count = 0
    #         for word in doc:
    #             word_tf_idf = self.cal_tf_idf(word, word_frequency)
    #             doc_vec[0, word_count] = word_tf_idf
    #             word_count += 1
    #         doc_list_vec[doc_count] = doc_vec
    #         doc_count += 1
    #     return doc_list_vec

    # def max_doc_length(self):
    #     """
    #     calculate max doc length, doc length smaller than
    #     max_doc_length should pad 0 in the end of doc vector
    #     """
    #     maxlen = 0
    #     for doc in self._doc_list:
    #         doc_len = len(doc)
    #         if doc_len > maxlen:
    #             maxlen = doc_len
    #     return maxlen

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

    # def cal_doc_similarity(self):
    #     time_start = time.time()
    #     doc_list_vec = self.build_doc_vector()
    #     doc_similarity_vec = np.zeros((self._num_doc, self._num_doc),
    #                                   dtype=np.float32)
    #     for i in range(self._num_doc):
    #         doc_similarity_vec[i, i] = 1
    #     for i in range(1, self._num_doc):
    #         if i % 100 == 0:
    #             print("%d vector has been processed, use time %f" % (i, (time.time() - time_start)))
    #         for j in range(i):
    #             similarity = cosine_distance(doc_list_vec[i], doc_list_vec[j])
    #             doc_similarity_vec[i][j] = doc_similarity_vec[j][i] = similarity
    #     return doc_similarity_vec


if __name__ == '__main__':
    # test max_doc_length
    doc_similarity = DocSimilarityCalculator(DOCUMENT_FILE)
    # doc_similarity.print_doc_attr()
    # print(doc_similarity._max_doc_len)

    # print(doc_similarity._num_doc)

    # test cal_doc_frequency
    # print(doc_similarity.cal_doc_frequency())

    # test build_doc_vector
    # doc_vec = doc_similarity.build_doc_vector()
    # print(doc_vec)

    # test cal_doc_similarity
    # doc_similarity_vec = doc_similarity.cal_doc_similarity()
    # print(doc_similarity_vec)

    # test cal_doc_similarity
    # docs_similarity = doc_similarity.cal_two_docs_similarity(doc_similarity._doc_list[0], doc_similarity._doc_list[1])
    # print(docs_similarity)
    all_doc_similarity = doc_similarity.cal_all_docs_similarity()
    print(all_doc_similarity)
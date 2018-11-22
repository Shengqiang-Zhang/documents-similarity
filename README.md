We use Vector Space Model(VSM) and Latent Semantic Indexing(LSI) Model to calculate documents similarity based on one part of People's Daily corpora, which contains about 3,000 documents.

# Data
There are two files under `data` directory.
* `199801_clear_1.txt` is the People's Daily corpora.
* `small_data_for_test.txt` is a small dataset just for testing codes.

# Data Preprocessing
See `dictionary_builder.py`

# Model
## VSM
See `doc_similarity_VSM.py`

## LSI Model
See `doc_similarity_LSI.py`

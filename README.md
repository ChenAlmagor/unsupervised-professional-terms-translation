# Unsupervised Word Translation for Fine-Grained Professional Terms.

This repository contains the implementation for the project Unsupervised Word Translation for Fine-Grained Professional Terms.

* **`get_embeddings.sh`** - script for generating embeddings to words in .txt file. requires the pre-trained `fastText` binary file. When referring in the code to a `.vec` file, that means that it requires to apply this script on the words in the `.txt` file with the same prefix. this files are available in the `data` folders. 
* **`parsing.py`** - contains parsing methods.
* **`medical-crawler`** - contains the data crawlers for English, Spanish and French dictionaries, and English-Spanish medical glossary.
* **`neural networks`** - the implementation of the neural networks experiments. Note that it requires to upload the relevant data.
   * for evaluation - download the saved model and load it in `eval_for_medical.py`
* **`inferred-lexicon-by-definitions`** - the implementation of the inferred lexiocons by definitions. 
    * `build_inferred_lexicon.ipynb` - the implementation for building the dictionary and extracting the definitions embeddings. 
    * `encoder-embeddings-alignment` - the implementation for the inferred lexicon by the alignment method.
      * requires to load the encoder embeddings of the inferred lexicon, and the dictionaries words encoder embeddings (are saved during the building method).
      * run `run_icp_for_medical.py` first. then, `create-lexicon-from-indices.py`.
    * `eval-inferred-dict.py`, `compare-by-cosine-distance.py` - evaluating the lexicons.

* **`Non-adversarialTranslation-Medical`** - the implementation for the MBC-ICP variations. 
  * `params.py` - control the trained model setup. uncomment the required variations to run it.
  * `run_icp_for_medical.py` - implementation for training the model. Note that multiple initialization available (with PCA, with random sampling, identity).
  * clustering - the implementation for the clustering method. 
      * run the whole space method (`run_icp_for_medical.py`) before.
      * run `run-clustering.py`.
      * evaluate the result - `eval_for_medical_clustering.py`
    





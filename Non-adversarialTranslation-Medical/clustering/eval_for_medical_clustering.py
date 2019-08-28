
import numpy as np
import utils
import params
#import faiss
from joblib import dump, load

def CountFrequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    for key, value in freq.items():
        print("% d : % d" % (key, value))

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_eval_ex, False)
src_embeddings = src_embeddings.astype('float32')
#index_x = faiss.read_index("%s/%s_clusters_index" % (params.cp_dir, params.src_lang))
index_x = load("%s/%s_clusters_index_sklearn.joblib" % (params.cp_dir, params.src_lang))
#distances_x, clusters_x = index_x.search(src_embeddings, 1)
clusters_x = index_x.predict(src_embeddings)
clusters_x = clusters_x.flatten()

CountFrequency(clusters_x)

tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_eval_ex, False)
#index_y = faiss.read_index("%s/%s_clusters_index" % (params.cp_dir, params.tgt_lang))
#distances_y, clusters_y = index_y.search(src_embeddings, 1)

# np.save("200k-emb-en", src_embeddings)
# np.save("200k-emb-es", tgt_embeddings)

print("%s_%s" % (params.src_lang, params.tgt_lang))
TX_clustering = np.load("%s/%s_%s_T_clusters-medical.npy" % (params.cp_dir, params.src_lang, params.tgt_lang))
TranslatedX = []
for i in range (len(src_embeddings)):
    cur_translation = src_embeddings[i].dot(np.transpose(TX_clustering[clusters_x[i]]))
    TranslatedX.append(cur_translation)
TranslatedX = np.array(TranslatedX)
# TranslatedX = src_embeddings.dot(np.transpose(T))
print(TranslatedX.shape)
cross_dict = utils.load_dictionary('../data-test/medical/test-dict-only-words.txt', src_word2id, tgt_word2id)
utils.get_word_translation_accuracy(params.src_lang, src_word2id, TranslatedX,
                                    params.tgt_lang, tgt_word2id, tgt_embeddings,
                                    params.method, cross_dict, src_id2word)

print("%s_%s" % (params.tgt_lang, params.src_lang))
# T = np.load("%s/%s_%s_T.npy" % (params.cp_dir, params.tgt_lang, params.src_lang))
# TranslatedY = tgt_embeddings.dot(np.transpose(T))
# cross_dict = utils.load_dictionary('data/%s-%s.5000-6500.txt' % (params.tgt_lang, params.src_lang), tgt_word2id, src_word2id)
# utils.get_word_translation_accuracy(params.tgt_lang, tgt_word2id, TranslatedY,
#                                     params.src_lang, src_word2id, src_embeddings,
#                                     params.method, cross_dict)

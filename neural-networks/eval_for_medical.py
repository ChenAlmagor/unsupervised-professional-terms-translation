import numpy as np
import utils
import params


src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.src_lang, params.n_eval_ex, False)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/wiki.%s.vec' % params.tgt_lang, params.n_eval_ex, False)

print("%s_%s" % (params.src_lang, params.tgt_lang))

TranslatedX  = np.load("output/TranslatedX-relu.npy")

cross_dict = utils.load_dictionary('../data-test/medical/test-dict-only-words.txt', src_word2id, tgt_word2id)
utils.get_word_translation_accuracy(params.src_lang, src_word2id, TranslatedX,
                                    params.tgt_lang, tgt_word2id, tgt_embeddings,
                                    params.method, cross_dict, src_id2word)


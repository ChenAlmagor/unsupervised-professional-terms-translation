# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import utils
import params


src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings('data/full-new-test-dict-only-words-lower-%s-emb.vec' % params.src_lang, params.n_eval_ex, full_vocab=False)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings('data/full-new-test-dict-only-words-lower-%s-emb.vec' % params.tgt_lang, params.n_eval_ex, full_vocab=False)

to_translate_src_id2word, to_translate_src_word2id, to_translate_src_embeddings = utils.read_txt_embeddings('data/intersection-1-parsed-%s-emb.vec' % params.src_lang, params.n_eval_ex, False)
translated_tgt_id2word, translated_tgt_word2id, translated_tgt_embeddings = utils.read_txt_embeddings('data/intersection-1-parsed-%s-emb.vec' % params.tgt_lang, params.n_eval_ex, False)

print("%s_%s" % (params.src_lang, params.tgt_lang))

TranslatedX = translated_tgt_embeddings

#uncomment for comparing with the global transformation

# T = np.load("../Non-adversarialTranslation/%s/%s_%s_T.npy" % (params.cp_dir, params.src_lang, params.tgt_lang))
# TranslatedX = to_translate_src_embeddings.dot(np.transpose(T))

cross_dict, existing_emb_tgt_translated, existing_emb_tgt_real, existing_src_id2word = utils.load_dictionary('data/new-test-dict-only-words-no-parsed.txt',
                                                                       src_word2id,
                                                                       tgt_word2id,
                                                                       to_translate_src_word2id,
                                                                       translated_tgt_word2id,
                                                                       TranslatedX,
                                                                       tgt_embeddings,
                                                                       src_embeddings,

                                                                                                             )


utils.get_word_translation_accuracy(params.src_lang, src_word2id, existing_emb_tgt_translated,
                                    params.tgt_lang, tgt_word2id, existing_emb_tgt_real,
                                    params.method, cross_dict, existing_src_id2word)


print("%s_%s" % (params.tgt_lang, params.src_lang))

#uncomment for comparing with the global transformation
TranslatedY = to_translate_src_embeddings

# T = np.load("../Non-adversarialTranslation/%s/%s_%s_T.npy" % (params.cp_dir, params.tgt_lang, params.src_lang))
# TranslatedY = translated_tgt_embeddings.dot(np.transpose(T))
cross_dict, existing_emb_tgt_translated, existing_emb_tgt_real, existing_src_id2word = utils.load_dictionary('data/new-test-dict-only-words-no-parsed-es-en.txt',
                                                                        tgt_word2id,
                                                                       src_word2id,

                                                                       translated_tgt_word2id,
                                                                        to_translate_src_word2id,
                                                                        TranslatedY,
                                                                        src_embeddings,

                                                                       tgt_embeddings,


                                                                                                             )




utils.get_word_translation_accuracy(params.tgt_lang, tgt_word2id, existing_emb_tgt_translated,
                                    params.src_lang, src_word2id, existing_emb_tgt_real,
                                    params.method, cross_dict, existing_src_id2word)

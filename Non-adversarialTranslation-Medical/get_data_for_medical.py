import numpy as np
import utils
import params
import json


def write_to_json_file(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings(params.init_emb_path_src, params.n_init_ex, False)
np.save('data/%s_init' % (params.src_lang), src_embeddings)
write_to_json_file(src_id2word, params.init_src_id2word_path)
write_to_json_file(src_word2id, params.init_src_word2id_path)

tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings(params.init_emb_path_tgt, params.n_init_ex, False)
np.save('data/%s_init' % (params.tgt_lang), tgt_embeddings)
write_to_json_file(tgt_id2word, params.init_tgt_id2word_path)
write_to_json_file(tgt_word2id, params.init_tgt_word2id_path)


src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings(params.training_emb_path_src, params.n_ft_ex, False)
np.save('data/%s_training' % (params.src_lang), src_embeddings)
write_to_json_file(src_id2word, params.training_src_id2word_path)
write_to_json_file(src_word2id, params.training_src_word2id_path)

tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings(params.training_emb_path_tgt, params.n_ft_ex, False)
np.save('data/%s_training' % (params.tgt_lang), tgt_embeddings)
write_to_json_file(tgt_id2word, params.training_tgt_id2word_path)
write_to_json_file(tgt_word2id, params.training_tgt_id2word_path)

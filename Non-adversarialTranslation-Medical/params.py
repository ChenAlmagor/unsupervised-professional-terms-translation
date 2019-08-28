src_lang = "en"
tgt_lang = "es"
icp_init_epochs = 100
icp_train_epochs = 50
icp_ft_epochs = 50
n_pca = 25
n_icp_runs = 15
n_init_ex = 5000
n_ft_ex = 16449
n_eval_ex = 200000
n_processes = 1
method = 'csls_knn_10' # nn|csls_knn_10
cp_dir = "output"

#unsupervised

# init_emb_path_src = '/Users/chenal/Projects/applied-dl-project/Non-adversarialTranslation-Medical/data/freq_unsupervised-en-emb.vec'
# init_emb_path_tgt = '/Users/chenal/Projects/applied-dl-project/Non-adversarialTranslation-Medical/data/freq_unsupervised-es-emb.vec'
init_emb_path_src = './data/init/intersection-with-phrases-en-emb.vec'
init_emb_path_tgt = './data/init/intersection-with-phrases-es-emb.vec'
training_emb_path_src = '/Users/chenal/Projects/applied-dl-project/Non-adversarialTranslation-Medical/data/training/ft-all-en-emb.vec'
training_emb_path_tgt = '/Users/chenal/Projects/applied-dl-project/Non-adversarialTranslation-Medical/data/training/ft-all-es-emb.vec'

#supervised

# init_emb_path_src = './data/training/supervision-ft-new-dataset-en-emb.vec'
# init_emb_path_tgt = './data/training/supervision-ft-new-dataset-es-emb.vec'
# training_emb_path_src = './data/training/supervision-ft-new-dataset-en-emb.vec'
# training_emb_path_tgt = './data/training/supervision-ft-new-dataset-es-emb.vec'


init_src_id2word_path = 'data/init_src_id2word.json'
init_src_word2id_path = 'data/init_src_word2id.json'
init_tgt_id2word_path = 'data/init_tgt_id2word.json'
init_tgt_word2id_path = 'data/init_tgt_word2id.json'

training_src_id2word_path = 'data/training_src_id2word.json'
training_src_word2id_path = 'data/training_src_word2id.json'
training_tgt_id2word_path = 'data/training_tgt_id2word.json'
training_tgt_word2id_path = 'data/training_tgt_word2id.json'

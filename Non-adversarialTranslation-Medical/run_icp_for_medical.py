
import utils
import time
import os
import numpy as np
from icp import ICPTrainer
import params

import json

src_W = np.load('data/%s_init.npy' % (params.src_lang)).T
tgt_W = np.load('data/%s_init.npy' % (params.tgt_lang)).T


data = np.zeros((params.n_icp_runs, 2))

def run_icp(idx_x, idx_y, i):
    icp = ICPTrainer(src_W.copy(), tgt_W.copy(), True, params.n_pca)
    t0 = time.time()
    indices_x, indices_y, rec, bb = icp.train_icp(params.icp_init_epochs, indices_x=idx_x, indices_y=idx_y)
    dt = time.time() - t0
    print("%d: Rec %f BB %d Time: %f" % (i, rec, bb, dt))
    return indices_x, indices_y, rec, bb


def run_icp_with_sample(i):
    sample = np.random.randint(1, src_W.shape[1], size=(10000))
    sample_src_W = np.take(a=src_W.copy(), indices=sample, axis=1)
    sample_tgt_W = np.take(a=tgt_W.copy(), indices=sample, axis=1)
    icp = ICPTrainer(sample_src_W, sample_tgt_W, True, params.n_pca)
    t0 = time.time()
    indices_x, indices_y, rec, bb = icp.train_icp(params.icp_init_epochs, indices_x=np.arange(len(sample)), indices_y=np.arange(len(sample)))
    dt = time.time() - t0
    print("%d: Rec %f BB %d Time: %f" % (i, rec, bb, dt))
    return indices_x, indices_y, rec, bb, sample_src_W, sample_tgt_W



def initialize_with_medical_dict():

    with open(params.init_src_id2word_path, 'r') as f:
        src_id2word = json.load(f)

    with open(params.init_src_word2id_path, 'r') as f:
        src_word2id = json.load(f)

    src_embeddings = src_W.T.copy()

    with open(params.init_tgt_id2word_path, 'r') as f:
        tgt_id2word = json.load(f)

    with open(params.init_tgt_word2id_path, 'r') as f:
        tgt_word2id = json.load(f)

    tgt_embeddings = tgt_W.T.copy()

    #map the indices
    results = []
    for run_i in range(params.n_icp_runs):
        idx_x = [None] * len(src_embeddings)
        idx_y = [None] * len(tgt_embeddings)

        for tgt_word, tgt_word_id in tgt_word2id.items():
            src_word = src_id2word[str(tgt_word_id)]
            src_word_id = src_word2id[src_word]
            idx_x[src_word_id] = tgt_word_id
            idx_y[tgt_word_id] = src_word_id


        idx_x = np.asarray(idx_x)
        assert np.equal(idx_x,np.arange(len(src_embeddings))).all()
        idx_y = np.asarray(idx_y)
        assert np.equal(idx_y,np.arange(len(src_embeddings))).all()
        results += [run_icp(idx_x, idx_y, run_i)]

    #get best result
    min_rec = 1e8
    min_bb = None
    for i, result in enumerate(results):
        indices_x, indices_y, rec, bb = result
        data[i, 0] = rec
        data[i, 1] = bb
        if rec < min_rec:
            best_idx_x = indices_x
            best_idx_y = indices_y
            min_rec = rec
            min_bb = bb
    return (best_idx_x, best_idx_y)


def initialize_with_medical_dict_with_sample():

    #map the indices
    results = []
    for run_i in range(params.n_icp_runs):

        results += [run_icp_with_sample(run_i)]

    #get best result
    min_rec = 1e8
    min_bb = None

    best_sample_src_W = None
    best_sample_tgt_W = None
    best_idx_y = None
    best_idx_x = None
    for i, result in enumerate(results):
        indices_x, indices_y, rec, bb, sample_src_W, sample_tgt_W = result
        data[i, 0] = rec
        data[i, 1] = bb
        if rec < min_rec:
            best_idx_x = indices_x
            best_idx_y = indices_y
            min_rec = rec
            min_bb = bb
            best_sample_src_W = sample_src_W
            best_sample_tgt_W = sample_tgt_W
    print(best_sample_src_W)
    return best_idx_x, best_idx_y, best_sample_src_W, best_sample_tgt_W



#best_idx_x, best_idx_y, best_sample_src_W, best_sample_tgt_W = initialize_with_medical_dict_with_sample()

#best_idx_x, best_idx_y = initialize_with_medical_dict()

best_idx_x, best_idx_y = (np.arange(src_W.shape[1]), np.arange(src_W.shape[1]))
idx = np.argmin(data[:, 0], 0)
print("Init - Achieved: Rec %f BB %d" % (data[idx, 0], data[idx, 1]))
icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
_, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, True, indices_x=best_idx_x, indices_y=best_idx_y)
print("Training - Achieved: Rec %f BB %d" % (rec, bb))
src_W = np.load('data/%s_training.npy' % (params.src_lang)).T
tgt_W = np.load('data/%s_training.npy' % (params.tgt_lang)).T
icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
icp_ft.icp.TX = icp_train.icp.TX
icp_ft.icp.TY = icp_train.icp.TY
indices_x, indices_y, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)
print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
TX = icp_ft.icp.TX
TY = icp_ft.icp.TY

if not os.path.exists(params.cp_dir):
    os.mkdir(params.cp_dir)


np.save("%s/%s_%s_T-medical" % (params.cp_dir, params.src_lang, params.tgt_lang), TX)
np.save("%s/%s_%s_T-medical" % (params.cp_dir, params.tgt_lang, params.src_lang), TY)

np.save('indices_x', indices_x)
np.save('indices_y', indices_y)



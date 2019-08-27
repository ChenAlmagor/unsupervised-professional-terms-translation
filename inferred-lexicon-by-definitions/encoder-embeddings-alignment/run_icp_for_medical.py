# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import utils

import time
import os
import multiprocessing
import sys
import numpy as np
from icp import ICPTrainer
import params
import tqdm
import pandas as pd
import json

src_W = np.load('data/embedding_reciprocal_%s.npy' % (params.src_lang)).T
tgt_W = np.load('data/embedding_reciprocal_%s.npy' % (params.tgt_lang)).T


data = np.zeros((params.n_icp_runs, 2))



best_idx_x, best_idx_y = (np.arange(src_W.shape[1]), np.arange(src_W.shape[1]))

idx = np.argmin(data[:, 0], 0)
icp_init = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
_, _, rec, bb = icp_init.train_icp(1, is_init=True, indices_x=best_idx_x, indices_y=best_idx_y)
print("Init - Achieved: Rec %f BB %d" % (rec, bb))
src_W = np.load('data/emb_%s.npy' % (params.src_lang)).T
tgt_W = np.load('data/emb_%s.npy' % (params.tgt_lang)).T
tgt_W = tgt_W[:,:src_W.shape[1]]

icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
icp_train.icp.TX = icp_init.icp.TX
icp_train.icp.TY = icp_init.icp.TY

_, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, is_init=False)

print("Training - Achieved: Rec %f BB %d" % (rec, bb))
icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
icp_ft.icp.TX = icp_train.icp.TX
icp_ft.icp.TY = icp_train.icp.TY
ind_x, ind_y, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)
print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
TX = icp_ft.icp.TX
TY = icp_ft.icp.TY

if not os.path.exists(params.cp_dir):
    os.mkdir(params.cp_dir)



np.save('indices_x_sentences-identity', np.array(ind_x))
np.save('indices_y_sentences-identity', np.array(ind_y))

np.save("%s/%s_%s_T-medical-match-sentences-identity-init" % (params.cp_dir, params.src_lang, params.tgt_lang), TX)
np.save("%s/%s_%s_T-medical-match-sentences-identity-init" % (params.cp_dir, params.tgt_lang, params.src_lang), TY)



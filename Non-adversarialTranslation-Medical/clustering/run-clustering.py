#import faiss
from sklearn.cluster import KMeans

import numpy as np
import params
from icp import ICPTrainer
import params
import os
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

TX = np.load('output/en_es_T-medical-unsupervised-before.npy')
TY = np.load('output/es_en_T-medical-unsupervised-before.npy')

x = np.load('data/%s_training.npy' % (params.src_lang)).astype('float32')
y_for_training = np.load('data/%s_training.npy' % (params.tgt_lang)).astype('float32')
y_for_clustering= x.dot(np.transpose(TX))


print(x.shape)
print(y_for_clustering.shape)

n= x.shape[0]
d = x.shape[1]
ncentroids = 7 #3 unsupervised also1
niter = 20
verbose = True
#
# kmeans_x = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
# kmeans_x.train(x)
# distances_x, clusters_x = kmeans_x.index.search(x, 1)
kmeans_x = KMeans(n_clusters=ncentroids)
kmeans_x.fit(x)
clusters_x = kmeans_x.labels_

print(CountFrequency(clusters_x.flatten()))


# kmeans_y = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
# kmeans_y.train(y_for_clustering)
# distances_y, clusters_y = kmeans_y.index.search(y_for_clustering, 1)
kmeans_y = KMeans(n_clusters=ncentroids)
kmeans_y.fit(y_for_clustering)
clusters_y = kmeans_y.labels_
print(CountFrequency(clusters_y.flatten()))

dump(kmeans_x, "%s/%s_clusters_index_sklearn.joblib"  % (params.cp_dir, params.src_lang) )
dump(kmeans_y, "%s/%s_clusters_index_sklearn.joblib"  % (params.cp_dir, params.tgt_lang))

# indices_x = np.load('indices_x_medical_unsupervised_before.npy')
# indices_y = np.load('indices_y_medical_unsupervised_before.npy')


indices_x = np.arange(n)
indices_y = np.arange(n)

#assert len(indices_x) == len(indices_y) == len(x) == len(y)

x_y_clusters_match = np.zeros((ncentroids, ncentroids)).astype('int32')
clusters_x = clusters_x.flatten()
clusters_x_emb = []
for i in range(ncentroids): clusters_x_emb.append([])
for i in range(len(indices_x)):
    y_idx = indices_x[i]
    clusters_x_emb[clusters_x[i]].append(x[i])
    if (indices_y[y_idx]) == i:
        x_y_clusters_match[clusters_x[i]][clusters_y[indices_x[i]]]+=1

print(np.array(clusters_x_emb[0]).shape)

y_x_clusters_match = np.zeros((ncentroids, ncentroids)).astype('int32')
clusters_y = clusters_y.flatten()
clusters_y_emb = []
# distances_y_training, clusters_y_training = kmeans_y.index.search(y_for_training, 1)
clusters_y_training = kmeans_y.predict(y_for_training)

for i in range(ncentroids): clusters_y_emb.append([])

for i,v in enumerate(clusters_y_emb): clusters_y_emb[i]=[]

for i in range(len(indices_y)):
    y_idx = indices_y[i]
    # clusters_y_emb[clusters_y[i]].append(y_for_training[i])
    if (indices_x[y_idx]) == i:
        y_x_clusters_match[clusters_y[i]][clusters_x[indices_y[i]]]+=1

clusters_y_training = clusters_y_training.flatten()
print(CountFrequency(clusters_y_training.flatten()))


for i in range(len(clusters_y_training)):
    clusters_y_emb[clusters_y_training[i]].append(y_for_training[i])


print(np.array(clusters_y_emb[0]).shape)


print(x_y_clusters_match)
print(y_x_clusters_match)

x_cluster_maps = x_y_clusters_match.argmax(axis=1)
y_cluster_maps = y_x_clusters_match.argmax(axis=1)

print('******')
print(x_cluster_maps)
print(y_cluster_maps)
#check that the clusters map is reciprocal
print(len((np.where(x_cluster_maps[y_cluster_maps] == np.arange(ncentroids)))[0]))

assert((x_cluster_maps[y_cluster_maps] == np.arange(ncentroids)).all())




TX_clusters = [None for i in range(ncentroids)]
TY_clusters = [None for i in range(ncentroids)]
print(TX_clusters)

counter = 0

for i in range (ncentroids):
    #TODO: need to adjust the sizes of the clusters - option 1: just cut the end of the bigger, option 2: init step of choosing the best from the bigger cluster

    cur_cluster_x = i
    #cur_cluster_y = i
    cur_cluster_y = x_cluster_maps[i]

    #if(cur_cluster_x == y_cluster_maps[cur_cluster_y]):
    # cur_cluster_x_emb = []
    # cur_cluster_y_emb = []
    # for c in range(len(x)):
    #     if (clusters_x[c] == cur_cluster_x):
    #         cur_cluster_x_emb.append(x[c])
    #         cur_cluster_y_emb.append(y_for_training[c])

    src_W = np.array(clusters_x_emb[cur_cluster_x]).T
    tgt_W = np.array(clusters_y_emb[cur_cluster_y]).T

    # src_W = np.array(cur_cluster_x_emb).T
    # tgt_W = np.array(cur_cluster_y_emb).T

    cluster_size = min(src_W.shape[1], tgt_W.shape[1])
    src_W = src_W[:,:cluster_size]
    tgt_W = tgt_W[:,:cluster_size]

    print(src_W.shape)
    print(tgt_W.shape)


    icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])

    #TODO: option 2: can initialize with the reciprocal pers of the cluster
    # icp_train.icp.TX = TX
    # icp_train.icp.TY = TY

    icp_train.icp.TX = TX
    icp_train.icp.TY = TY

    _, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, is_init=False)
    #_, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, is_init=True, indices_x=np.arange(src_W.shape[1]), indices_y=np.arange(src_W.shape[1]))

    print("Training - Achieved: Rec %f BB %d" % (rec, bb))
    icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
    icp_ft.icp.TX = icp_train.icp.TX
    icp_ft.icp.TY = icp_train.icp.TY
    ind_x, ind_y, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)
    print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
    TX_clusters[cur_cluster_x] = icp_ft.icp.TX
    TY_clusters[cur_cluster_y] = icp_ft.icp.TY
    # else:
    #     counter+=1
    #     TX_clusters[cur_cluster_x] = TX
    #     TY_clusters[cur_cluster_y] = TY

if not os.path.exists(params.cp_dir):
    os.mkdir(params.cp_dir)

print('NON_ADJUSTED', counter)
np.save("%s/%s_%s_T_clusters-medical" % (params.cp_dir, params.src_lang, params.tgt_lang), TX_clusters)
np.save("%s/%s_%s_T_clusters-medical" % (params.cp_dir, params.tgt_lang, params.src_lang), TY_clusters)
# faiss.write_index(kmeans_x.index, "%s/%s_clusters_index" % (params.cp_dir, params.src_lang))
# faiss.write_index(kmeans_y.index, "%s/%s_clusters_index" % (params.cp_dir, params.tgt_lang))
#





import faiss
import numpy as np
import params
from icp import ICPTrainer
import params
import os

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



def match_by_reciprocal_pairs():
    x = np.load('data/%s_training.npy' % (params.src_lang)).astype('float32')
    y = np.load('data/%s_training.npy' % (params.tgt_lang)).astype('float32')

    n= x.shape[0]
    d = x.shape[1]
    ncentroids = 4
    niter = 20
    verbose = True

    kmeans_x = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans_x.train(x)
    distances_x, clusters_x = kmeans_x.index.search(x, 1)
    print(CountFrequency(clusters_x.flatten()))


    kmeans_y = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans_y.train(y)
    distances_y, clusters_y = kmeans_y.index.search(y, 1)
    print(CountFrequency(clusters_y.flatten()))


    indices_x = np.load('indices_x.npy')
    indices_y = np.load('indices_y.npy')

    assert len(indices_x) == len(indices_y) == len(x) == len(y)

    x_y_clusters_match = np.zeros((ncentroids, ncentroids))
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
    for i in range(ncentroids): clusters_y_emb.append([])

    for i,v in enumerate(clusters_y_emb): clusters_y_emb[i]=[]

    for i in range(len(indices_y)):
        y_idx = indices_y[i]
        clusters_y_emb[clusters_y[i]].append(y[i])
        if (indices_x[y_idx]) == i:
            y_x_clusters_match[clusters_y[i]][clusters_x[indices_y[i]]]+=1

    print(np.array(clusters_y_emb[0]).shape)


    print(x_y_clusters_match)
    print(y_x_clusters_match)

    x_cluster_maps = x_y_clusters_match.argmax(axis=1)
    y_cluster_maps = y_x_clusters_match.argmax(axis=1)

    #check that the clusters map is reciprocal
    assert((x_cluster_maps[y_cluster_maps] == np.arange(ncentroids)).all())



    TX = np.load('output/en_es_T-medical.npy')
    TY = np.load('output/es_en_T-medical.npy')

    TX_clusters = [None for i in range(ncentroids)]
    TY_clusters = [None for i in range(ncentroids)]
    print(TX_clusters)

    for i in range (ncentroids):

        cur_cluster_x = i
        cur_cluster_y = x_cluster_maps[i]

        src_W = np.array(clusters_x_emb[cur_cluster_x]).T
        tgt_W = np.array(clusters_y_emb[cur_cluster_y]).T

        cluster_size = min(src_W.shape[1], tgt_W.shape[1])
        src_W = src_W[:,:cluster_size]
        tgt_W = tgt_W[:,:cluster_size]

        print(src_W.shape)
        print(tgt_W.shape)


        icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])

        icp_train.icp.TX = TX
        icp_train.icp.TY = TY

        _, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, is_init=False)

        print("Training - Achieved: Rec %f BB %d" % (rec, bb))
        icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
        icp_ft.icp.TX = icp_train.icp.TX
        icp_ft.icp.TY = icp_train.icp.TY
        ind_x, ind_y, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)
        print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
        TX_clusters[cur_cluster_x] = icp_ft.icp.TX
        TY_clusters[cur_cluster_y] = icp_ft.icp.TY
        #
    if not os.path.exists(params.cp_dir):
        os.mkdir(params.cp_dir)



    np.save("%s/%s_%s_T_clusters-medical" % (params.cp_dir, params.src_lang, params.tgt_lang), TX_clusters)
    np.save("%s/%s_%s_T_clusters-medical" % (params.cp_dir, params.tgt_lang, params.src_lang), TY_clusters)
    faiss.write_index(kmeans_x.index, "%s/%s_clusters_index" % (params.cp_dir, params.src_lang))
    faiss.write_index(kmeans_y.index, "%s/%s_clusters_index" % (params.cp_dir, params.tgt_lang))




def match_by_whole_space_transformation():
    TX = np.load('output/en_es_T-medical-unsupervised-before.npy')
    TY = np.load('output/es_en_T-medical-unsupervised-before.npy')

    x = np.load('data/%s_training.npy' % (params.src_lang)).astype('float32')
    y_for_training = np.load('data/%s_training.npy' % (params.tgt_lang)).astype('float32')
    y_for_clustering = x.dot(np.transpose(TX))



    n = x.shape[0]
    d = x.shape[1]
    ncentroids = 3
    niter = 20
    verbose = True

    kmeans_x = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans_x.train(x)
    distances_x, clusters_x = kmeans_x.index.search(x, 1)

    print(CountFrequency(clusters_x.flatten()))

    kmeans_y = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans_y.train(y_for_clustering)
    distances_y, clusters_y = kmeans_y.index.search(y_for_clustering, 1)

    print(CountFrequency(clusters_y.flatten()))

    indices_x = np.arange(n)
    indices_y = np.arange(n)


    x_y_clusters_match = np.zeros((ncentroids, ncentroids)).astype('int32')
    clusters_x = clusters_x.flatten()
    clusters_x_emb = []
    for i in range(ncentroids): clusters_x_emb.append([])
    for i in range(len(indices_x)):
        y_idx = indices_x[i]
        clusters_x_emb[clusters_x[i]].append(x[i])
        if (indices_y[y_idx]) == i:
            x_y_clusters_match[clusters_x[i]][clusters_y[indices_x[i]]] += 1

    print(np.array(clusters_x_emb[0]).shape)

    y_x_clusters_match = np.zeros((ncentroids, ncentroids)).astype('int32')
    clusters_y = clusters_y.flatten()
    clusters_y_emb = []
    distances_y_training, clusters_y_training = kmeans_y.index.search(y_for_training, 1)

    for i in range(ncentroids): clusters_y_emb.append([])

    for i, v in enumerate(clusters_y_emb): clusters_y_emb[i] = []

    for i in range(len(indices_y)):
        y_idx = indices_y[i]
        if (indices_x[y_idx]) == i:
            y_x_clusters_match[clusters_y[i]][clusters_x[indices_y[i]]] += 1

    clusters_y_training = clusters_y_training.flatten()
    print(CountFrequency(clusters_y_training.flatten()))

    for i in range(len(clusters_y_training)):
        clusters_y_emb[clusters_y_training[i]].append(y_for_training[i])


    x_cluster_maps = x_y_clusters_match.argmax(axis=1)
    y_cluster_maps = y_x_clusters_match.argmax(axis=1)

    print('******')
    print(x_cluster_maps)
    print(y_cluster_maps)
    # check that the clusters map is reciprocal
    print(len((np.where(x_cluster_maps[y_cluster_maps] == np.arange(ncentroids)))[0]))

    assert ((x_cluster_maps[y_cluster_maps] == np.arange(ncentroids)).all())

    TX_clusters = [None for i in range(ncentroids)]
    TY_clusters = [None for i in range(ncentroids)]
    print(TX_clusters)


    for i in range(ncentroids):

        cur_cluster_x = i
        cur_cluster_y = x_cluster_maps[i]


        src_W = np.array(clusters_x_emb[cur_cluster_x]).T
        tgt_W = np.array(clusters_y_emb[cur_cluster_y]).T


        cluster_size = min(src_W.shape[1], tgt_W.shape[1])
        src_W = src_W[:, :cluster_size]
        tgt_W = tgt_W[:, :cluster_size]

        print(src_W.shape)
        print(tgt_W.shape)

        icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])

        icp_train.icp.TX = TX
        icp_train.icp.TY = TY

        _, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, is_init=False)

        print("Training - Achieved: Rec %f BB %d" % (rec, bb))
        icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
        icp_ft.icp.TX = icp_train.icp.TX
        icp_ft.icp.TY = icp_train.icp.TY
        ind_x, ind_y, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)
        print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
        TX_clusters[cur_cluster_x] = icp_ft.icp.TX
        TY_clusters[cur_cluster_y] = icp_ft.icp.TY


    if not os.path.exists(params.cp_dir):
        os.mkdir(params.cp_dir)


    np.save("%s/%s_%s_T_clusters-medical" % (params.cp_dir, params.src_lang, params.tgt_lang), TX_clusters)
    np.save("%s/%s_%s_T_clusters-medical" % (params.cp_dir, params.tgt_lang, params.src_lang), TY_clusters)
    faiss.write_index(kmeans_x.index, "%s/%s_clusters_index" % (params.cp_dir, params.src_lang))
    faiss.write_index(kmeans_y.index, "%s/%s_clusters_index" % (params.cp_dir, params.tgt_lang))








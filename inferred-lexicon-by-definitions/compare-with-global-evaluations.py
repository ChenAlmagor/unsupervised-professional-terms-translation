import pandas as pd
import io
import numpy as np
import sys
import Levenshtein as levenshtein
from gensim.models.keyedvectors import KeyedVectors


def compare_icp_and_matches_distances(lang1, lang2):
    my_lang1_emb = []
    my_lang1_word2id = {}
    with io.open(
            ('data/intersection-with-phrases-%s-emb.vec' % lang1),
            'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            word, vect = line.lower().rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            my_lang1_emb.append(vect)
            my_lang1_word2id[word] = len(my_lang1_emb) - 1

    my_lang2_emb = []
    my_lang2_emb_id2word = {}
    with io.open(
            ('data/intersection-with-phrases-%s-emb.vec' % lang2),
            'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            my_lang2_emb_id2word[len(my_lang2_emb)] = word
            my_lang2_emb.append(vect)

    T = np.load('/Non-adversarialTranslation/output/%s_%s_T.npy' % (lang1, lang2))
    TranslatedX = np.array(my_lang1_emb).dot(np.transpose(T))

    ds_lang1_emb = []
    ds_lang1_word2id = {}
    with io.open(
            ('/data/5-words-test-dict-lower-%s-emb.vec' % lang1),
            'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            word, vect = line.lower().rstrip().split(' ', 1)
            word = word.replace(':', '').replace(';', '')
            vect = np.fromstring(vect, sep=' ')
            ds_lang1_emb.append(vect)

            if word in ds_lang1_word2id:
                ds_lang1_word2id[word] = ds_lang1_word2id[word] + [len(ds_lang1_emb) - 1]
            else:
                ds_lang1_word2id[word] = [len(ds_lang1_emb) - 1]

    ds_lang2_emb = []
    ds_lang2_emb_id2word = {}
    with io.open(
            ('/data/5-words-test-dict-lower-%s-emb.vec' % lang2),
            'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)

            vect = np.fromstring(vect, sep=' ')
            ds_lang2_emb_id2word[len(ds_lang2_emb)] = word
            ds_lang2_emb.append(vect)

    my_relevant_lang2_emb = []
    closest_emb_in_ds = []
    words_to_eval_translate_2id = {}

    df = pd.DataFrame(
        columns=['word', 'score_match', 'score_translation', 'diff', 'match_word', 'ref_word', 'levenshtein_rank'])
    for id, word in enumerate(my_lang1_word2id):
        if word in ds_lang1_word2id:

            ids_of_emb_word = ds_lang1_word2id[word]
            tgt_emb = np.take(ds_lang2_emb, ids_of_emb_word, axis=0)  # take the emb from the tgt embs

            cur_tgt_words = []
            for tgt_id in ids_of_emb_word:
                cur_tgt_words.append(ds_lang2_emb_id2word[tgt_id])

            tgt_translated_emb = TranslatedX[id]
            translated_dup = np.array([tgt_translated_emb] * len(tgt_emb))

            # take closest
            options_score = np.linalg.norm(translated_dup - tgt_emb, ord=2, axis=1, keepdims=True)
            min_option_idx = np.argmin(options_score, axis=0)[0]
            my_relevant_lang2_emb.append(tgt_translated_emb)
            closest_emb_in_ds.append(tgt_emb[min_option_idx])
            words_to_eval_translate_2id[word] = len(my_relevant_lang2_emb) - 1

            normal_emb1_translation = tgt_translated_emb / np.linalg.norm(tgt_translated_emb, ord=2, axis=0,
                                                                          keepdims=True)
            normal_emb2 = (tgt_emb[min_option_idx]) / np.linalg.norm((tgt_emb[min_option_idx]), ord=2, axis=0,
                                                                     keepdims=True)

            tgt_match_emb = my_lang2_emb[id]
            translated_dup = np.array([tgt_match_emb] * len(tgt_emb))

            # take closest
            options_score = np.linalg.norm(translated_dup - tgt_emb, ord=2, axis=1, keepdims=True)
            min_option_idx = np.argmin(options_score, axis=0)[0]
            my_relevant_lang2_emb.append(tgt_match_emb)
            closest_emb_in_ds.append(tgt_emb[min_option_idx])
            words_to_eval_translate_2id[word] = len(my_relevant_lang2_emb) - 1

            testset_word = cur_tgt_words[min_option_idx]

            normal_emb1_match = tgt_match_emb / np.linalg.norm(tgt_match_emb, ord=2, axis=0,
                                                               keepdims=True)
            normal_emb2 = (tgt_emb[min_option_idx]) / np.linalg.norm((tgt_emb[min_option_idx]), ord=2, axis=0,
                                                                     keepdims=True)

            score_match = normal_emb1_match.dot(normal_emb2.T)
            score_translation = normal_emb1_translation.dot(normal_emb2.T)

            levenshtein_rank = levenshtein.ratio(word, testset_word)
            df = df.append({'word': word, 'score_match': score_match, 'score_translation': score_translation,
                            'diff': score_match - score_translation, 'match_word': my_lang2_emb_id2word[id],
                            'ref_word': testset_word, 'levenshtein_rank': levenshtein_rank}, ignore_index=True)

    df.to_csv('scores_comparison-en-es-phrases.csv')



#score according to med embeddings

def get_translation_score(lang1, lang2):
    ds_lang2_word_translations = {}
    with io.open(
            (
            'data/new-test-dict-only-words-no-parsed.txt'),
            'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            word_lang1, word_lang2 = line.rstrip().lower().split()

            if (word_lang2 in ds_lang2_word_translations):
                ds_lang2_word_translations[word_lang2] = ds_lang2_word_translations[word_lang2] + [word_lang1]

            else:
                ds_lang2_word_translations[word_lang2] = [word_lang1]

    word_vect = KeyedVectors.load_word2vec_format('Embeddings/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    ranks = {}
    df_translations = pd.read_csv('without_pharses_intersection_1.csv')
    for index, row in df_translations.iterrows():
        en_term = row['term_en'].lower()
        es_term = row['term_es'].lower()

        if (es_term in ds_lang2_word_translations):
            min_rank = sys.maxsize
            for translation in ds_lang2_word_translations[es_term]:
                try:
                    rank = word_vect.rank(en_term, translation)
                    if rank < min_rank:
                        min_rank = rank
                except:
                    continue

            ranks[es_term] = min_rank

    print(ranks)

    sorted_ranks = sorted(ranks.items(), key=lambda kv: kv[1])
    print(sorted_ranks)
    counter = 0
    for i, w in enumerate(sorted_ranks):

        if (sorted_ranks[w] > 0.9):
            counter += 1
    print(counter)


from gensim.models.wrappers import FastText


#produce med embeddings
def get_embedding_med():
    word_vect = FastText.load_fasttext_format(
        'Embeddings/BioWordVec_PubMed_MIMICIII_d200.bin')
    with io.open(
            ('data/intersection-1-parsed-en.txt'),
            'r', encoding='utf-8') as f:
        dict = []
        for _, line in enumerate(f):
            word = line.rstrip()
            try:
                emb = word_vect.wv[word]
            except:
                emb = word_vect.mo

            emb = np.array(emb)
            emb = np.array2string(emb, max_line_width=sys.maxsize)
            emb = emb.replace('[', '').replace(']', '')
            dict = [word + ' ' + emb]

    file1 = open("intersection-1-parsed-en-emb-med.vec", "w")

    file1.writelines(dict)
    file1.close()


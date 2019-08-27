import pandas as pd
import io
import numpy as np
import sys


def remove_phrases(path, en, es):
    df_without_phrases = pd.read_csv(path)
    for index, row in df_without_phrases.iterrows():
        print(index)
        if (len(row[en].split(' ')) > 1 or (len(row[es].split(' ')) > 1)):
            df_without_phrases.drop(index, inplace=True)

    df_without_phrases.to_csv('without_pharses.csv')


def n_words_only(n, path, en, es):
    df_without_phrases = pd.read_csv(path)
    for index, row in df_without_phrases.iterrows():
        if (
                isinstance(row[en], str) == False
                or isinstance(row[es], str) == False
                or (len(row[en].split(' ')) > n
                or (len(row[es].split(' ')) > n))

        ):
            df_without_phrases.drop(index, inplace=True)
        else:

            val_es = row[es].replace(',', '').replace(':', '').replace(' ', '-')
            val_en = row[en].replace(',', '').replace(':', '').replace(' ', '-')
            print(val_en)
            df_without_phrases.set_value(index=index , col=en, value=val_en)
            df_without_phrases.set_value(index=index, col=es, value=val_es)

    df_without_phrases.to_csv('parsed-up-to-n.csv')





def parse_phrases(path, en, es):
    df_without_phrases = pd.read_csv(path)
    for index, row in df_without_phrases.iterrows():

        val_es = row[es]
        val_en = row[en]
        print(val_en)
        df_without_phrases.set_value(index=index , col=en, value=val_en.replace(' ', '-').replace(',', '').lower())
        df_without_phrases.set_value(index=index, col=es, value=val_es.replace(' ', '-').replace(',', '').lower())



    df_without_phrases.to_csv('phrases_parsed.csv')



def remove_dup():
    df_init_no_dup = pd.read_csv('intercection_5_en_es.csv')
    en_words = {}
    es_words = {}
    for index, row in df_init_no_dup.iterrows():

        val_es = row['term_es']
        val_en = row['term_en']
        if ((val_en  not in en_words) and (val_es not in es_words)):
            en_words[val_en] = index
            es_words[val_es] = index
        else:
            df_init_no_dup.drop(index, inplace=True)


    df_init_no_dup.to_csv('no_dup_intercection_5.csv')


def write_to_txt_file(dict, path):
    file1 = open(path, "w")
    file1.writelines(dict)
    file1.close()

def remove_phrases_in_test_dataset(path, en, es):
    dict = []
    words_en = {}
    df_without_phrases = pd.read_csv(path)
    for index, row in df_without_phrases.iterrows():

        try:
            if (isinstance(row[en] , str) and
                    isinstance(row[es] , str)and
                    '--' not in row[en] and
                    '--' not in row[es] and
                    '' not in row[en] and
                    '' not in row[es] and
                    len(row[en].split(' ')) == 1
                    and (len(row[es].split(' ')) == 1)):
                en_parsed = row[en].lower().replace(';', '').replace(':', '')
                es_parsed = row[es].lower().replace(';', '').replace(':', '')
                dict.append(en_parsed + ' ' + es_parsed + '\n')
            else:
                df_without_phrases.drop(index, inplace=True)


        except:
            print(row[en] + ' ' + row[es])
            pass

    write_to_txt_file(dict)
    df_without_phrases.to_csv('remove-phrases-parsed.csv')



def get_words_freq(sorted_emb_path , path, full_vocab=False, lang='es'):
    df_words = pd.read_csv(path)
    words_freq = {}

    for index, row in df_words.iterrows():
        words_freq[row[lang]] = sys.maxsize
    print(words_freq)


    # load pretrained embeddings
    with io.open(sorted_emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                continue
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in words_freq:
                    words_freq[word] = i



    sorted_words = sorted(words_freq.items(), key=lambda kv: kv[1])
    sorted_words = [x[0] for x in sorted_words]
    print(sorted_words)

    df_words_sorted = pd.DataFrame(columns=['en', 'es'])
    for w in sorted_words:
        row = df_words[df_words[lang] == w]
        assert len(row != 1)
        df_words_sorted = df_words_sorted.append(row)

    df_words_sorted.to_csv('sorted_by_freq_unsupervised_en.csv')


def parse_txt_file(path):
    dict = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            parsed_word = line.rstrip().lower()
            dict.append(parsed_word+'\n')

    write_to_txt_file(dict)


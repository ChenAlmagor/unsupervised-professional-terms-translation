import numpy as np
import pandas as pd


def write_to_txt_file(dict, file_name):
    file1 = open(file_name , "w")
    file1.writelines(dict)
    file1.close()

def create_lexicon_from_indices():
    indices_x = np.load('indices_x_sentences-identity-init.npy')
    indices_y = np.load('indices_y_sentences-identity-init.npy')

    df_all_en = pd.read_csv('test-en.csv')
    df_all_es = pd.read_csv('test-es.csv')
    match_en = []
    match_es = []


    for x_idx in range(len(indices_x)):
        y_idx = indices_x[x_idx]
        
        if (indices_y[y_idx]) == x_idx:

            match_en.append(df_all_en.iloc[x_idx]['Terms'] + '\n')
            match_es.append(df_all_es.iloc[y_idx]['Terms'] + '\n')

    print(len(match_es))
    write_to_txt_file(match_en, 'match_en.txt')
    write_to_txt_file(match_es, 'match_es.txt')


create_lexicon_from_indices()
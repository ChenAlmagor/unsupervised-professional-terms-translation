import pandas as pd
import os
import csv
import json
import numpy as np


if __name__ == '__main__':


    directory = './data/fr'

    terms = []
    definitions = []
    for filename in os.listdir(directory):
        with open(directory + '/' + filename) as json_file:
            data = json.load(json_file)
            terms.extend(list(data.keys()))
            definitions.extend(list(data.values()))


    table = np.transpose([terms, definitions])
    df = pd.pandas.DataFrame(table, columns=['Terms', 'Definitions'])
    df.to_csv('test-fr.csv')



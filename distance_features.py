"""
@author: Bhavika Tekwani
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter

load_dotenv(find_dotenv())

genre_count = int(os.getenv('genre_count'))
img_count = int(os.getenv('sample_img_count'))

train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv'.format(genre_count * img_count)),
                       sep=';')


def gist_distance(x, y):
    d = (x - y) * (x - y)
    return np.sum(d)


def GIST_distance_matrix(path):
    gist_desc = np.load(path)
    gist_desc = np.concatenate(gist_desc.tolist(), axis=0)
    d = pairwise_distances(gist_desc, metric=gist_distance)
    return d


def get_class(df, train_case):
    return df.iloc[train_case].Class


def kNN(distance_matrix, df, k):
    '''
    kNN with GIST descriptors. 
    :param distance_matrix: np.array
    :param df: DataFrame
    :param k: int, no of clusters
    :return: pandas DataFrame with neighbours 
    '''
    neighbours = pd.DataFrame(distance_matrix.apply(lambda s: s.nsmallest(k+1).index.tolist()[1:], axis=1))
    neighbours = neighbours.rename(columns={0: 'nbr_list'})
    nbr_class = []
    votes = []

    for i in range(neighbours.shape[0]):  # row
        nbr_classes = []
        for j in range(len(neighbours['nbr_list'].iloc[i])):  # element
            a = neighbours['nbr_list'].iloc[i][j]
            nbr_classes.append(get_class(df, int(a)))
        nbr_class.append(nbr_classes)

    for l in nbr_class:
        b = Counter(l)
        vote = b.most_common(1)[0][0]
        votes.append(vote)

    neighbours['Prediction'] = pd.Series(votes)

    return neighbours


def main():
    # train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_sample.csv'), sep=';')
    # test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'test_sample.csv'), sep=';')
    path = 'temp/GISTDesc.npy'
    D = GIST_distance_matrix(path)
    D = pd.DataFrame(D)
    kNN(D, train_df, k=5)

if __name__ == '__main__':
    main()
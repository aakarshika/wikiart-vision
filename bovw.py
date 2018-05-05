from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np

load_dotenv(find_dotenv())


class BOVW():
    def __init__(self):
        self.feature_stack = None

    def flatten_features(self, features):
        """
        Takes a DataFrame series consisting of a list of lists and flattens it into a numpy array
        :param features: 
        :return: 1D numpy array
        """
        return features.reshape(-1, 2)

    def cluster(self, features, n_clusters, n_init, max_iter=300, random_state=7):
        """
        Cluster SIFT features using KMeans clustering 
        :param features: 
        :param n_clusters: int
        :param n_init: int
        :param max_iter: int
        :param random_state: int 
        :return: 
        """

        km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
        clusters = km.fit_predict(features)
        return clusters


def main():
    start_time = time()

    # genre_count = int(os.getenv('genre_count'))
    # img_count = int(os.getenv('sample_img_count'))

    features = np.load('temp/features_100.npy')
    features_df = pd.DataFrame(features, columns=['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation',
                                                  'ColorHist', 'GISTDesc', 'LocalMaxima',
                                                  'LocalMinima', 'Mean_HSVYBGR'])

    features_df['SIFT'] = features_df['SIFTDesc'].apply(lambda x: BOVW().flatten_features(x))

    # print(features_df['SIFTDesc'][0].shape, features_df['SIFTDesc'][1].shape)
    # print(features_df['SIFT'][0].shape, features_df['SIFT'][1].shape)

    sift = features_df['SIFTDesc'].as_matrix()

    kmeans_clusters = BOVW().cluster(sift, n_clusters=20, n_init=10, max_iter=300)
    #
    # print(kmeans_clusters)



if __name__ == '__main__':
    main()
import os
import cv2
import leargist
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import pairwise_distances

load_dotenv(find_dotenv())


def GIST_distance(X, Y):
    d = (X - Y) * (X - Y)
    return np.sum(d)


class DistanceFeatureExtractor():
    def GIST_distance_matrix(self):
        GISTDesc = np.load('temp/GISTDesc.npy')
        GISTDesc = np.concatenate(GISTDesc.tolist(), axis=0)

        D = pairwise_distances(GISTDesc, metric = GIST_distance )
        return D





def main():
    # train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_sample.csv'), sep=';')

    GIST_distance_matrix = DistanceFeatureExtractor().GIST_distance_matrix()

    print(GIST_distance_matrix)


if __name__ == '__main__':
    main()
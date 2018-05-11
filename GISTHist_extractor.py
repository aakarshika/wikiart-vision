import os
import cv2
import leargist
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from PIL import Image
from time import time
from tqdm import tqdm
from scipy.cluster.vq import *
from scipy.misc import imresize
from scipy.signal import argrelextrema

load_dotenv(find_dotenv())

HISTOGRAM_BINS = 10



class Features():
    def __init__(self, GISTfeatures, test=False):
        # self.df = df
        self.GISTfeatures = GISTfeatures
        self.test = test
        self.gist_descriptor_pool = None
        self.image_data = []
        self.vocab = None
        self.features = None
        self.KMEANS_CLUSTERS_FOR_SIFT = 25
        self.KMEANS_CLUSTERS_FOR_GIST = 25


    def createHistogram(self, descriptor_list, voc, k):
        features = np.zeros(k, "float32")
        words, distance = vq(descriptor_list, voc)
        for w in words:
            features[w] += 1
        return features

    def clusterDescriptors(self, descriptor_pool, k):
        voc, variance = kmeans2(descriptor_pool, k)
        self.vocab = voc

    def createFeatures(self, vocab=None, test=False):

        for GISTDesc in tqdm(self.GISTfeatures):
            
            img_features = {
                            'GISTDesc': GISTDesc
                            }

            if not test:

                if self.gist_descriptor_pool is None:
                    self.gist_descriptor_pool = img_features['GISTDesc']
                else:
                    self.gist_descriptor_pool = np.vstack((self.gist_descriptor_pool, img_features['GISTDesc']))

                if self.vocab is None:
                    print("Started kMeans clustering")
                    self.clusterDescriptors(self.gist_descriptor_pool, self.KMEANS_CLUSTERS_FOR_GIST)
                elif vocab is not None:
                    self.vocab = vocab

            self.image_data.append(img_features)

        for im in tqdm(self.image_data):
            if not test:
                vocab = self.vocab
            hist = self.createHistogram(im['GISTDesc'], vocab, self.KMEANS_CLUSTERS_FOR_GIST)
            im['features'] = hist

            if self.features is None:
                self.features = im['features']
            else:
                self.features = np.vstack((self.features,im['features']))

        print(self.features.shape)
        vocab = self.vocab
        return vocab

def generate_files(count):
    """
    Creates all possible numpy arrays - featuresets for train and test - 100 and 1000 cases each. 
    GIST descriptors for train and test - 100 and 1000 cases each. 
    :return: 
    """

    start = time()

    # train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv'.format(count)),
    #                        sep=';')
    # test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'test_{}.csv'.format(count)),
    #                       sep=';')


    features_GIST = np.load('data/GISTDesc_train_{}.npy'.format(count))
    features_GIST = np.concatenate(features_GIST.tolist(), axis=0)
    
    features_test_GIST = np.load('data/GISTDesc_test_{}.npy'.format(count))
    features_test_GIST = np.concatenate(features_test_GIST.tolist(), axis=0)

    f_train = Features(GISTfeatures = features_GIST)
    train_vocab = f_train.createFeatures()
    # np.save('data/features_train_{}.npy'.format(count), f_train.features)
    # print("Saved train features")

    # np.save('data/vocab_train_{}.npy'.format(count), train_vocab)
    # print("Saved train vocab")

    f_test = Features(GISTfeatures = features_test_GIST)
    f_test.createFeatures(vocab=train_vocab, test=True)
    # np.save('data/features_test_{}.npy'.format(count), f_test.features)

    # print("Saved test features")

    # train_df['GISTDesc'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).GIST())
    # test_df['GISTDesc'] = test_df['Path'].apply(lambda x: FeatureExtractor(x).GIST())

    # x = train_df.as_matrix(columns=['GISTDesc'])
    # np.save('data/GISTDesc_train_{}.npy'.format(count), x)

    # print("Saved GIST for train")

    # y = test_df.as_matrix(columns=['GISTDesc'])
    # np.save('data/GISTDesc_test_{}.npy'.format(count), y)

    # print("Saved GIST for test")

    np.save('data/GISTHist_train_{}.npy'.format(count), f_train.features)

    print("Saved GISTHist for train")

    np.save('data/GISTHist_test_{}.npy'.format(count), f_test.features)

    print("Saved GISTHist for test")

    print("Elapsed time: ", time()-start)

if __name__ == '__main__':
    columns = ['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation', 'ColorHist',
             'GISTDesc', 'LocalMaxima', 'LocalMinima', 'Mean_HSVYBGR']

    generate_files(1000)

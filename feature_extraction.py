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


class FeatureExtractor():

    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(self.path)
        self.img_small = imresize(self.img, (640, 640))
        self.yuv_img = cv2.cvtColor(self.img_small, cv2.COLOR_BGR2YUV)
        self.bw_img = cv2.cvtColor(self.img_small, cv2.COLOR_BGR2GRAY)
        self.hsv_img = cv2.cvtColor(self.img_small, cv2.COLOR_BGR2HSV)
        self.bgr_hist = None
        self.hsv_hist = None
        self.yuv_hist = None

    def histograms(self):
        bhist = cv2.calcHist([self.img_small], [0], None, [HISTOGRAM_BINS], [0, 256])
        ghist = cv2.calcHist([self.img_small], [1], None, [HISTOGRAM_BINS], [0, 256])
        rhist = cv2.calcHist([self.img_small], [2], None, [HISTOGRAM_BINS], [0, 256])
        self.bgr_hist = [bhist, ghist, rhist]

        hhist = cv2.calcHist([self.hsv_img], [0], None, [HISTOGRAM_BINS], [0, 256])
        shist = cv2.calcHist([self.hsv_img], [1], None, [HISTOGRAM_BINS], [0, 256])
        vhist = cv2.calcHist([self.hsv_img], [2], None, [HISTOGRAM_BINS], [0, 256])
        self.hsv_hist = [hhist, shist, vhist]

        yhist = cv2.calcHist([self.yuv_img], [0], None, [HISTOGRAM_BINS], [0, 256])
        uhist = cv2.calcHist([self.yuv_img], [1], None, [HISTOGRAM_BINS], [0, 256])
        vhist = cv2.calcHist([self.yuv_img], [2], None, [HISTOGRAM_BINS], [0, 256])
        self.yuv_hist = [yhist, uhist, vhist]
        # print(self.bgr_hist)

    def color_histogram(self):
        bhist = cv2.calcHist([self.img_small], [0], None, [HISTOGRAM_BINS], [0, 256])
        ghist = cv2.calcHist([self.img_small], [1], None, [HISTOGRAM_BINS], [0, 256])
        rhist = cv2.calcHist([self.img_small], [2], None, [HISTOGRAM_BINS], [0, 256])
        bhist = np.transpose(bhist)[0]
        ghist = np.transpose(ghist)[0]
        rhist = np.transpose(rhist)[0]
        colorHist = np.append(bhist, [ghist, rhist])
        return colorHist

    def brightness(self):
        hist = cv2.calcHist([self.yuv_img], [0], None, [HISTOGRAM_BINS], [0, 256])
        # self.histograms()
        # hist = self.yuv_hist[0]
        hist = np.transpose(hist)[0]
        return hist

    def saturation(self):
        hist = cv2.calcHist([self.hsv_img], [1], None, [HISTOGRAM_BINS], [0, 256])
        # self.histograms()
        # hist = self.yuv_hist[1]
        hist = np.transpose(hist)[0]
        return hist

    def GIST(self):
        x = Image.open(self.path)
        descriptors = leargist.color_gist(x)
        return np.array(descriptors)
    
    def SIFT(self):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.bw_img, None)
        if descriptors is None:
            descriptors = np.zeros((0, 128))
        return descriptors


    def mean_HSVYBGR(self):
        self.histograms()
        mh = np.mean(cv2.normalize(self.hsv_hist[0], None))
        ms = np.mean(cv2.normalize(self.hsv_hist[1], None))
        mv = np.mean(cv2.normalize(self.hsv_hist[2], None))
        my = np.mean(cv2.normalize(self.yuv_hist[0], None))
        mb = np.mean(cv2.normalize(self.bgr_hist[0], None))
        mg = np.mean(cv2.normalize(self.bgr_hist[1], None))
        mr = np.mean(cv2.normalize(self.bgr_hist[2], None))
        return [mh, ms, mv, my, mb, mg, mr]

    def edge_count(self):
        blurred= cv2.GaussianBlur(self.bw_img,(3,3),0)

        sigma=0.33
        v = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        canny = cv2.Canny(blurred,lower,upper)

        return np.sum(canny/255)



class Features():
    def __init__(self, df, test=False):
        self.df = df
        self.test = test
        self.sift_descriptor_pool = None
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

        for i in tqdm(range(self.df.shape[0])):
            path = self.df['Path'].iloc[i]

            fe = FeatureExtractor(path=path)

            img_features = {'Path': path,
                            'ColorHist': fe.color_histogram(),
                            'Brightness': fe.brightness(),
                            'Saturation': fe.saturation(),
                            'SIFTDesc': fe.SIFT(),
                            'GISTDesc': fe.GIST(),
                            'Mean_HSVYBGR': fe.mean_HSVYBGR(),
                            'EdgeCount': fe.edge_count()
                            }

            if not test:
                if self.sift_descriptor_pool is None:
                    self.sift_descriptor_pool = img_features['SIFTDesc']
                else:
                    self.sift_descriptor_pool = np.vstack((self.sift_descriptor_pool, img_features['SIFTDesc']))

                if self.gist_descriptor_pool is None:
                    self.gist_descriptor_pool = img_features['GISTDesc']
                else:
                    self.gist_descriptor_pool = np.vstack((self.gist_descriptor_pool, img_features['GISTDesc']))

                if self.vocab is None:
                    print("Started kMeans clustering for SIFT")
                    self.clusterDescriptors(self.sift_descriptor_pool, self.KMEANS_CLUSTERS_FOR_SIFT)
                    print("Started kMeans clustering for GIST")
                    self.clusterDescriptors(self.gist_descriptor_pool, self.KMEANS_CLUSTERS_FOR_GIST)
                elif vocab is not None:
                    self.vocab = vocab

            self.image_data.append(img_features)

        for im in tqdm(self.image_data):
            if not test:
                vocab = self.vocab
            SIFThist = self.createHistogram(im['SIFTDesc'], vocab, self.KMEANS_CLUSTERS_FOR_SIFT)
            im['SIFTHist'] = SIFThist
            GISThist = self.createHistogram(im['GISTDesc'], vocab, self.KMEANS_CLUSTERS_FOR_SIFT)
            im['GISTHist'] = GISThist

            im['features'] =  im['ColorHist']
            im['features'] = np.append(im['features'], im['Brightness'])
            im['features'] = np.append(im['features'], im['Saturation'])
            im['features'] = np.append(im['features'], im['SIFTHist'])
            im['features'] = np.append(im['features'], im['GISThist'])
            im['features'] = np.append(im['features'], im['Mean_HSVYBGR'])
            im['features'] = np.append(im['features'], im['EdgeCount'])

            if self.features is None:
                self.features = im['features']
            else:
                self.features = np.vstack((self.features, im['features']))

        vocab = self.vocab
        return vocab


def generate_files(count):
    """
    Creates all possible numpy arrays - featuresets for train and test - 100 and 1000 cases each. 
    GIST descriptors for train and test - 100 and 1000 cases each. 
    :return: 
    """

    start = time()

    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv'.format(count)),
                           sep=';')
    test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'test_{}.csv'.format(count)),
                          sep=';')

    f_train = Features(df=train_df)
    train_vocab = f_train.createFeatures()
    np.save('data/features_train_{}.npy'.format(count), f_train.features)
    print("Saved train features")
    np.save('data/vocab_train_{}.npy'.format(count), train_vocab)
    print("Saved train vocab")

    f_test = Features(df=test_df)
    f_test.createFeatures(vocab=train_vocab, test=True)
    np.save('data/features_test_{}.npy'.format(count), f_test.features)
    print("Saved test features")


    X_GIST = f_train.as_matrix(columns=['GISTDesc'])
    np.save('data/GIST_descriptors_train_{}.npy'.format(count), x)
    print("Saved GIST for train")

    y = f_test.as_matrix(columns=['GISTDesc'])
    np.save('data/GIST_descriptors_test_{}.npy'.format(count), y)
    print("Saved GIST for test")

    print("Elapsed time: ", time()-start)



if __name__ == '__main__':
    columns = ['Painting', 'Class', 'Path', 'ColorHist','SIFTDesc', 'Brightness', 'Saturation', 
             'GISTDesc', 'Mean_HSVYBGR']

    genre_count = int(os.getenv('genre_count'))
    img_count = int(os.getenv('sample_img_count'))
    # img_count = int(os.getenv('img_count'))

    generate_files(genre_count*img_count)

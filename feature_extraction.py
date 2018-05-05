import os
import cv2
import leargist
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import matplotlib.pyplot as plt
from time import time
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


    def SIFT(self):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.bw_img, None)
        return descriptors

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

    def brightness(self):
        hist = cv2.calcHist([self.yuv_img], [0], None, [HISTOGRAM_BINS], [0, 256])
        # self.histograms()
        # hist = self.yuv_hist[0]
        hist = np.transpose(hist)[0]
        return hist

    def saturation(self):
        hist = cv2.calcHist([self.yuv_img], [1], None, [HISTOGRAM_BINS], [0, 256])
        # self.histograms()
        # hist = self.yuv_hist[1]
        hist = np.transpose(hist)[0]
        return hist

    def color_histogram(self):
        bhist = cv2.calcHist([self.img_small], [0], None, [HISTOGRAM_BINS], [0, 256])
        ghist = cv2.calcHist([self.img_small], [1], None, [HISTOGRAM_BINS], [0, 256])
        rhist = cv2.calcHist([self.img_small], [2], None, [HISTOGRAM_BINS], [0, 256])
        # self.histograms()
        # bhist = self.bgr_hist[0]
        # ghist = self.bgr_hist[1]
        # rhist = self.bgr_hist[2]

        bhist = np.transpose(bhist)[0]
        ghist = np.transpose(ghist)[0]
        rhist = np.transpose(rhist)[0]
        colorHist = np.append(bhist, [ghist, rhist])
        return colorHist

    def pos_local_maxima_HSVYBGR(self):
        self.histograms()
        h_lm = argrelextrema(self.hsv_hist[0], np.greater)
        s_lm = argrelextrema(self.hsv_hist[1], np.greater)
        v_lm = argrelextrema(self.hsv_hist[2], np.greater)
        y_lm = argrelextrema(self.yuv_hist[0], np.greater)
        b_lm = argrelextrema(self.bgr_hist[0], np.greater)
        g_lm = argrelextrema(self.bgr_hist[1], np.greater)
        r_lm = argrelextrema(self.bgr_hist[2], np.greater)
        return [h_lm, s_lm, v_lm, y_lm, b_lm, g_lm, r_lm]
    
    def pos_local_minima_HSVYBGR(self):
        self.histograms()
        h_lm = argrelextrema(self.hsv_hist[0], np.less)
        s_lm = argrelextrema(self.hsv_hist[1], np.less)
        v_lm = argrelextrema(self.hsv_hist[2], np.less)
        y_lm = argrelextrema(self.yuv_hist[0], np.less)
        b_lm = argrelextrema(self.bgr_hist[0], np.less)
        g_lm = argrelextrema(self.bgr_hist[1], np.less)
        r_lm = argrelextrema(self.bgr_hist[2], np.less)
        return [h_lm, s_lm, v_lm, y_lm, b_lm, g_lm, r_lm]
    
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

    def GIST(self):
        x = Image.open(self.path)
        descriptors = leargist.color_gist(x)
        return descriptors


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

    def createHistogram(self, descriptor_list, voc, k):
        features = np.zeros(k, "float32")
        words, distance = vq(descriptor_list, voc)
        for w in words:
            features[w] += 1
        return features

    def clusterDescriptors(self, descriptor_pool, k):
        vocab, variance = kmeans2(descriptor_pool, k)
        self.vocab = vocab

    def createFeatures(self, vocab=None, test=True):

        for i in range(self.df.shape[0] - 95):
            path = self.df['Path'].iloc[i]

            fe = FeatureExtractor(path=path)

            img_features = {'Path': path,
                            'SIFTDesc': fe.SIFT(),
                            'Brightness': fe.brightness(),
                            'Saturation': fe.saturation(),
                            'ColorHist': fe.color_histogram(),
                            'Mean_HSVYBGR': fe.mean_HSVYBGR(),
                            'GISTDesc': fe.GIST()}

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
                    print("Started kMeans clustering")
                    self.clusterDescriptors(self.sift_descriptor_pool, self.KMEANS_CLUSTERS_FOR_SIFT)
                else:
                    self.vocab = vocab

            self.image_data.append(img_features)

            for img in self.image_data:
                hist = self.createHistogram(img['SIFTDesc'], self.vocab, self.KMEANS_CLUSTERS_FOR_SIFT)
                img['SIFTHist'] = hist
                img['features'] = hist

                img['features'] = np.append(img['features'], img['Brightness'])
                img['features'] = np.append(img['features'], img['ColorHist'])
                img['features'] = np.append(img['features'], img['Mean_HSVYBGR'])
                img['features'] = np.append(img['features'], img['Saturation'])

                if self.features is None:
                    self.features = img['features']
                else:
                    self.features = np.vstack((self.features, img['features']))

        return self.image_data, self.features, self.vocab


def main():

    start_time = time()

    genre_count = int(os.getenv('genre_count'))
    img_count = int(os.getenv('sample_img_count'))

    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv'.format(genre_count*img_count)), sep=';')

    f = Features(df=train_df)
    _, train_features, vocab = f.createFeatures(train_df, test=False)

    # train_df['SIFTDesc'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).SIFT())
    # train_df['Brightness'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).brightness())
    # train_df['Saturation'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).saturation())
    # train_df['ColorHist'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).color_histogram())
    # train_df['GISTDesc'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).GIST())
    # train_df['LocalMaxima'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).pos_local_maxima_HSVYBGR())
    # train_df['LocalMinima'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).pos_local_minima_HSVYBGR())
    # train_df['Mean_HSVYBGR'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).mean_HSVYBGR())
    #
    # # just for understanding the structure of features - to be removed later
    # train_df['SIFTShape'] = train_df['SIFTDesc'].apply(lambda x: x.shape)
    # train_df['ColorHistShape'] = train_df['ColorHist'].apply(lambda x: x.shape)
    # train_df['BrightnessShape'] = train_df['Brightness'].apply(lambda x: x.shape)
    # train_df['SaturationShape'] = train_df['Saturation'].apply(lambda x: x.shape)
    # train_df['GISTShape'] = train_df['GISTDesc'].apply(lambda x:x.shape)
    # train_df['Mean_HSVYBGRShape'] = train_df['Mean_HSVYBGR'].apply(lambda x: len(x))

    # print(train_df.head(10))
    #
    # tr = train_df.as_matrix()
    # np.save('temp/features_{}.npy'.format(genre_count*img_count), tr)
    #
    # # if df is saved to file, then it can be read in distance_features.py, else:
    # x = train_df.as_matrix(columns=['GISTDesc'])
    # np.save('temp/GISTDesc_{}.npy'.format(img_count*genre_count), x)

    print("Finished creating features in:", time()-start_time)


if __name__ == '__main__':
    main()

    # features = np.load('temp/features.npy')
    # features_df = pd.DataFrame(features, columns=['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation',
    #                                               'ColorHist', 'GISTDesc', 'LocalMaxima',
    #                                               'LocalMinima', 'Mean_HSVYBGR'])
    #
    # features_df['SIFTShape'] = features_df['SIFTDesc'].apply(lambda x: x.shape)
    #
    # print(features_df.head(5))
import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os

load_dotenv(find_dotenv())

HISTOGRAM_BINS = 10


class FeatureExtractor():

    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(self.path)
        self.yuv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        self.bw_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def SIFT(self):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.bw_img, None)
        return descriptors
        # decide what to do with these descriptors

    def brightness(self):
        hist = cv2.calcHist([self.yuv_img], [0], None, [HISTOGRAM_BINS], [0, 256])
        hist = np.transpose(hist)[0]
        return hist

    def saturation(self):
        hist = cv2.calcHist([self.yuv_img], [1], None, [HISTOGRAM_BINS], [0, 256])
        hist = np.transpose(hist)[0]
        return hist

    def color_histogram(self):
        bhist = cv2.calcHist([self.img], [0], None, [HISTOGRAM_BINS], [0, 256])
        ghist = cv2.calcHist([self.img], [1], None, [HISTOGRAM_BINS], [0, 256])
        rhist = cv2.calcHist([self.img], [2], None, [HISTOGRAM_BINS], [0, 256])
        bhist = np.transpose(bhist)[0]
        ghist = np.transpose(ghist)[0]
        rhist = np.transpose(rhist)[0]
        colorHist = np.append(bhist, [ghist, rhist])
        return colorHist


def main():
    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_sample.csv'), sep=';')

    train_df['Descriptors'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).SIFT())
    train_df['Brightness'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).brightness())
    train_df['Saturation'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).saturation())
    train_df['ColorHist'] = train_df['Path'].apply(lambda x: FeatureExtractor(x).color_histogram())

    # just for understanding the structure of features - to be removed later
    train_df['DescriptorShape'] = train_df['Descriptors'].apply(lambda x: x.shape)
    train_df['ColorHistShape'] = train_df['ColorHist'].apply(lambda x: x.shape)
    train_df['BrightnessShape'] = train_df['Brightness'].apply(lambda x: x.shape)
    train_df['SaturationShape'] = train_df['Saturation'].apply(lambda x: x.shape)

    print(train_df.head(25))

if __name__ == '__main__':
    main()
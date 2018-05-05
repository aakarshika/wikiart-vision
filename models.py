from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

genre_count = int(os.getenv('genre_count'))
img_count = int(os.getenv('sample_img_count'))

features = np.load('temp/features_{}.npy'.format(genre_count*img_count))

# features_df = pd.DataFrame(features, columns=['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation',
#                                                    'ColorHist', 'GISTDesc', 'LocalMaxima',
#                                                    'LocalMinima', 'Mean_HSVYBGR'])

train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv'.format(genre_count*img_count)),
                       sep=';')

feature_cols = ['SIFTDesc', 'Brightness', 'Saturation', 'ColorHist', 'GISTDesc', 'Mean_HSVYBGR']

rfc = RandomForestClassifier()
rfc.fit(X=features, y=train_df['Class'])

"""
@author: Bhavika Tekwani
"""

import os
from shutil import copy, rmtree
import zipfile

dest = 'CS682_Project_btekwani_apriydar'

if not os.path.isdir(dest):
    os.mkdir(dest)
    os.chdir(dest)
    os.mkdir('viz')
    os.mkdir('Notes')
    os.mkdir('data')
    os.mkdir('src')
else:
    rmtree(dest)
    os.mkdir(dest)

os.chdir('../')
copy('viz/confusion_matrix.py', dest+'/viz/')
copy('cnn.py', dest+'/src/')
copy('resnet.py', dest+'/src/')
copy('resnet_eval.py', dest+'/src/')
copy('models.py', dest+'/src/')
copy('create_dataset.py', dest+'/src/')
copy('distance_features.py', dest+'/src/')
copy('feature_extraction.py', dest+'/src/')
copy('wikiart.py', dest+'/src/')

copy('Notes/leargist_setup.md', dest+'/Notes/')

copy('README.md', dest)
copy('requirements.txt', dest)
copy('sample.env', dest)
copy('aws.env', dest)

copy('aws/genre_class.txt', dest+'/data/')
copy('aws/genre_train.csv', dest+'/data/')
copy('aws/genre_val.csv', dest+'/data/')
copy('aws/test_1000.csv', dest+'/data/')
copy('aws/train_1000.csv', dest+'/data/')


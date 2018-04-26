from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd

load_dotenv(find_dotenv())

classes = os.getenv('genre_class')
class_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), classes), sep=' ', header=None)

train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), os.getenv('train')), sep=',', header=None, names=['Painting', 'Class'])
test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), os.getenv('test')), sep=',', header=None, names=['Painting', 'Class'])

print(train_df.shape)
print(test_df.shape)

# How many paintings in train by class
for cls, group in train_df.groupby('Class'):
    print(cls, group.size)

print("Count by class in test")
for cls, group in test_df.groupby('Class'):
    print(cls, group.size)


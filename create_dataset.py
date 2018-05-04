from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from scipy.misc import imread, imsave, imresize

load_dotenv(find_dotenv())

classes = os.getenv('genre_class')
class_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), classes), sep=' ', header=None)

train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), os.getenv('train')), sep=',', header=None, names=['Painting', 'Class'])
test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), os.getenv('test')), sep=',', header=None, names=['Painting', 'Class'])

print("Train dataset shape", train_df.shape)
print("Test dataset shape", test_df.shape)

# How many paintings in train by class
for cls, group in train_df.groupby('Class'):
    print(cls, group.size)

print("Count by class in test")
for cls, group in test_df.groupby('Class'):
    print(cls, group.size)


# Create sample train and test sets

def create_dataset(df, genre_count, image_count, output_fname):
    """
    
    Creates a CSV file with the image location, artist and class label (encoded for artists). 
    :param filepath: 
    """

    counts = df['Class'].value_counts()
    filter = list(counts[counts > image_count].index)

    # filter for genres that appear image_count times or more
    more_than_img_count = (df[df['Class'].isin(filter)])
    topn = more_than_img_count.groupby('Class').head(image_count)
    topn.sort_values(by='Class', inplace=True, ascending=True)
    topn = topn.head(genre_count*image_count)

    topn['Path'] = os.getenv('dataset_location') + "/"+topn['Painting']

    topn.to_csv(os.path.join(os.getenv('dataset_location'), output_fname), sep=';', index=False)


def explore(df):
    print("Height & width")
    for i in range(df.shape[0]):
        f = df['Path'].iloc[i]
        img = imread(f)
        # shows the shape (h, w, channels)
        print(img.dtype, img.shape)


def main():
    global train_df
    global test_df

    genre_count = 10
    img_count = 100

    create_dataset(train_df, genre_count, img_count, output_fname='train_{}.csv'.format(genre_count*img_count))
    create_dataset(test_df, genre_count, img_count, output_fname='test_{}.csv'.format(genre_count*img_count))
    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_sample.csv'), sep=';')
    explore(train_df)


if __name__ == '__main__':
    main()


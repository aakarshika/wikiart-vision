from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd

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


def main():
    create_dataset(train_df, 10, 10, output_fname='train_sample.csv')
    create_dataset(test_df, 10, 10, output_fname='test_sample.csv')

if __name__ == '__main__':
    main()
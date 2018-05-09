import torch.utils.data as data_utils
import torch
import pandas as pd
from PIL import Image
import numpy as np
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())


class WikiartDataset(data_utils.Dataset):
    def __init__(self, config):
        self.wikiart_path = config['wikiart_path']
        self.images = config['images_path']
        self.num_samples = config['size']
        self.ids_list = list(range(1, self.num_samples+1))
        self.arch = config.get('arch')
        self.train = config.get('train')
        # random.shuffle(self.ids_list)

    def __getitem__(self, index):
        dataset = pd.read_csv(self.wikiart_path, sep=',')
        row = dataset.iloc[index]
        if self.train is True:
            path = os.getenv('train_aws_dataset') + "/" + row['Path']
            image = Image.open(path)
        else:
            path = os.getenv('test_aws_dataset') + "/" + row['Path']
            image = Image.open(path)
        if self.arch == 'cnn':
            image = image.resize((32, 32))
        else:
            image = image.resize((224, 224))
        image = np.array(image).astype(np.float32)
        label = row['Class']

        sample = {'image': torch.from_numpy(image), 'class': label}
        return sample

    def __len__(self):
        return len(self.ids_list)
import numpy as np

x = np.load('data/features_train_100.npy')
print(x.shape)

y = np.load('data/features_test_100.npy')
print(y.shape)

vocab = np.load('data/vocab_train_100.npy')
print(vocab.shape)

gist_x = np.load('data/GISTDesc_train_100.npy')
print(gist_x.shape)

gist_y = np.load('data/GISTDesc_test_100.npy')
print(gist_y.shape)

x2 = np.load('data/features_train_1000.npy')
print(x2.shape)

y2 = np.load('data/features_test_1000.npy')
print(y2.shape)

vocab2 = np.load('data/vocab_train_1000.npy')
print(vocab2.shape)

gist_x2 = np.load('data/GISTDesc_train_1000.npy')
print(gist_x2.shape)

gist_y2 = np.load('data/GISTDesc_test_1000.npy')
print(gist_y2.shape)

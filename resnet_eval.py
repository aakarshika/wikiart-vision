import pickle
import sklearn.metrics
from sklearn.metrics import classification_report
import os
from dotenv import load_dotenv, find_dotenv
from pprint import pprint

load_dotenv(find_dotenv())
pickle_path = os.getenv('pickle_path')

lrs = ['1e-05', '0.0001', '0.001', '0.01', '0.1']
epochs = '20'

for lr in lrs:
    for f in os.listdir(pickle_path):
        if f.endswith('pkl'):
                if lr in f and epochs in f:
                    if 'cnn' in f:
                        y_pred_path = 'cnn_ypred_{}_{}.pkl'.format(lr, epochs)
                        y_act_path = 'cnn_y_actual_{}_{}.pkl'.format(lr, epochs)
                        pkl_act = pickle.load(open(pickle_path + y_act_path, 'rb'))
                        pkl_pred = pickle.load(open(pickle_path + y_pred_path, 'rb'))

                        print("CNN - 2 Layer, Learning rate: {}".format(lr))
                        print("Accuracy: ",sklearn.metrics.accuracy_score(pkl_pred, pkl_act) * 100)
                        print("Precision: ", sklearn.metrics.precision_score(pkl_pred, pkl_act, average='weighted') * 100)
                        print("Recall:", sklearn.metrics.recall_score(pkl_pred, pkl_act, average='weighted') * 100)

                        pprint(classification_report(pkl_act, pkl_pred, target_names=['abstract_painting',
                                                                                            'cityscape',
                                                                                            'genre_painting',
                                                                                            'illustration', 'landscape',
                                                                                            'nude_painting',
                                                                                            'portrait',
                                                                                            'religious_painting',
                                                                                            'sketch_and_study',
                                                                                            'still_life']))

                    elif 'rnet' in f:
                        y_pred_path = 'rnet18_re_ypred_{}_{}.pkl'.format(lr, epochs)
                        y_act_path = 'rnet18_re_y_actual_{}_{}.pkl'.format(lr, epochs)
                        pkl_act = pickle.load(open(pickle_path + y_act_path, 'rb'))
                        pkl_pred = pickle.load(open(pickle_path + y_pred_path, 'rb'))
                        print("ResNet (transfer learning) with learning rate {}".format(lr))
                        print("Accuracy:", sklearn.metrics.accuracy_score(pkl_pred, pkl_act) * 100)
                        print("Precision:", sklearn.metrics.precision_score(pkl_pred, pkl_act, average='weighted') * 100)
                        print("Recall:", sklearn.metrics.recall_score(pkl_pred, pkl_act, average='weighted') * 100)

                        pprint(classification_report(pkl_act, pkl_pred, target_names=['abstract_painting',
                                                                                      'cityscape',
                                                                                      'genre_painting',
                                                                                      'illustration', 'landscape',
                                                                                      'nude_painting',
                                                                                      'portrait',
                                                                                      'religious_painting',
                                                                                      'sketch_and_study',
                                                                                      'still_life']))

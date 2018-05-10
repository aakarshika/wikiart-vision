from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import itertools
from pprint import pprint

lrs = [1e-5, 1e-4, 1e-3, 1e-2]
epochs = [20]

combinations = list(itertools.product(lrs, epochs))

for c in combinations:
    actual = 'models/rnet18_re_y_actual_{}_{}.pkl'.format(c[0], c[1])
    pred = 'models/rnet18_re_ypred_{}_{}.pkl'.format(c[0], c[1])

    y = pickle.load(open(actual, 'rb'))
    y_pred = pickle.load(open(pred, 'rb'))

    y_act_list = []
    y_pred_list = []

    for i in y:
        y_act_list.append(i.item())
    for i in y_pred:
        y_pred_list.append(i.item())

    print("For learning rate {}".format(c[0]))
    print("Accuracy", accuracy_score(y_act_list, y_pred_list) * 100)

    pprint(classification_report(y_act_list, y_pred_list, target_names=['abstract_painting',
                                                                       'cityscape',
                                                                       'genre_painting',
                                                                       'illustration', 'landscape',
                                                                       'nude_painting',
                                                                       'portrait',
                                                                       'religious_painting',
                                                                       'sketch_and_study',
                                                                       'still_life']))
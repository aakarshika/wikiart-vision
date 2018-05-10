
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *
from sklearn import metrics
from sklearn.feature_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

load_dotenv(find_dotenv())

genre_count = int(os.getenv('genre_count'))
img_count = int(os.getenv('sample_img_count'))

cross_val_folds = 5


def gist_distance(x, y):
    d = (x - y) * (x - y)
    return np.sum(d)


class Classifier():

    # def __init__(self):
        # self.X = X
        # self.y = y
    
    def RFC(self):
        return RandomForestClassifier()
    def XGBoost(self):
        print("XGBoost")
    def KNN(self):
        return KNeighborsClassifier()

class FeatureSelector():
    def SFM(self,clf):
        return SelectFromModel(clf)
    def SKB(self):
        return SelectKBest(chi2)

class FeatureVectorSelector():
    def selectTopN(self,clf,n):
        print("Yo")

# def importance(rfc):
#     importances = rfc.feature_importances_
#     # importances = pipe.named_steps.randomTree.feature_importances_

#     print(importances)

#     # std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
#     #              axis=0)
#     indices = np.argsort(importances)[::-1]

#     # Print the feature ranking
#     print("Feature ranking:")

#     for f in range(X.shape[1]):
#         print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])  )

#     # Plot the feature importances of the forest
#     plt.figure()
#     plt.title("Feature importances")
#     plt.bar(range(X.shape[1]), importances[indices],
#            color="r"
#            # , yerr=std[indices]
#             ,align="center")
#     plt.xticks(range(X.shape[1]), indices)
#     plt.xlim([-1, X.shape[1]])
#     plt.show()


def model_1():
    """ 
    Feature Selection:
        SelectFromModel(RandomForest) 
    Classifiers: 
        KNN 
        RandomForestClassifier """

    clf = Classifier()
    randomforest = clf.RFC()
    knearest = clf.KNN()

    fs = FeatureSelector()
    sfm = fs.SFM(randomforest)


    pipe = Pipeline(steps=[
        ('selectFeatures', sfm),
        ('classify', randomforest)
        ])

    # just checking
    # K_BestSelect = [20,30,50]
    K_Nearest_Neighboors = [5,10]
    SELECT_FEATURES = [sfm]
    CLASSIFY = [randomforest , knearest]

    param_grid = [
        {
            'selectFeatures': SELECT_FEATURES,
            'classify': [knearest],
            'classify__n_neighbors': K_Nearest_Neighboors
        }
        ,
        {
            'selectFeatures': SELECT_FEATURES,
            'classify': [randomforest] 
        }
    ]

    grid = GridSearchCV(pipe, cv=cross_val_folds, n_jobs=1, param_grid=param_grid)
    return grid


def XGB_model2():
    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
    xgb = XGBClassifier(
        learning_rate=0.02, 
        n_estimators=10, 
        objective='multi:softmax',
        silent=True, 
        nthread=1)
    random_search = RandomizedSearchCV(xgb, 
        param_distributions=params,
        n_iter=5, 
        # scoring='roc_auc', 
        n_jobs=4, 
        cv=cross_val_folds, 
        verbose=-1, 
        random_state=1001 )
    return random_search


def display_results(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cmat = confusion_matrix(y_test, y_pred, labels=range(genre_count))
    print("Confusion Matrix:")
    print(cmat)
    print("Class-wise accuracy:")
    print(cmat.diagonal()/cmat.sum(axis=1))
    # print(grid.best_estimator_)

    print("\n\n")




def main():

    genre_count = int(os.getenv('genre_count'))
    img_count = int(os.getenv('img_count'))

    load_data_start = time()

    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv').format(genre_count*img_count), sep=';')
    features = np.load('data/features_train_{}.npy'.format(genre_count*img_count))
    
    features_GIST = np.load('data/GISTDesc_train_{}.npy'.format(genre_count*img_count))
    features_GIST = np.concatenate(features_GIST.tolist(), axis=0)
    
    test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'test_{}.csv'.format(genre_count*img_count)), sep=';')
    features_test = np.load('data/features_test_{}.npy'.format(genre_count*img_count))
    
    features_test_GIST = np.load('data/GISTDesc_test_{}.npy'.format(genre_count*img_count))
    features_test_GIST = np.concatenate(features_test_GIST.tolist(), axis=0)


    print(features.shape)
    print(features_GIST.shape)
    print(train_df.shape)

    print("Data files loaded in: ", time()-load_data_start)

    # features_df = pd.DataFrame(features, columns=['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation',
    #                                                    'ColorHist', 'GISTDesc', 'LocalMaxima',
    #                                                    'LocalMinima', 'Mean_HSVYBGR'])

    feature_cols = ['SIFTDesc', 'Brightness', 'ColorHist', 'Mean_HSVYBGR','Saturation', 'GISTDesc']
    features_lens = [25, 10, 30, 10, 8]

    X = features
    X_GIST = features_GIST
    y = train_df['Class']
    y_test = test_df['Class']


    # Models:

    start = time()
    print("Starting model training---")

    #### General ####
    grid = model_1()
    grid.fit(X, y)
    # mean_scores = np.array(grid.cv_results_['mean_test_score'])

    y_pred=grid.predict(features_test)
    print("General feature classification")
    display_results(y_test, y_pred)


    #### XGBoost ####
    grid = XGB_model2()
    # y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
    grid.fit(X, y)

    y_pred=grid.predict(features_test)
    print("XGBoost classification")
    display_results(y_test, y_pred)


    #### GIST KNN ####
    KNN_GIST = Classifier().KNN()
    KNN_GIST.fit(X_GIST , y)

    y_pred_gist = KNN_GIST.predict(features_test_GIST)
    print("GIST KNN classification")
    display_results(y_test, y_pred_gist)

    print("Finished fitting models in: ", time()-start)

main()
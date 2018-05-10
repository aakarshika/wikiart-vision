
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

from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

load_dotenv(find_dotenv())

genre_count = int(os.getenv('genre_count'))
img_count = int(os.getenv('img_count'))

cross_val_folds = 5

def gist_distance(x, y):
    d = (x - y) * (x - y)
    return np.sum(d)

class FeatureVectorSelector(object):
    def __call__(self, ftii=None):
        self.feature_to_ignore_index = ftii

    def transform(self, X):

        X1 = []
        X2 = []

        if self.feature_to_ignore_index == 0:
            return X[:,features_lens_com[self.feature_to_ignore_index+1]:82]
        
        X1 = X[:,0:features_lens_com[self.feature_to_ignore_index]]

        if self.feature_to_ignore_index < len(features_lens)-1:
            X2 = X[:,features_lens_com[self.feature_to_ignore_index+1]:82]
            return np.concatenate( (X1, X2), axis=1)

        raise ValueError('Feature vector index out of bounds.')

    def fit(self, X, y=None):
        return self


class Classifier():

    def RFC(self):
        return RandomForestClassifier()
    def KNN(self):
        return KNeighborsClassifier()
    def XGB(self):
        xgb = XGBClassifier(
            learning_rate=0.1, 
            n_estimators=50, 
            objective='multi:softmax',
            silent=True, 
            nthread=1)

        return xgb


class FeatureSelector():
    
    def SFM(self,clf):
        return SelectFromModel(clf)
    
    def SKB(self):
        return SelectKBest(chi2)

    def FVS(self):
        return FeatureVectorSelector()

feature_cols = ['SIFTDesc', 'Brightness', 'BHist','GHist','RHist', 'Mean_H', 'Mean_S','Mean_V','Mean_Y',' Mean_B','Mean_G','Mean_R','Saturation', 'GISTDesc']
features_lens = [25, 10, 10,10,10, 1,1,1,1,1,1,1, 10]
features_lens_com = [25, 35, 45,55,65, 66,67,68,69,70,71,72, 82]


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

def XGBoost():
    params={

        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    xgb = XGBClassifier(
        learning_rate=0.1, 
        n_estimators=50, 
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

def display_results(y_test, y_pred):
    # print(classification_report(y_test, y_pred))
    cmat = confusion_matrix(y_test, y_pred, labels=range(genre_count))
    print("Confusion Matrix:")
    print(cmat)
    print("Class-wise accuracy:")
    print(cmat.diagonal()/cmat.sum(axis=1))
    # print(grid.best_estimator_)

    print("\n\n")

def mega_combo_model(xgb_params):
    """ 
    Feature Selection:
        SelectFromModel(RandomForest) 
    Classifiers: 
        KNN 
        RandomForestClassifier
        XGBoost """

    clf = Classifier()
    randomforest = clf.RFC()
    knearest = clf.KNN()
    xgb = clf.XGB()

    fs = FeatureSelector()
    sfm = fs.SFM(randomforest)
    fvs = fs.FVS()


    pipe = Pipeline(steps=[
        ('selectFeatures', sfm),
        ('classify', randomforest)
        ])

    # just checking
    # K_BestSelect = [20,30,50]
    K_Nearest_Neighboors = [5]
    SELECT_FEATURES = [
                    fvs(0), 
                    fvs(1), 
                    fvs(2), 
                    fvs(3), 
                    fvs(4), 
                    fvs(5), 
                    fvs(6), 
                    fvs(7), 
                    fvs(8), 
                    fvs(9), 
                    fvs(10),
                    fvs(11),
                    fvs(12)]
    CLASSIFY = [randomforest , knearest]

    param_grid = [
        # {
        #     'selectFeatures': SELECT_FEATURES,
        #     'classify': [knearest],
        #     'classify__n_neighbors': K_Nearest_Neighboors
        # }
        # ,
        {
            'selectFeatures': SELECT_FEATURES,
            'classify': [randomforest] 
        }
        # ,
        # {
        #     'selectFeatures':SELECT_FEATURES,
        #     'classify': [xgb],
        #     'classify__min_child_weight': [xgb_params['0min_child_weight']],
        #     'classify__gamma': [xgb_params['gamma']],
        #     'classify__subsample': [xgb_params['subsample']],
        #     'classify__colsample_bytree': [xgb_params['colsample_bytree']],
        #     'classify__max_depth': [xgb_params['max_depth']]
        # }
    ]
    grid = GridSearchCV(pipe, cv=cross_val_folds, n_jobs=4, param_grid=param_grid)
    return grid


def main():

    genre_count = int(os.getenv('genre_count'))
    img_count = int(os.getenv('img_count'))
    # img_count = int(os.getenv('sample_img_count'))
    

    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_{}.csv').format(genre_count*img_count), sep=';')
    features = np.load('data/features_train_{}.npy'.format(genre_count*img_count))
    
    features_GIST = np.load('data/GISTDesc_train_{}.npy'.format(genre_count*img_count))
    features_GIST = np.concatenate(features_GIST.tolist(), axis=0)
    
    test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'test_{}.csv'.format(genre_count*img_count)), sep=';')
    features_test = np.load('data/features_test_{}.npy'.format(genre_count*img_count))
    
    features_test_GIST = np.load('data/GISTDesc_test_{}.npy'.format(genre_count*img_count))
    features_test_GIST = np.concatenate(features_test_GIST.tolist(), axis=0)

    
    # print(features.shape)
    # print(features_GIST.shape)
    # print(train_df.shape)

    # features_df = pd.DataFrame(features, columns=['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation',
    #                                                    'ColorHist', 'GISTDesc', 'LocalMaxima',
    #                                                    'LocalMinima', 'Mean_HSVYBGR'])

    # feature_cols = ['SIFTDesc', 'Brightness', 'ColorHist', 'Mean_HSVYBGR','Saturation', 'GISTDesc']


    X = features
    X_GIST = features_GIST
    X_all = np.concatenate((X, X_GIST), axis=1)
    # print(X_all.shape)

    O = features_test
    O_GIST = features_test_GIST
    O_all = np.concatenate((O, O_GIST), axis=1)
    # print(O_all.shape)

    y = train_df['Class']
    y_test = test_df['Class']

    
    # Models:

    # #### General ####
    # grid = model_1()
    # grid.fit(X, y)
    # # mean_scores = np.array(grid.cv_results_['mean_test_score'])

    # y_pred=grid.predict(features_test)
    # print("General feature classification")
    # display_results(y_test, y_pred)


    #### XGBoost ####
    # grid = XGBoost()
    # grid.fit(X, y)
    # best_param_values_xgb = grid.cv_results_['params'][grid.best_index_]
    
    best_param_values_xgb = {'subsample': 0.8, 'min_child_weight': 5, 'max_depth': 5, 'gamma': 1, 'colsample_bytree': 0.8}
    # print(best_param_values_xgb)

    # #### Mega Model ####
    grid = mega_combo_model(xgb_params=best_param_values_xgb)
    grid.fit(X, y)

    y_pred=grid.predict(features_test)

    print("Combo classification")
    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # print(feature_cols[0:13])
    # print(mean_scores)
    zipped = zip(mean_scores, feature_cols[0:13])
    for x in sorted(zipped, key = lambda t: t[0]):
        print(x)
    display_results(y_test, y_pred)



    
    # #### GIST KNN ####
    # KNN_GIST = Classifier().KNN()
    # KNN_GIST.fit(X_GIST , y)

    # y_pred_gist = KNN_GIST.predict(O_GIST)
    # print("GIST KNN classification")
    # display_results(y_test, y_pred_gist)


    # rfc = Classifier().RFC()
    # rfc.fit(X,y)

    # model = FeatureVectorSelector(11)
    # X_new = model.transform(X)
    # print(X_new.shape)

main()
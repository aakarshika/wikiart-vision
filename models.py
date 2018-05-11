
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
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

load_dotenv(find_dotenv())


cross_val_folds = 5

def gist_distance(x, y):
    d = (x - y) * (x - y)
    return np.sum(d)

class FeatureVectorSelector(object):
    def __call__(self, ftii=None):
        self.feature_to_ignore_index = ftii
    def set_param(self,ftii=None):
        self.feature_to_ignore_index = ftii
        
    def transform(self, X):

        X1 = []
        X2 = []

        if self.feature_to_ignore_index == 0:
            return X[:,features_lens_com[1]:108]
        
        X1 = X[:,0:features_lens_com[self.feature_to_ignore_index-1]]

        if self.feature_to_ignore_index < len(features_lens)-1:
            X2 = X[:,features_lens_com[self.feature_to_ignore_index+1]:108]
            return np.concatenate( (X1, X2), axis=1)
        elif self.feature_to_ignore_index == len(features_lens)-1:
            return X1
        raise ValueError("Feature vector index out of bounds.")

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
            objective="multi:softmax",
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

feature_cols = ["SIFTHist", "Brightness", "BHist","GHist","RHist", "Mean_H", "Mean_S","Mean_V","Mean_Y"," Mean_B","Mean_G","Mean_R","Saturation","EdgeCount","GISTHist"]
features_lens =     [25, 10, 10,10,10, 1, 1, 1, 1, 1, 1, 1, 10,   1,25]
features_lens_com = [25, 35, 45,55,65,66,67,68,69,70,71,72, 82, 83,108]
#                   [0,  1,  2   3  4  5  6  7  8  9  10 11 12  13   14]

def XGBoost():
    params={

        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5]
    }

    xgb = XGBClassifier(
        learning_rate=0.1, 
        n_estimators=50, 
        objective="multi:softmax",
        silent=True, 
        nthread=1)

    random_search = RandomizedSearchCV(xgb, 
        param_distributions=params,
        n_iter=5, 
        # scoring="roc_auc", 
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
        ("selectFeatures", sfm),
        ("classify", randomforest)
        ])

    # just checking
    # K_BestSelect = [20,30,50]
    K_Nearest_Neighboors = [5,10]
    SELECT_FEATURES = [sfm]
    CLASSIFY = [randomforest , knearest]

    param_grid = [
        {
            "selectFeatures": SELECT_FEATURES,
            "classify": [knearest],
            "classify__n_neighbors": K_Nearest_Neighboors
        }
        ,
        {
            "selectFeatures": SELECT_FEATURES,
            "classify": [randomforest] 
        }
    ]

    grid = GridSearchCV(pipe, cv=cross_val_folds, n_jobs=1, param_grid=param_grid)
    return grid

def display_results(y_test, y_pred):
    print(classification_report(y_test, y_pred))
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
        # ("selectFeatures", sfm),
        ("classify", xgb)
        ])

    # just checking
    # K_BestSelect = [20,30,50]
    K_Nearest_Neighboors = [5]
    SELECT_FEATURES = [
                    sfm
                    # fvs(0), 
                    # fvs(1), 
                    # fvs(2), 
                    # fvs(3), 
                    # fvs(4), 
                    # fvs(5), 
                    # fvs(6), 
                    # fvs(7), 
                    # fvs(8), 
                    # fvs(9), 
                    # fvs(10),
                    # fvs(11),
                    # fvs(12),
                    # fvs(13),
                    # fvs(14)
                    ]
    CLASSIFY = [randomforest , knearest]

    param_grid = [
        {
            # "selectFeatures": SELECT_FEATURES,
            "classify": [knearest],
            "classify__n_neighbors": K_Nearest_Neighboors
        }
        ,
        {
            # "selectFeatures": SELECT_FEATURES,
            "classify": [randomforest] 
        }
        ,
        {
            # "selectFeatures":SELECT_FEATURES,
            "classify": [xgb],
            "classify__min_child_weight": [xgb_params["min_child_weight"]],
            "classify__gamma": [xgb_params["gamma"]],
            "classify__subsample": [xgb_params["subsample"]],
            "classify__colsample_bytree": [xgb_params["colsample_bytree"]],
            "classify__max_depth": [xgb_params["max_depth"]]
        }
    ]
    grid = GridSearchCV(pipe, cv=cross_val_folds, n_jobs=4, param_grid=param_grid)
    return grid


genre_count = int(os.getenv("genre_count"))
img_count = int(os.getenv("img_count"))
# img_count = int(os.getenv("sample_img_count"))

def get_X_O_all(singles_scaled=True):
    

    features = np.load("data/features_train_{}.npy".format(genre_count*img_count))
    features_test = np.load("data/features_test_{}.npy".format(genre_count*img_count))
    
    features_test_EdgeCount = np.load("data/EdgeCount_test_{}.npy".format(genre_count*img_count))
    features_EdgeCount = np.load("data/EdgeCount_train_{}.npy".format(genre_count*img_count))
    
    features_test_GISTHist = np.load("data/GISTHist_test_{}.npy".format(genre_count*img_count))
    features_GISTHist = np.load("data/GISTHist_train_{}.npy".format(genre_count*img_count))
    
    # print(features.shape)
    # print(features_GIST.shape)
    # print(train_df.shape)
    X = np.concatenate((features, features_EdgeCount, features_GISTHist), axis=1)

    if singles_scaled:
        # find single features, scale them, put them back in X:

        f=X[:,features_lens_com[5] : features_lens_com[11]]
        Xs=np.concatenate((f,features_EdgeCount[:]), axis=1)
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(Xs)
        X[:,features_lens_com[5] : features_lens_com[11]] = Xs[:,0:6]
        X[:,features_lens_com[13]] = Xs[:,6]

    # X_GISTHist = features_GISTHist
    # print(X_GISTHist.shape)
    # print(X_ec.shape)
    # print(X.shape)
    # return X
    features_GIST = np.load("data/GISTDesc_train_{}.npy".format(genre_count*img_count))
    features_GIST = np.concatenate(features_GIST.tolist(), axis=0)
    features_test_GIST = np.load("data/GISTDesc_test_{}.npy".format(genre_count*img_count))
    features_test_GIST = np.concatenate(features_test_GIST.tolist(), axis=0)

    X_GIST = features_GIST

    O = np.concatenate((features_test, features_test_EdgeCount, features_test_GISTHist), axis=1)
    O_GIST = features_test_GIST
    # print(O.shape)
    
    return X, O, X_GIST, O_GIST
    


def main():
    
    train_df = pd.read_csv(os.path.join(os.getenv("dataset_location"), "train_{}.csv").format(genre_count*img_count), sep=";")
    test_df = pd.read_csv(os.path.join(os.getenv("dataset_location"), "test_{}.csv".format(genre_count*img_count)), sep=";")
    
    y = train_df["Class"]
    y_test = test_df["Class"]
    
    with_gist = True
    singles_scaled = False

    X, O, X_GIST, O_GIST = get_X_O_all(singles_scaled)
    print(X.shape)


    if with_gist:
        fvs=FeatureVectorSelector()
        fvs.set_param(14)


        X=fvs.transform(X)
        O=fvs.transform(O)
        print(X.shape)
    # Models:

    # #### General ####
    # grid = model_1()
    # grid.fit(X, y)
    # # mean_scores = np.array(grid.cv_results_["mean_test_score"])

    # y_pred=grid.predict(features_test)
    # print("General feature classification")
    # display_results(y_test, y_pred)
    if singles_scaled:
        print("SCALED.")
    else:
        print("UNSCALED.")

    # print("Combo classification: \n")
    
    # print(" Feature Selection:\n",
    # "\tSelectFromModel(RandomForest)\n ",
    # "Classifiers:\n", 
    #     "\tKNN\n",
    #     "\tRandomForestClassifier\n",
    #     "\tXGBoost\n")
    # print("Using parameters from best fit of XGBoost:")

    # # ### XGBoost ####
    # # grid = XGBoost()
    # # grid.fit(X, y)
    # # best_param_values_xgb = grid.cv_results_["params"][grid.best_index_]
    # # print(best_param_values_xgb)
    
    #### Mega Model ####
    best_param_values_xgb = {"subsample": 0.8, "min_child_weight": 5, "max_depth": 5, "gamma": 1, "colsample_bytree": 0.8}

    grid = mega_combo_model(xgb_params=best_param_values_xgb)
    grid.fit(X, y)
    y_pred=grid.predict(O)
    mean_scores = np.array(grid.cv_results_["mean_test_score"])

    print("Accuracy: ", max(mean_scores))
    
    display_results(y_test, y_pred)

    fi=grid.best_estimator_.named_steps["classify"].feature_importances_

    i=0
    flabels=[]
    for f in (feature_cols):
        for x in range(0,features_lens[i]):
            flabels.append(f)
        i=i+1

    zipped = zip(flabels, fi)
    for x in sorted(zipped, key = lambda t: t[1]):
        print(x)
    # print(grid.best_estimator_.named_steps["classify"].get_booster().get_fscore())

    # #### GIST KNN ####
    # KNN_GIST = Classifier().KNN()
    # KNN_GIST.fit(X_GIST , y)

    # y_pred_gist = KNN_GIST.predict(O_GIST)
    # print("\nGIST K-NearestNeighbour Classification\n")
    # display_results(y_test, y_pred_gist)


main()
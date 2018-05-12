"""
@author: Aakarshika Priydarshi
"""


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
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings(module="sklearn*", action="ignore", category=DeprecationWarning)

load_dotenv(find_dotenv())


cross_val_folds = 5

def gist_distance(x, y):
    d = (x - y) * (x - y)
    return np.sum(d)


class Classifier():

    def RFC(self):
        return RandomForestClassifier()
    def KNN(self):
        return KNeighborsClassifier(n_neighbors = 5)
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

def display_results(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cmat = confusion_matrix(y_test, y_pred, labels=range(genre_count))
    print("Confusion Matrix:")
    print(cmat)
    print("Class-wise accuracy:")
    print(cmat.diagonal()/cmat.sum(axis=1))
    # print(grid.best_estimator_)

    print("\n\n")


def model(c, select=False, xgb_params=None):

    clf = Classifier()
    knearest = None
    xgb = None
    randomforest = None
    
    steps=[]
    param_grid = []


    fs = FeatureSelector()
    randomforest = clf.RFC()
    sfm = fs.SFM(randomforest)
    if select:
        steps.append(("selectFeatures", sfm))
    
    if c == 'rf':
        randomforest = clf.RFC()
        param_grid=[{
            "classify": [randomforest]
        }]
        steps.append(("classify",randomforest))
    elif c == 'knn':
        knearest = clf.KNN()
        param_grid=[{
            "classify": [knearest]
        }]
        steps.append(("classify",knearest))
    elif c == 'xgb':
        xgb = clf.XGB()
        param_grid=[{
            "classify": [xgb],
            "classify__min_child_weight": [xgb_params["min_child_weight"]],
            "classify__gamma": [xgb_params["gamma"]],
            "classify__subsample": [xgb_params["subsample"]],
            "classify__colsample_bytree": [xgb_params["colsample_bytree"]],
            "classify__max_depth": [xgb_params["max_depth"]]
        }]
        steps.append(("classify",xgb))

    pipe = Pipeline(steps=steps)

    grid = GridSearchCV(pipe, cv=cross_val_folds, n_jobs=4, param_grid=param_grid)
    return grid


def fit_n_predict(clf,X,y,O,y_test, verbose=False):
    clf.fit(X, y)
    y_pred=clf.predict(O)

    mean_scores = np.array(clf.cv_results_["mean_test_score"])
    a = max(mean_scores)
    p,r,_,_ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    if verbose:
        display_results(y_test,y_pred)
    return a,p,r #accuracy, precision, recall average.


genre_count = int(os.getenv("genre_count"))
img_count = int(os.getenv("img_count"))
# img_count = int(os.getenv("sample_img_count"))


def main():
    
    train_df = pd.read_csv(os.path.join(os.getenv("dataset_location"), "train_{}.csv").format(genre_count*img_count), sep=";")
    test_df = pd.read_csv(os.path.join(os.getenv("dataset_location"), "test_{}.csv".format(genre_count*img_count)), sep=";")

    y = train_df["Class"]
    y_test = test_df["Class"]
    
    X=features_train = np.load("data/features_train_{}.npy".format(genre_count*img_count))
    O=features_test = np.load("data/features_test_{}.npy".format(genre_count*img_count))

    Xs=features_scaled_train = np.load("data/features_singlesscaled_train_{}.npy".format(genre_count*img_count))
    Os=features_scaled_test = np.load("data/features_singlesscaled_test_{}.npy".format(genre_count*img_count))

    Xg=GIST_desc_train = np.load("data/GIST_descriptors_train_{}.npy".format(genre_count*img_count))
    Og=GIST_desc_test = np.load("data/GIST_descriptors_test_{}.npy".format(genre_count*img_count))


    # Models:

    print("\nModel\t\tAccuracy\tPrecision\tRecall\n")

    # # grid = XGBoost()
    # # grid.fit(X, y)
    # # best_param_values_xgb = grid.cv_results_["params"][grid.best_index_]
    # best parameters were calculate using XGBoost() function that does RandomSearchCV.
    best_param_values_xgb = {"subsample": 0.8, "min_child_weight": 5, "max_depth": 5, "gamma": 1, "colsample_bytree": 0.8}

    clf=model('xgb',xgb_params=best_param_values_xgb)
    a,p,r = fit_n_predict(clf,X,y,O,y_test)
    print("XGBoost\t\t",a,p,r)

    clf=model('rf')
    a,p,r = fit_n_predict(clf,X,y,O,y_test)
    print("RandomForest\t",a,p,r)
    
    clf=model('knn')
    a,p,r = fit_n_predict(clf,X,y,O,y_test)
    print("K-NN\t\t",a,p,r)

    clf=model('knn')
    a,p,r = fit_n_predict(clf,Xg,y,Og,y_test)
    print("K-NN GIST\t",a,p,r)

    print("\nWith Select From Model\n")

    best_param_values_xgb = {"subsample": 0.8, "min_child_weight": 5, "max_depth": 5, "gamma": 1, "colsample_bytree": 0.8}
    clf=model('xgb',select=True,xgb_params=best_param_values_xgb)
    a,p,r = fit_n_predict(clf,X,y,O,y_test)
    print("XGBoost\t\t",a,p,r)

    clf=model('rf',select=True)
    a,p,r = fit_n_predict(clf,X,y,O,y_test)
    print("RandomForest\t",a,p,r)
    
    clf=model('knn',select=True)
    a,p,r = fit_n_predict(clf,X,y,O,y_test)
    print("K-NN\t\t",a,p,r)
    
    clf=model('knn',select=True)
    a,p,r = fit_n_predict(clf,Xg,y,Og,y_test)
    print("K-NN GIST\t",a,p,r)



main()
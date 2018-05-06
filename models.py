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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

load_dotenv(find_dotenv())

genre_count = int(os.getenv('genre_count'))
img_count = int(os.getenv('sample_img_count'))

cross_val_folds = 5


class Classifier():

    # def __init__(self):
        # self.X = X
        # self.y = y
    
    def RFC(self):
        return RandomForestClassifier()
    def XGBoost(self):
        print("XGBoost")
    def KNN(self):
        print("KNN")

class FeatureSelector():
    def SFM(self,clf):
        return SelectFromModel(clf)
    def SKB(self):
        return SelectKBest(chi2)

def importance(rfc):
    importances = rfc.feature_importances_
    # importances = pipe.named_steps.randomTree.feature_importances_

    print(importances)

    # std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
    #              axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])  )

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r"
           # , yerr=std[indices]
            ,align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()



def main():

    genre_count = int(os.getenv('genre_count'))
    img_count = int(os.getenv('sample_img_count'))

    train_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'train_100.csv'), sep=';')

    features = np.load('data/features_train_{}.npy'.format(100))


    print(features[0])

    print(train_df.shape)

    # features_df = pd.DataFrame(features, columns=['Painting', 'Class', 'Path', 'SIFTDesc', 'Brightness', 'Saturation',
    #                                                    'ColorHist', 'GISTDesc', 'LocalMaxima',
    #                                                    'LocalMinima', 'Mean_HSVYBGR'])

    feature_cols = ['SIFTDesc', 'Brightness', 'Saturation', 'ColorHist', 'Mean_HSVYBGR', 'GISTDesc']
    features_lens = [25 , 10 , 10 , 30, 8]

    # X = features
    X = features[0:100]
    y = train_df['Class']
    print(features)
    clf = Classifier()
    randomforest = clf.RFC()

    fs = FeatureSelector()
    sfm = fs.SFM(randomforest)
    skb = fs.SKB()



    pipe = Pipeline(steps=[
        ('selectFeatures', sfm),
        ('classify', randomforest)
        ])

    # just checking
    K_BEST_ = [20,30,50]
    
    param_grid = [
        {
            'selectFeatures': [sfm]
        }
        ,
        {
            'selectFeatures': [skb],
            'selectFeatures__k': K_BEST_
        }
    ]

    grid = GridSearchCV(pipe, cv=cross_val_folds, n_jobs=1, param_grid=param_grid)
    grid.fit(X, y)
    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # mean_scores = mean_scores.max(axis=0)
    print('SelectFromModel, Select20Best, Select30Best, Select50Best' )
    print(mean_scores)


    
    # importance(randomforest)

    # model = SelectFromModel(randomforest, prefit=True)
    # X_new = model.transform(X=features)

    # scores = cross_val_score(pipe, X, y, cv=cross_val_folds)
    # predicted = cross_val_predict(pipe, X, y, cv=cross_val_folds)
    # m=metrics.accuracy_score(y, predicted)
    # print(m)



    test_df = pd.read_csv(os.path.join(os.getenv('dataset_location'), 'test_{}.csv'.format(genre_count*img_count)), sep=';')
    y_test = test_df['Class']
    features_test = np.load('data/features_test_{}.npy'.format(genre_count*img_count))
    y_pred=grid.predict(features_test[:100])

    print(y_pred)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(genre_count)))



main()
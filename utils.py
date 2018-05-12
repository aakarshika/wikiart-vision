
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv


from sklearn.preprocessing import MinMaxScaler
load_dotenv(find_dotenv())


# 				0				1 			2        3       4
feature_cols = ["SIFTHist", "Brightness", "BHist","GHist","RHist", 
#  5           6     7        8         9       10        11         12        13
"Mean_H", "Mean_S","Mean_V","Mean_Y"," Mean_B","Mean_G","Mean_R","Saturation","EdgeCount",
#  14
"GISTHist"]
features_lens =     [25, 10, 10,10,10, 1, 1, 1, 1, 1, 1, 1, 10,   1,25]
features_lens_com = [25, 35, 45,55,65,66,67,68,69,70,71,72, 82, 83,108]
f_start = 			[0,  25, 35,45,55,65,66,67,68,69,70,71,72,82,83,108]
#                   [0,  1,  2   3  4  5  6  7  8  9  10 11 12 13 14]

nfeature_cols = [ "BHist","GHist","RHist","Brightness","Saturation","SIFTHist", "GISTHist" "Mean_H", "Mean_S","Mean_V","Mean_Y"," Mean_B","Mean_G","Mean_R","EdgeCount"]
nfeatures_lens =     [10, 10, 10, 10 ,10, 25, 25,  1,  1,  1,  1,  1,  1,  1,  1]
nfeatures_lens_com = [10, 20, 30, 40, 50, 75, 100, 101,102,103,104,105,106,107,108]
#                     [0,  1,  2   3  4   5    6   7   8   9   10  11  12  13   14]

def FeatureVectorSelector(X,i):
	v=X[:,f_start[i]:f_start[i+1]]
	print(feature_cols[i])
	print(v.shape)
	return v



genre_count = 10
img_count = 100
count = img_count*genre_count

train_df = pd.read_csv(os.path.join(os.getenv("dataset_location"), "train_{}.csv").format(genre_count*img_count), sep=";")
test_df = pd.read_csv(os.path.join(os.getenv("dataset_location"), "test_{}.csv".format(genre_count*img_count)), sep=";")

y = train_df["Class"]
y_test = test_df["Class"]

with_gist = True
singles_scaled = False

features = np.load("data2/features_train_{}.npy".format(genre_count*img_count))
features_test = np.load("data2/features_test_{}.npy".format(genre_count*img_count))

features_test_EdgeCount = np.load("data2/EdgeCount_test_{}.npy".format(genre_count*img_count))
features_EdgeCount = np.load("data2/EdgeCount_train_{}.npy".format(genre_count*img_count))

features_test_GISTHist = np.load("data2/GISTHist_test_{}.npy".format(genre_count*img_count))
features_GISTHist = np.load("data2/GISTHist_train_{}.npy".format(genre_count*img_count))

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
features_GIST = np.load("data2/GISTDesc_train_{}.npy".format(genre_count*img_count))
features_GIST = np.concatenate(features_GIST.tolist(), axis=0)
features_test_GIST = np.load("data2/GISTDesc_test_{}.npy".format(genre_count*img_count))
features_test_GIST = np.concatenate(features_test_GIST.tolist(), axis=0)

X_GIST = features_GIST

O = np.concatenate((features_test, features_test_EdgeCount, features_test_GISTHist), axis=1)
O_GIST = features_test_GIST


# print(X.shape)
# print(X_GIST.shape)
# print(O.shape)
# print(X_GIST.shape)


# 				0				1 			2        3       4
feature_cols = ["SIFTHist", "Brightness", "BHist","GHist","RHist", 
#  5           6     7        8         9       10        11         12        13
"Mean_H", "Mean_S","Mean_V","Mean_Y"," Mean_B","Mean_G","Mean_R","Saturation","EdgeCount",
#  14
"GISTHist"]
features_lens =     [25, 10, 10,10,10, 1, 1, 1, 1, 1, 1, 1, 10,   1,25]
features_lens_com = [25, 35, 45,55,65,66,67,68,69,70,71,72, 82, 83,108]
f_start = 			[0,  25, 35,45,55,65,66,67,68,69,70,71,72,82,83,108]
#                   [0,  1,  2   3  4  5  6  7  8  9  10 11 12 13 14]

nfeature_cols = [ "BHist","GHist","RHist","Brightness","Saturation","SIFTHist", "GISTHist" "Mean_H", "Mean_S","Mean_V","Mean_Y"," Mean_B","Mean_G","Mean_R","EdgeCount"]
nfeatures_lens =     [10, 10, 10, 10 ,10, 25, 25,  1,  1,  1,  1,  1,  1,  1,  1]
nfeatures_lens_com = [10, 20, 30, 40, 50, 75, 100, 101,102,103,104,105,106,107,108]
#                     [0,  1,  2   3  4   5    6   7   8   9   10  11  12  13   14]



newX = FeatureVectorSelector(X,2)

newX = np.concatenate((newX, FeatureVectorSelector(X,3)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,4)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,1)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,12)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,0)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,14)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,5)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,6)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,7)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,8)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,9)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,10)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,11)), axis =1)
newX = np.concatenate((newX, FeatureVectorSelector(X,13)), axis =1)

print(newX.shape)




newO = FeatureVectorSelector(O,2)

newO = np.concatenate((newO, FeatureVectorSelector(O,3)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,4)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,1)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,12)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,0)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,14)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,5)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,6)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,7)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,8)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,9)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,10)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,11)), axis =1)
newO = np.concatenate((newO, FeatureVectorSelector(O,13)), axis =1)

print(newO.shape)


np.save('newdata/features_train_{}.npy'.format(count), newX)
# np.save('newdata/GIST_descriptors_train_{}.npy'.format(count), O_GIST)

np.save('newdata/features_test_{}.npy'.format(count), newO)
# np.save('newdata/GIST_descriptors_test_{}.npy'.format(count), O_GIST)

## Builds and saves a model for later use based on claim and not claim dataset
##

# import relevant libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import pickle

# load data
allFeatureName = pd.read_pickle('../Data/allFeatureNames.pkl') # feature names
needCiteFeat = np.load('../Data/NeedCiteFeatMat.npy') # feature matrix of claim dataset
notClaimFeat = np.load('../Data/NotClaimFeatMat.npy') # feature matrix of not claim dataset

# concatenate all data
allFeats = np.concatenate((needCiteFeat, notClaimFeat), 0)
allClass = np.concatenate((np.ones((len(needCiteFeat), 1)), np.zeros((len(notClaimFeat), 1))), 0)
needCiteFeat = list() # clear data to save memory
notClaimFeat = list()

# feature number 2353 is ']' and is irrelevant, an error in data pre-processing
allFeats[:,2353] = 0

# reorder elements
reOrder = np.random.permutation(len(allFeats))
allFeats = allFeats[reOrder,:]
allClass = allClass[reOrder,:]
y = np.ravel(allClass)

# perform random forest classification to get feature importances
clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(allFeats, y)
featImp = clf.feature_importances_ # get feature importances
featImpSort = np.argsort(featImp) # sort feature importances

# rearrange dimensions so they are sorted by feature importance
allFeats_sort = np.zeros((len(allFeats), int(allFeats.shape[1])))
featName_sort = list()
for i in range(0,len(featImpSort)):
    indx = featImpSort[-(i + 1)]
    allFeats_sort[:,i] = allFeats[:,indx]
    featName_sort.append(allFeatureName.FeatName[indx])

## build a model with only 600 predictors since we found that accuracy does not
## change that much after the first 600 most important predictors
subFeats = allFeats_sort[:, 0:599]
subFeatName = featName_sort[0:599]
myModel = ExtraTreesClassifier(n_estimators=10)
myModel.fit(subFeats, y)

## save the model to disk
filename = '../Data/finalized_model.sav'
pickle.dump(myModel, open(filename, 'wb'))

## save the feature names, feature matrix, and class labels used to create the model
np.save('../Data/Features_finalized_model.npy', subFeats)
np.save('../Data/ClassLabels_finalized_model.npy', y)
subFeatArr = np.array(subFeatName)
np.save('../Data/FeatureName_finalized_model.npy', subFeatArr)

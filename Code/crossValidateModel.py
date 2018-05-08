## Read in feature matrices and class labels and test performance of Extremely random forest ExtraTreesClassifier
## 1) loads in feature matrices and class labels from claim and not claim datasets
## 2) trains a ExtraTreesClassifier on the entire dataset to do feature selection
## 3) orders the feature matrix by the order of increasing importance of features
## 4) iteratively trains extra trees classifiers on increasing subsets of features with cross validation
## 5) saves results of study

# import relevant libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# define a function that perform cross validation with train-test splitting
# on a given dataset
def RandomForestCrossVal(X, y, numCrossVals, testRat):
    Accu_0 = np.zeros(numCrossVals)
    Accu_1 = np.zeros(numCrossVals)
    Accu_a = np.zeros(numCrossVals)
    y = np.ravel(y)

    for i in range(0,numCrossVals):
        clf = ExtraTreesClassifier(n_estimators=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRat)
        clf.fit(X_train, y_train)
        myPred_forest = clf.predict(X_test)
        forestConfMat = confusion_matrix(y_test, myPred_forest)
        Accu_0[i] = forestConfMat[0,0]/float(np.sum(forestConfMat[:,0]))
        Accu_1[i] = forestConfMat[1,1]/float(np.sum(forestConfMat[:,1]))
        Accu_a[i] = (forestConfMat[0,0] + forestConfMat[1,1])/float(len(y_test))

    Accu_0_mean = np.mean(Accu_0)
    Accu_0_std = np.std(Accu_0)
    Accu_1_mean = np.mean(Accu_1)
    Accu_1_std = np.std(Accu_1)
    Accu_a_mean = np.mean(Accu_a)
    Accu_a_std = np.std(Accu_a)

    return Accu_0_mean, Accu_0_std, Accu_1_mean, Accu_1_std, Accu_a_mean, Accu_a_std


# load data
allFeatureName = pd.read_pickle('../Data/allFeatureNames.pkl') # get names of features
needCiteFeat = np.load('../Data/NeedCiteFeatMat.npy') # feature matrix for citation needed dataset
notClaimFeat = np.load('../Data/NotClaimFeatMat.npy') # feature matrix for no citation needed (not a claim) dataset

# concatenate all data
allFeats = np.concatenate((needCiteFeat, notClaimFeat), 0)
allClass = np.concatenate((np.ones((len(needCiteFeat), 1)), np.zeros((len(notClaimFeat), 1))), 0)
needCiteFeat = list() # clear data to save memory
notClaimFeat = list()

# feature number 2353 is character ']' and is irrelevant, an error in data pre processing
allFeats[:,2353] = 0

# randomly reorder elements
reOrder = np.random.permutation(len(allFeats))
allFeats = allFeats[reOrder,:]
allClass = allClass[reOrder,:]
y = np.ravel(allClass)

# perform random forest classification to get feature importances
clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(allFeats, y)
featImp = clf.feature_importances_ # get feature importances
featImpSort = np.argsort(featImp)

# rearrange dimensions so they are sorted by feature importance
allFeats_sort = np.zeros((len(allFeats), int(allFeats.shape[1])))
featName_sort = list()
for i in range(0,len(featImpSort)):
    indx = featImpSort[-(i + 1)]
    allFeats_sort[:,i] = allFeats[:,indx]
    featName_sort.append(allFeatureName.FeatName[indx])

# create subsets of dimensions to use based on importance 100-1000 so it is less than 100 fold of Ns
testRat = 0.2
numCrossVals = 10
dimsToTry = np.arange(100,1100,100)
dimMeanAccu0 = np.zeros(len(dimsToTry))
dimStdAccu0 = np.zeros(len(dimsToTry))
dimMeanAccu1 = np.zeros(len(dimsToTry))
dimStdAccu1 = np.zeros(len(dimsToTry))
dimMeanAccua = np.zeros(len(dimsToTry))
dimStdAccua = np.zeros(len(dimsToTry))

# perform 10-fold cross validation on each dimension
for i in range(0,len(dimsToTry)):
    dims = dimsToTry[i]
    thisX = allFeats_sort[:,0:dims]
    Accu_0_mean, Accu_0_std, Accu_1_mean, Accu_1_std, Accu_a_mean, Accu_a_std = RandomForestCrossVal(thisX, y, numCrossVals, testRat)
    dimMeanAccu0[i] = Accu_0_mean
    dimStdAccu0[i] = Accu_0_std
    dimMeanAccu1[i] = Accu_1_mean
    dimStdAccu1[i] = Accu_1_std
    dimMeanAccua[i] = Accu_a_mean
    dimStdAccua[i] = Accu_a_std

# create dataframe to save cross validation results
CrossVal_extraTrees_10estimators = pd.DataFrame({
    'NumOfFeatDims': dimsToTry,
    'NotClaimAccu': dimMeanAccu0,
    'NotClaimAccuStd': dimStdAccu0,
    'ClaimAccu': dimMeanAccu1,
    'ClaimAccuStd': dimStdAccu1,
    'OverallAccu': dimMeanAccua,
    'OverallAccuStd': dimStdAccua
})

# save dataframe
CrossVal_extraTrees_10estimators.to_pickle('../Data/CrossValidationResults_ExtraTrees_10estimators.pkl')

##
##
##

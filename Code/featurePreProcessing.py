## Feature selection of citation needed and no citation needed (not a claim) datasets
## This code reads in the output from "proto_featureExtract.py"
## 1) the part of speech filtered data sets including both words and parts of speech
## 2) the counts

# import relevant libraries
import pandas as pd
import numpy as np

# load in data
NeedCite = pd.read_pickle('../Data/NeedCiteFilt.pkl') # part of speech filtered clauses and sentences that need citations
NotClaim = pd.read_pickle('../Data/NotClaimFilt.pkl') # that do not need citations

UniqWords = pd.read_pickle('../Data/UniqueWords.pkl') # word occurances in the above datasets
UniqPOS = pd.read_pickle('../Data/UniquePOS.pkl') # part of speech occurances in the above datasets

## order words by most and least occurting
totOccur_claim = float(np.sum(np.array(UniqWords.WordOccurClaim)))
wordProp_claim = np.array(UniqWords.WordOccurClaim)/totOccur_claim
#### divide occurance of each word by total number of occurances of all words to get a proportion

totOccur_notClaim = float(np.sum(np.array(UniqWords.WordOccurNotClaim)))
wordProp_notClaim = np.array(UniqWords.WordOccurNotClaim)/totOccur_notClaim

wordProp_claim_sort = np.argsort(wordProp_claim)
wordProp_notClaim_sort = np.argsort(wordProp_notClaim)

## get ratio of occurance proportions
wordProp_ratio = wordProp_claim/wordProp_notClaim
wordProp_ratio_adj = wordProp_ratio/(totOccur_claim/totOccur_notClaim)
#### ratios greater than 1 suggests that the word or part of speech occurs (by proportion) more often
#### in the claim (citation needed) dataset while ratios less than 1
#### suggest that the word or part of speech occurs more often in the not claim
#### (citation not needed) dataset.

## filter words based on occurance and ratio thresholds
#### we want words that occur often and have large disparities in occurance in the two datasets
#### it helps to plot the proportion against the ratio to see why I picked these
#### thresholds
#### e.g. plot(wordProp_claim, wordProp_ratio_adj)
propThresh = 0.000015 # occurance threshold (these may need to be changed )
ratioThresh_hi = 1.25 # ratio threshold for more common than not claims
ratioThresh_lo = 1/ratioThresh_hi # ratio threhsold for words less common that not claims
isProp = wordProp_claim > propThresh
lessThanInf = np.logical_and(wordProp_ratio_adj > ratioThresh_hi, wordProp_ratio_adj < np.Inf)
bigThanZero = np.logical_and(wordProp_ratio_adj < ratioThresh_lo, wordProp_ratio_adj > 0)
isRatio = np.logical_or(lessThanInf, bigThanZero)
isInRegion = np.logical_and(isProp, isRatio)
totWords = np.sum(isInRegion)

# get words that have high probabilities of occurance and large or small ratios
isInRegionIndx = [j for j, x in enumerate(isInRegion) if x] # grab index of words
uniqWords_filt = list()
for indx in isInRegionIndx:
    uniqWords_filt.append(UniqWords.UniqueWords[indx])

# each filtered sentence will be represented as a binary vector where each element signifies
# if a word or part-of-speech occurs in the sentence or not
allFeatures = list()
allFeatures.append(uniqWords_filt)
allFeatures.append(UniqPOS.UniquePOS)
allFeature_flat = [item for sublist in allFeatures for item in sublist]
needCiteFeats_word = np.zeros((len(NeedCite), totWords))
needCiteFeats_POS = np.zeros((len(NeedCite), len(UniqPOS)))
needCiteClass = np.ones((len(NeedCite), 1))
notClaimFeats_word = np.zeros((len(NotClaim), totWords))
notClaimFeats_POS = np.zeros((len(NotClaim), len(UniqPOS)))
notClaimClass = np.zeros((len(NeedCite), 1))

# go through each sentence in list of sentences and populate feature vectors
for i in range(0,len(NeedCite)):
    thisSent = NeedCite.NeedCiteWord[i]
    thisPOSs = NeedCite.NeedCitePOS[i]
    for j in range(0,totWords):
        thisWord = uniqWords_filt[j]
        if thisWord in thisSent:
            needCiteFeats_word[i,j] = 1
    for k in range(0,len(UniqPOS)):
        thisPOS = UniqPOS.UniquePOS[k]
        if thisPOS in thisPOSs:
            needCiteFeats_POS[i,k] = 1

for i in range(0,len(NotClaim)):
    thisSent = NotClaim.NotClaimWord[i]
    thisPOSs = NotClaim.NotClaimPOS[i]
    for j in range(0,totWords):
        thisWord = uniqWords_filt[j]
        if thisWord in thisSent:
            notClaimFeats_word[i,j] = 1
    for k in range(0,len(UniqPOS)):
        thisPOS = UniqPOS.UniquePOS[k]
        if thisPOS in thisPOSs:
            notClaimFeats_POS[i,k] = 1

# create full feature matrix
NeedCite_fullFeats = np.concatenate((needCiteFeats_word, needCiteFeats_POS), 1)
NotClaim_fullFeats = np.concatenate((notClaimFeats_word, notClaimFeats_POS), 1)
allFeatures = pd.DataFrame({
    'FeatName': allFeature_flat
})

# save feature matrices and feature names
np.save('../Data/NeedCiteFeatMat.pkl', NeedCite_fullFeats)
np.save('../Data/NotClaimFeatMat.pkl', NotClaim_fullFeats)
allFeatures.to_pickle('../Data/allFeatureNames.pkl')

##
##
##

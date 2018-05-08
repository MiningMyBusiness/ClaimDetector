## Extract features from clauses and sentences that need citations and those that do not
## this code reads in two data files:
## 1) one contains sentences and clauses that need citations
## 2) the other contains sentences that do no
## Then is filters the words in the sentence by parts of speech and stems the words
## It also calculates the occurance of the unique words and parts of speech in the two datasets
## finally it saves these filtered data sets and the counts of the unique features in each dataset

# import relevant libraries
import nltk
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer

# Create p_stemmer object
p_stemmer = SnowballStemmer("english", ignore_stopwords=True)

# load data which contain sentences that need citations and sentences that do not (are not claims)
needCite = pd.read_pickle('../Data/CitationNeeded.pkl') # need citations
noClaim = pd.read_pickle('../Data/NotACLaim.pkl') # do NOT need citations (are not claims)

# tokenize sentences into words and tag parts of speech
# keep nouns (NN), adjectives (JJ), verbs (VB), adverbs (RB), numberical/cardinal (CD),
# determiner (DT)

# features will include words that are any of the previous, the length of the sentence or clause

needCite_filtSent = list() # list to store word tokenized and filtered sentences from citation needed list
needCite_wordTag = list() # list to store the part of speech of each word
noClaim_filtSent = list() # list to store word tokenized and filtered sentences from not a claim list
noClaim_wordTag = list() # list to store the part of speech of each word
allWordList = list() # list that stores all words in both data sentences
allPOSList = list() # list that stores all POS of all words in both datasets

for sent in needCite.CitationNeeded:
    sent_token = nltk.word_tokenize(sent) # word tokenize the sentence
    sent_pos = nltk.pos_tag(sent_token) # tag with part of speech
    sent_filt_word = list() # create list to store filtered sentence words
    sent_filt_pos = list() # create list to store the filtered parts of speech
    for item in sent_pos: # for each item in the sentence
        if len(item) > 1:
            thisTag = item[1] # grab the part of speech
            if 'NN' in thisTag or 'JJ' in thisTag or 'VB' in thisTag or 'RB' in thisTag or 'CD' in thisTag or 'DT' in thisTag: # if the tag is an approved part of speech
                thisWord = item[0].encode('ascii', 'ignore')
                sent_filt_word.append(p_stemmer.stem(thisWord.lower()))
                sent_filt_pos.append(thisTag)
                allWordList.append(p_stemmer.stem(thisWord.lower()))
                allPOSList.append(thisTag)
    needCite_filtSent.append(sent_filt_word)
    needCite_wordTag.append(sent_filt_pos)

for sent in noClaim.NotAClaim:
    sent_token = nltk.word_tokenize(sent) # word tokenize the sentence
    sent_pos = nltk.pos_tag(sent_token) # tag with part of speech
    sent_filt_word = list() # create list to store filtered sentence words
    sent_filt_pos = list() # create list to store the filtered parts of speech
    for item in sent_pos: # for each item in the sentence
        if len(item) > 1:
            thisTag = item[1] # grab the part of speech
            if 'NN' in thisTag or 'JJ' in thisTag or 'VB' in thisTag or 'RB' in thisTag or 'CD' in thisTag or 'DT' in thisTag: # if the tag is an approved part of speech
                thisWord = item[0].encode('ascii', 'ignore')
                sent_filt_word.append(p_stemmer.stem(thisWord.lower()))
                sent_filt_pos.append(thisTag)
                allWordList.append(p_stemmer.stem(thisWord.lower()))
                allPOSList.append(thisTag)
    noClaim_filtSent.append(sent_filt_word)
    noClaim_wordTag.append(sent_filt_pos)

## compute word occurances in sentences
uniqWords = list(set(allWordList)) # find all uniqwords in the dataset
wordOccur_claim  = list() # list to store number of times word occurs in claim dataset
wordOccur_notClaim = list() # list to store number of times word occurs in not claim data set

for i in range(0,len(uniqWords)): # for each word
    word = uniqWords[i]
    numOfTimes = 0
    for sent in needCite_filtSent:
        if word in sent:
            numOfTimes = numOfTimes + len([j for j, x in enumerate(sent) if x == word])
    wordOccur_claim.append(numOfTimes)
    numOfTimes = 0
    for sent in noClaim_filtSent:
        if word in sent:
            numOfTimes = numOfTimes + len([j for j, x in enumerate(sent) if x == word])
    wordOccur_notClaim.append(numOfTimes)

## compute POS occurances in sentences
uniqPOS = list(set(allPOSList)) # find all uniqwords in the dataset
posOccur_claim = list() # for part of speech
posOccur_notClaim = list()

for i in range(0,len(uniqPOS)): # for each word
    word = uniqPOS[i]
    numOfTimes = 0
    for sent in needCite_wordTag:
        if word in sent:
            numOfTimes = numOfTimes + len([j for j, x in enumerate(sent) if x == word])
    posOccur_claim.append(numOfTimes)
    numOfTimes = 0
    for sent in noClaim_wordTag:
        if word in sent:
            numOfTimes = numOfTimes + len([j for j, x in enumerate(sent) if x == word])
    posOccur_notClaim.append(numOfTimes)

## save all data
UniqWords = pd.DataFrame(
    {'UniqueWords': uniqWords,
    'WordOccurClaim': wordOccur_claim,
    'WordOccurNotClaim': wordOccur_notClaim
    })
UniqWords.to_pickle('../Data/UniqueWords.pkl')

UniqPOS = pd.DataFrame(
    {'UniquePOS': uniqPOS,
    'POSOccurClaim': posOccur_claim,
    'POSOccurNotClaim': posOccur_notClaim
    })
UniqPOS.to_pickle('../Data/UniquePOS.pkl')

NeedCite = pd.DataFrame(
    {'NeedCiteWord': needCite_filtSent,
    'NeedCitePOS': needCite_wordTag
    })
NeedCite.to_pickle('../Data/NeedCiteFilt.pkl')

NotClaim = pd.DataFrame(
    {'NotClaimWord': noClaim_filtSent,
    'NotClaimPOS': noClaim_wordTag
    })
NotClaim.to_pickle('../Data/NotClaimFilt.pkl')





##
##
##

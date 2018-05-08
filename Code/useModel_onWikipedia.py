## Run saved model on wikipedia page to classify text as needing citation
## 1) loads in text from wikipedia page with python wikipedia library
## 2) breaks text into sentences
## 3) gets parts of speech and stems words to create binary feature vector
## 4) tests each binary feature vector representing a sentence with the model
## 5) saves output in a text file where each line is a sentence with a 0 or 1 to
##### indicate if the sentence needs citation or not.

# import relevant libraries
import wikipedia
import nltk
import pandas as pd
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer

## Create p_stemmer of class PorterStemmer
p_stemmer = SnowballStemmer("english", ignore_stopwords=True)

## pull text from wikipedia page
wikiPage = wikipedia.page("Perlin noise") # CHANGE PAGE NAME IF DESIRED
content = wikiPage.content # Content of page

## extract sentences from paragraph
fullSents = nltk.sent_tokenize(content)

## extract words from sentences
## find parts of speech of words
wordsInSents = list()
posInSents = list()
for sent in fullSents:
    sent_token = nltk.word_tokenize(sent) # word tokenize the sentence
    sent_pos = nltk.pos_tag(sent_token) # tag with part of speech
    stemWords = list() # create list to store stemmed words
    onlyPOS = list() # create list to store sentence parts of speech
    for item in sent_pos:
        word = item[0]
        stemWords.append(p_stemmer.stem(word.lower())) # lower case and stem words
        onlyPOS.append(item[1])
    wordsInSents.append(stemWords)
    posInSents.append(onlyPOS)

## go through each word and the pos of each word
## to see if it matches the 600 feature names of the
## finalized model to create the 600 element binary
## vector
featureNames = np.load("../Data/FeatureName_finalized_model.npy")
featureMat = np.zeros((len(wordsInSents), len(featureNames)))
for i in range(0,len(wordsInSents)):
    words = wordsInSents[i]
    POSes = posInSents[i]
    for j in range(0,len(featureNames)):
        name = featureNames[j]
        isWord = name in words
        isPOS = name in POSes
        if isWord or isPOS:
            featureMat[i,j] = 1

## feed the binary vector into the model to get a prediction
loaded_model = pickle.load(open("../Data/finalized_model.sav", "rb"))
myPredictions = loaded_model.predict(featureMat)

## write out each sentence and its prediction into
## a new text file
file = open("../Data/classifiedText_wikipedia.txt", "w")

for i in range(0,len(fullSents)):
    sent = fullSents[i]
    sent = sent.encode("ascii", "ignore")
    sent_complete = sent + "    " + str(myPredictions[i]) + "\n"
    file.write(sent_complete)

file.close()

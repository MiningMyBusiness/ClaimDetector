# Details on code files 

### Master.py
The master run file that runs all relevant code files in order. The output of each code file is the input for the next file in that runs. 

### featureExtract.py
This file reads in two data files, one of which contains a data frame with clauses or sentences that need citations and the other contains a data frame with sentences that do not need citations. It filters the words by parts of speech and saves new data files which contain the filtered sentences and the unique words and parts of speech in the dataset. 

### featurePreProcessing.py
This file reads in the data files saved by featureExtract.py as input. It picks a subset of the unique words and parts of speech based on the frequency of occurance in each dataset of claims and not claims and also the ratio of frequency of occurance in the two datasets. We would want to select words and parts of speech that occur often and at different rates in the two classes. 

### crossValidateModel.py
This file takes as input the data files saved as output from featurePreProcessing.py and builds a model for classification. First it reads in the dataset and builds an extremely random forest classifier on the entire dataset with all features. It uses the feature importance metric from the model to reorder features by importance. Then it successively builds models with larger subsets of the features with cross validation to discover the fewest number of features necessary to get good performance. 

### buildSaveModel.py
This file takes as input the data files saved as output from featureProcessing.py and builds one model with a subset of the features. The number of features is selected based on the results from the crossValidateModel.py run. 

### useModel_onWikipedia.py
This file uses the output of the buildSaveModel.py and runs it on the text on a wikipedia page. It processes the text of a wikipedia page (the page is defined in the code). Takes each sentence as input and puts it through the classifier to get a value 0 or 1, where 0 is needs no citation and 1 is needs citation. It saves a text file where each sentence is a new line and the score is at the end of the sentence. 

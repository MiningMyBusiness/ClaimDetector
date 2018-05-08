## Master run file
## Executes all files related to building a classifier model
## for citation needed and not a claim data set in order

# extract features from raw data
execfile('featureExtract.py')

# pre process features before model building
execfile('featurePreProcessing.py')

# build and cross validate models with subsets of features
execfile('crossValidateModel.py')

# build and save the final model
execfile('buildSaveModel.py')

# use the model of a wikipedia page
execfile('useModel_onWikipedia.py')

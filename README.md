# Claim Detector
Text classification project to detect sentences that are claims which need corroboration or citations. 
This project is made in Python 2.7.13

### Project overview 
This project aims to create an an automated classifier for detecting whether sentences need citations or corroboration. The ability to automatically identify such sentences will help in any fact-checking pipeline.

The classifier was trained on data scraped from Wikipedia's list of all articles with unsourced statements (https://en.wikipedia.org/wiki/Category:All_articles_with_unsourced_statements). The dataset has over 100,000 clauses and sentences of which about half had the "citation needed" designation and the other half had no such designation and no existing citations, putting them in the category of claims that do not require corroboration. 

### Description of repo
The code folder has all relevant code which processes and classifies the data in the Data folder. The "Master.py" file must be run once for all relevant data folders to be generated. This may take many hours...

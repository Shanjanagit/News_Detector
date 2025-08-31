## News_Detector
This is a news detector which detects whether the input news given by the user is a Fake news or a True news.

# Fake News Detector
A Machine Learning project to classify news articles as Fake or True using Natural Language Processing (NLP) techniques.


# Installation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Overall process

1. Text preprocessing - For this the Regex has been used to remove the unwanted punctuation marks and for removing the spaces.
2. TF-IDF vectorization for dealing feature extraction
3. Used 4 main models for training the model and to develop best results for any input and to predict proper results.
Through the following models, achived an accuracy of 99%.
The models which has been used for this are:
a. Logistic Regression
b. Decision Tree classifier
c. Gradient boosting classifier
d. Random Forest Classifier

# Data Exploration
Explored the True and Fake dataset using the head, tail, info methods

# EDA
Analyzed the EDA apporoach through checking the null values, duplicated values.

# Data Preprocessing
Here comes the text classification task as here similar words has been collected and classified using the embedding algorithm TFIDF - Vectorizer. 

# Data training
The training has been done through 4 main algorithms were three algorithms were classification algorithms used for dealing with numeric data and one regression algorithm used for dealing with categorical data.

# Evaluation of model performance
A final function has been developed which checks whether the input given by the user is Fake news or not a fake news.

# Visualization Techniques
The visualization can be done through the implementation of matplotlib and seaborn

## Dataset 
For dataset kindly refer this link:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

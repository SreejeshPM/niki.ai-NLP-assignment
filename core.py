# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 00:12:19 2018

@author: Sreejesh
"""
import string
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics import classification_report
from sklearn import  svm
import pickle

header = ["utterance", "label"]
data = pd.read_csv('LabelledData_Modified.txt',sep=',',header=None, names=header)
train_data = data[:1300]
test_data = data[1301:]


"""Preprocessing of data.
First we call data to a non ascii remover to normalise the data to ASCII standard.
followed by a tokenization and finally stemming. These stemed words will be feeded to a tfid vectorizer. 
Whichvectorizes the strings and removes stop words. And resulted vector can be called for preprocessing.
"""

def remove_non_ascii(textVal):
    return ''.join([i if ord(i) < 128 else ' ' for i in textVal])

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
train_vectors = vectorizer.fit_transform(train_data.utterance)
test_vectors = vectorizer.transform(test_data.utterance)

"""Estimator 
The estimator used is called SVM.SVC(Support vector classifier) with linear kernel.
With default paramters."""

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_data.label)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_data.label, prediction_linear))


"""Model persisitance with pickle
The resulant model is pushed as pickle file. Were further prediction calls can be 
done. 
"""

with open ('labelpredicter.pkl','wb') as f:
    pickle.dump((vectorizer,classifier_linear), f)

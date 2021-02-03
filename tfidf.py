# roshan shrestha
# document classification

# imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

alpha = 0.3
dataset = []

# read documents
# title contains the name of the folder where
# the documents are stored
title = 'docs'
folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]
# remove trailing '/'
folders[0] = folders[0][:len(folders[0])-1]

for folder in folders:
    # print(folder)
    files = os.listdir(folder+'/')
    txtFiles = [f for f in files if f.endswith('.txt')]
    for txtFile in txtFiles:
        # print(txtFile)
        dataset.append(str(folder)+'/'+str(txtFile))

def print_doc(id):
    print(dataset[id])
    '''
    f = open(dataset[id], 'r', encoding='utf8', errors='ignore')
    text = f.read().strip()
    f.close()
    print(text)
    '''

# preprocessing
def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data

processed_text = []

# extracting data
for d in dataset:
    f = open(d, 'r', encoding='utf8', errors='ignore')
    text = f.read().strip()
    f.close()
    processed_text.append(word_tokenize(str(preprocess(text))))

# calculate DF
DF = {}
N = len(dataset)

for i in range(N):
    tokens = processed_text[i]
    for token in tokens:
        try:
            DF[token].add[i]
        except:
            DF[token] = {i}
for i in DF:
    DF[i] = len(DF[i])

total_vocab_size = len(DF)

total_vocab = [x for x in DF]

def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

# calculate TF-IDF
doc = 0
tf_idf = {}

for i in range(N):
    tokens = processed_text[i]
    counter = Counter(tokens)
    words_count = len(tokens)

    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1))
        tf_idf[doc, token] = tf*idf
    doc += 1

for i in tf_idf:
    tf_idf[i] *= alpha

# TF-IDF matching score ranking
def matching_score(k, query):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("Matching Score\nTop 10 results")
    #print("\nQuery:", query)
    #print("")
    #print(tokens)
    
    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")
    
    l = []
    
    for i in query_weights[:10]:
        l.append(i[0])
    
    print(l)
    return l

# ------------------------------------------------------------------------------------
# testing

for d in dataset:
    print(d)
print('\n')

# update test_file_path to test a new file
test_file_path = os.getcwd()+'/downloads'+'/abc.txt'
f = open(test_file_path, 'r', encoding='utf8', errors='ignore')
query = f.read().strip()
f.close()

l = matching_score(2, query)
print('\n')

print_doc(l[0])

# ------------------------------------------------------------------------------------
# References
# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

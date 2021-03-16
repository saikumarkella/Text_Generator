### Here we will do teh data preparation 
### we will preprocess the dataset and reshape the dataset that has to be feasible 

## importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

## For this example i am not going to do any preprocessing
### reading the dataset

def Preprocess(corpus):
    corpus = corpus.lower()
    return corpus


def Creating_vocabulary(corpus):
    chars = list(set(corpus))
    char_int = dict((c,i) for i,c in enumerate(chars))
    f = open("/home/sai/Documents/new ml/Text Generator/output/char_int.json","w")
    json_string = json.dump(char_int,f)
    return char_int


def Train_data_preparation(char_int,corpus,seq_length):
    x=[]
    y=[]
    length=len(corpus)
    len_vocab=len(char_int)
    for i in tqdm(range(0,length-seq_length)):
        inp=corpus[i:i+seq_length]
        out=corpus[i+seq_length]
        x.append([char_int[c] for c in inp])
        ## one hot encoding for the y
        a=np.zeros(len_vocab)
        a[char_int[out]]=1
        y.append(a)
    x = np.array(x)
    y = np.array(y)
    x=x.reshape(x.shape[0],x.shape[1],1)
    x=x/len(char_int)
    return x,y

def Base():
    file_path="/home/sai/Documents/new ml/Text Generator/input/Alice_wonder_land.txt"
    fp = open(file_path,'r')
    data = fp.read()
    seq_length = 100
    corpus = Preprocess(data)
    char_int = Creating_vocabulary(corpus)
    x,y = Train_data_preparation(char_int,corpus,seq_length)
    return x,y
    
        
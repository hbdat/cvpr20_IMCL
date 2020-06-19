# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:45:40 2019

@author: badat
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)

import wikipedia
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import MWETokenizer
import multiprocessing as mp
import pdb
import pickle
#%%
k = 5
#%%
def load_1k_name():
    path = '/home/project_amadeus/mnt/raptor/hbdat/data/MSCOCO_1k/meta/vocab_coco.pkl' #'./data/MSCOCO/vocab_coco.pkl'#
    with open(path,'rb') as f:
        vocab = pickle.load(f)
    return vocab['words'],vocab['poss']
classes,poss = load_1k_name()
classes=np.array(classes)
poss=np.array(poss)
n_classes = len(classes)
#%%
def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words]
    return words
def get_encoded_word(word):
    return tokenizer.tokenize(tokenize(word))[0]

def get_top_relation(keyword):
    content=wikipedia.summary(keyword)#wikipedia.page(name).content#
    words=tokenizer.tokenize(tokenize(content))
    words = [word for word in words if (word in classes_NN)]
    fdist = nltk.FreqDist(words)
    top_relation = []
    active_label = np.zeros(n_classes)
    active_label[classes_encode.index(get_encoded_word(keyword))]=1
    for word, frequency in fdist.most_common(k):
        top_relation.append(word)
        active_label[classes_encode.index(word)]=frequency
        print(word+' '+str(frequency),end='|')
    print()
    return top_relation,active_label
#%%
tokenizer = MWETokenizer()
print('create tokenizer')
for idx_c,clss in enumerate(classes):
    words=tokenize(clss)
#    print(words)
    tokenizer.add_mwe(words)
print('Done')
#%%
classes_encode = [get_encoded_word(name) for name in classes]
classes_NN = classes[np.array(poss)=='NN']
#%%
df_summary = pd.DataFrame()
label_graph = np.eye(n_classes)

#pool = mp.Pool(processes=4)
#
#results = [pool.apply_async(get_top_relation, args=(name,)) for name in classes]
#output = [p.get() for p in results]
#print(output)
for idx,name in enumerate(classes):#num_class
    
    print('-'*50)
    print(name)
    if poss[idx] == 'NN':
        try:
            summary=wikipedia.summary(name)
            top_relation,active_label=get_top_relation(name)
            label_graph[:,idx]=active_label
            df_summary=df_summary.append({'class':name,'summary':summary,'relation':top_relation},ignore_index = True)
        except Exception as e:
            df_summary=df_summary.append({'class':name,'summary':'error','relation':''},ignore_index = True)
            pass
#        if idx % 100 == 0:
#            df_summary.to_csv('./label_graph/relation.csv',sep='\t', encoding='utf-8')
    else:
        df_summary=df_summary.append({'class':name,'summary':'not noun','relation':name},ignore_index = True)
#%%
df_summary.to_csv('./label_graph/relation_MSCOCO_k_{}.csv'.format(k),sep='\t', encoding='utf-8')
np.save('./label_graph/graph_label_wiki_MSCOCO_k_{}.npz'.format(k),label_graph)
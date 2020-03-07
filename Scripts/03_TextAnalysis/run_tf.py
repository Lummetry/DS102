# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:30:37 2019

@author: damia
"""

import os
import re
import numpy as np


from model_tf import get_model

import tensorflow as tf



def load_docs(folder, splitter_func, max_files=2000):
  files = os.listdir(folder)
  lst_out = []
  max_len = min(max_files, len(files))
  for i,_fn in enumerate(files):
    if i > max_len:
      break
    print("\rProcessing {} {:.1f}%".format(folder, i/max_len * 100), end='', flush=True)
    fn = os.path.join(folder, _fn)
    with open(fn,'rt', encoding="utf8") as f:
      _str = f.read()
      _splitted = splitter_func(_str)
      lst_out.append(_splitted)
  print("")
  return lst_out
    

def get_vocab(lst_train_docs, lst_test_docs):
  set_train_vocab = set()
  set_test_vocab = set()
  for i,doc in enumerate(lst_train_docs):
    print("\rProcessing train VOCAB {:.1f}%".format(i/len(lst_train_docs) * 100), end='', flush=True)
    set_train_vocab.update(set(doc))
  for i,doc in enumerate(lst_test_docs):
    print("\rProcessing train VOCAB {:.1f}%".format(i/len(lst_test_docs) * 100), end='', flush=True)
    set_test_vocab.update(set(doc))
  print("\rTrain vocab: {}\t\t\t\t".format(len(set_train_vocab)))
  print("Test vocab: {}".format(len(set_test_vocab)))
  print("Test words not in train: {}".format((len(set_test_vocab-set_train_vocab))))
  set_vocab = set()
  set_vocab.update(set_train_vocab)
  set_vocab.update(set_test_vocab)
  vocab = {k:i+2 for i,k in enumerate(set_vocab)} # 0 is PAD and 1 is UNK
  reverse = {i:k for k,i in vocab.items()}
  print("Total vocab size: {}".format(len(vocab)))
  return vocab, reverse
  

def tokenize(doc, vocab, splitter_func):
  if type(doc) == str:
    words = splitter_func(doc)
  else:
    words = doc
  return [vocab[x] if x in vocab else 1 for x in words]


def wordize(tokens, reverse_vocab):
  return " ".join([reverse_vocab[x] for x in tokens if x not in [0,1]])


def simple_splitter(text):
  words = re.split("\W+", text)
  return words


def get_dataset(lst_docs, vocab, doc_size=500):
  lst_tokens = []
  for doc in lst_docs:
    tokens = tokenize(doc, vocab, simple_splitter)
    if len(tokens) < doc_size:
      tokens += [0] * (doc_size - len(tokens))
    tokens = tokens[:doc_size]
    lst_tokens.append(tokens)
  return np.array(lst_tokens)


def load_datasets(neg_folder, pos_folder, n_files_each):
  lst_neg_docs = load_docs(neg_folder, splitter_func=simple_splitter, max_files=n_files_each)
  lst_pos_docs = load_docs(pos_folder, splitter_func=simple_splitter, max_files=n_files_each)
  labels = [0] * len(lst_neg_docs) + [1] * len(lst_pos_docs)
  lst_all_docs = lst_neg_docs.copy() + lst_pos_docs.copy()
  del lst_neg_docs
  del lst_pos_docs
  return lst_all_docs, labels
    

if __name__ == '__main__':
  tf.set_random_seed(1234)
  neg_folder = 'data/neg'  
  pos_folder = 'data/pos'
  neg_test = 'data/test/neg'
  pos_test = 'data/test/pos'
  n_files_each_train = 500
  n_files_each_test = 100
  max_doc_size = 500
  epochs = 10
  bsize = 32
  
  lst_train_docs, train_labels = load_datasets(neg_folder, pos_folder, n_files_each=n_files_each_train)
  lst_test_docs, test_labels = load_datasets(neg_test, pos_test, n_files_each=n_files_each_test)
  
  
  d_voc, d_rev_voc = get_vocab(lst_train_docs, lst_test_docs)
  
  if False: # sanity check
    t = tokenize("This is a test nla nla blah!!!!", d_voc, simple_splitter)
    text = wordize(t, reverse_vocab=d_rev_voc)
    print(t)
    print(text)
  
  x_train = get_dataset(lst_train_docs, d_voc, doc_size=max_doc_size)
  y_train = np.array(train_labels).reshape(-1,1)
  
  x_test = get_dataset(lst_test_docs, d_voc, doc_size=max_doc_size)
  y_test = np.array(test_labels).reshape(-1,1)

  if False: # sanity check  
    print(lst_train_docs[101])
    print(wordize(x_train[101], d_rev_voc))
  
  
  m = get_model(vocab_size=len(d_voc)+2,
               embed_size=64, 
               n_lstm_units=128, 
               n_lstm_layers=2, 
               pre_readout=False,
               is_bidi=True,
               drop_rate=0.7,
               masking=False                
                )

  m.fit(x=x_train, y=y_train, 
        batch_size=32, 
        epochs=epochs,
        validation_data=(x_test,y_test))    
  
  
  
  
  
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:15:40 2019

@author: Andrei Damian
"""

import os
import re
import numpy as np
from time import time
from collections import OrderedDict
import pandas as pd
import torch as th

from model_th import DocumentClassifier, train_epoch, batched_test




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
  lst_lens = []
  for doc in lst_docs:
    tokens = tokenize(doc, vocab, simple_splitter)
    lst_lens.append(min(doc_size,len(tokens)))
    if len(tokens) < doc_size:
      tokens += [0] * (doc_size - len(tokens))
    tokens = tokens[:doc_size]
    lst_tokens.append(tokens)
  return np.array(lst_tokens), np.array(lst_lens).reshape(-1,1)


def load_datasets(neg_folder, pos_folder, n_files_each):
  lst_neg_docs = load_docs(neg_folder, splitter_func=simple_splitter, max_files=n_files_each)
  lst_pos_docs = load_docs(pos_folder, splitter_func=simple_splitter, max_files=n_files_each)
  labels = [0] * len(lst_neg_docs) + [1] * len(lst_pos_docs)
  lst_all_docs = lst_neg_docs.copy() + lst_pos_docs.copy()
  del lst_neg_docs
  del lst_pos_docs
  return lst_all_docs, labels
  


#####

    

if __name__ == '__main__':
  th.manual_seed(1234)
  th.cuda.manual_seed(1234)
  
  neg_folder = 'data/neg'  
  pos_folder = 'data/pos'
  neg_test = 'data/test/neg'
  pos_test = 'data/test/pos'
  n_files_each_train = 500
  n_files_each_test = 100
  max_doc_size = 500
  batch_size=32
  epochs = 15
  direct_GPU_train_val = False
  
  lst_train_docs, train_labels = load_datasets(neg_folder, pos_folder, n_files_each=n_files_each_train)
  lst_test_docs, test_labels = load_datasets(neg_test, pos_test, n_files_each=n_files_each_test)
  
  
  d_voc, d_rev_voc = get_vocab(lst_train_docs, lst_test_docs)
  
  if False: # sanity check
    t = tokenize("This is a test nla nla blah!!!!", d_voc, simple_splitter)
    text = wordize(t, reverse_vocab=d_rev_voc)
    print(t)
    print(text)
  
  x_train_sent, x_train_lens = get_dataset(lst_train_docs, d_voc, doc_size=max_doc_size)
  y_train = np.array(train_labels).reshape(-1,1)
  
  x_test_sent, x_test_lens = get_dataset(lst_test_docs, d_voc, doc_size=max_doc_size)
  y_test = np.array(test_labels).reshape(-1,1)

  if False: # sanity check  
    print(lst_train_docs[101])
    print(wordize(x_train_sent[101], d_rev_voc))
  
  th_x_train_sent = th.LongTensor(x_train_sent) 
  th_x_train_lens = th.LongTensor(x_train_lens)
  th_y_train = th.Tensor(y_train)    
  
  th_x_test_sent = th.LongTensor(x_test_sent)
  th_x_test_lens = th.LongTensor(x_test_lens)
  th_y_test = th.Tensor(y_test)

  if direct_GPU_train_val:
    th_x_train_sent = th_x_train_sent.cuda()
    th_x_train_lens = th_x_train_lens.cuda()
    th_y_train = th_y_train.cuda()
    th_x_test_sent = th_x_test_sent.cuda()
    th_x_test_lens = th_x_test_lens.cuda()
    th_y_test = th_y_test.cuda()
    th_train_loader = (th_x_train_sent, th_x_train_lens, th_y_train)
    th_test_loader = (th_x_test_sent, th_x_test_lens, th_y_test)
  else:
    th_dataset = th.utils.data.TensorDataset(th_x_train_sent, th_x_train_lens, th_y_train)
    th_train_loader = th.utils.data.DataLoader(dataset=th_dataset, 
                                               batch_size=32,
                                               shuffle=True,
                                               )
    
    th_test_dataset = th.utils.data.TensorDataset(th_x_test_sent, th_x_test_lens, th_y_test)
    th_test_loader = th.utils.data.DataLoader(dataset=th_test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              )
  
  params = [      
  {"drop_rate":0.0,"is_bidi":False,"n_lstm_layers":1,"n_lstm_units":128, "last_layer": True,"masking":False,"pre_read":False},
  {"drop_rate":0.7,"is_bidi":False,"n_lstm_layers":1,"n_lstm_units":128, "last_layer": True,"masking":False,"pre_read":False},
  {"drop_rate":0.7,"is_bidi": True,"n_lstm_layers":1,"n_lstm_units":128, "last_layer": True,"masking":False,"pre_read":False},
  {"drop_rate":0.7,"is_bidi": True,"n_lstm_layers":2,"n_lstm_units":128, "last_layer": True,"masking":False,"pre_read":False},
  {"drop_rate":0.7,"is_bidi": True,"n_lstm_layers":2,"n_lstm_units":128, "last_layer": True,"masking": True,"pre_read":False},
  {"drop_rate":0.7,"is_bidi": True,"n_lstm_layers":2,"n_lstm_units":256, "last_layer": True,"masking": True,"pre_read":False},
  {"drop_rate":0.9,"is_bidi": True,"n_lstm_layers":2,"n_lstm_units":256, "last_layer": True,"masking": True,"pre_read":False},
  {"drop_rate":0.9,"is_bidi": True,"n_lstm_layers":2,"n_lstm_units":256, "last_layer": True,"masking": True,"pre_read": True},
  ]
  
  dct_res = OrderedDict({'Val':[], 'Ep':[]})
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)
  for k in params[0]:
    dct_res[k] = []
  for model_no,param in enumerate(params):
    print("Training/validating model {}...".format(model_no+1))
    for k in param:
      dct_res[k].append(param[k])

    th.manual_seed(1234)
    th.cuda.manual_seed(1234)

    clf = DocumentClassifier(vocab_size=len(d_voc)+2,
                             embed_size=128,
                             **param)
    clf = clf.cuda()
    opt = th.optim.Adam(params=clf.parameters())
    lossfn = th.nn.BCEWithLogitsLoss()
    
    best_val = 0
    best_ep = 0
    for epoch in range(epochs):
      batch_losses = []
      print("Training epoch {} ...".format(epoch+1))
      t0 = time()
      loss = train_epoch(clf, th_train_loader,  
                         batch_size=batch_size,
                         optimizer=opt, loss_func=lossfn,
                         direct_gpu=direct_GPU_train_val)
      t_epoch = time() - t0
      print("\rEpoch {} loss {:.4f} - {:.1f}s elapsed".format(epoch+1, 
            loss, t_epoch))
      t1 = time()
      val = batched_test(clf, th_test_loader) * 100
      trn = batched_test(clf, th_train_loader) * 100
      t_test = time() - t1
      print("\r  Train/val acc: {:.1f}% / {:.1f}% - {:.1f}s elapsed".format(
          trn, val, t_test))
      if best_val < val:
        best_val = val
        best_ep = epoch + 1
    dct_res['Val'].append(best_val)
    dct_res['Ep'].append(best_ep)
    df = pd.DataFrame(dct_res).sort_values('Val')
    print(df)
    
    
    
    
    
  
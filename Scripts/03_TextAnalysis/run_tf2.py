# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:30:37 2019

@author: damia
"""

import os
import re
import numpy as np
import pickle


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
    
@tf.function
def graph_batch_train(model, tf_x, tf_y, loss_func, opt):
  with tf.GradientTape() as tape:
    tf_yhat = model(tf_x, training=True)
    tf_loss = loss_func(tf_y, tf_yhat)
  lst_grads = tape.gradient(tf_loss, m.trainable_weights)
  opt.apply_gradients(zip(lst_grads, m.trainable_weights))
  return tf_loss

@tf.function
def compute_acc(model, x, y):
  tf_yh = model(x) #, training=False)
  tf_ypred = tf.cast(tf_yh >= 0.5, tf.int32)
  tf_res = tf.cast(y == tf_ypred, tf.int32)
  tf_acc = tf.reduce_sum(tf_res) / y.shape[0]
  return tf_acc
  

def train_and_validate(model, train_data, test_data, 
                       n_epochs, batch_size,
                       in_graph=False):
  x_train, y_train = train_data
  x_test, y_test = test_data  
  n_obs = x_train.shape[0]
  print("Training on {} obs with {} batch-size for {} epochs".format(
      n_obs, batch_size, n_epochs))

  trn_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
  trn_shuffled_ds = trn_ds.shuffle(buffer_size=n_obs)
  trn_batched_ds = trn_shuffled_ds.batch(batch_size)
  trn_ready_ds = trn_batched_ds.prefetch(1)
  n_batches = n_obs / bsize
  loss_func = tf.keras.losses.get(model.loss)
  opt = tf.keras.optimizers.get(model.optimizer)
  
  t_start = tf.timestamp()
  for epoch in range(epochs):
    t_epoch_start = tf.timestamp()
    epoch_loss = 0.
    updates = 0
    for i, (tf_x_batch, tf_y_batch) in enumerate(trn_ready_ds):
      t_iter_start = tf.timestamp()
      if in_graph:
        tf_loss = graph_batch_train(model=m,
                              tf_x=tf_x_batch,
                              tf_y=tf_y_batch,
                              loss_func=loss_func,
                              opt=opt)
      else:        
        with tf.GradientTape() as tape:
          tf_yhat = m(tf_x_batch)
          tf_loss = loss_func(tf_y_batch, tf_yhat)        
        lst_grads = tape.gradient(tf_loss, model.trainable_weights)
        opt.apply_gradients(zip(lst_grads, model.trainable_weights))
      epoch_loss += tf_loss
      updates += 1
      t_iter_end = tf.timestamp()
      print("\rTraining epoch {} - {:.1f}% - {:.2f} s/itr".format(epoch + 1, 
                    i / n_batches * 100, t_iter_end - t_iter_start),
            flush=True, end='')
          
    t_epoch_end = tf.timestamp()
    print("\rTraining epoch {} done. Mean loss: {:.4f} {:.2f} s/ep".format(
        epoch + 1, epoch_loss / updates, 
        t_epoch_end-t_epoch_start), flush=True)
    print("Testing...", flush=True, end='')
    t1 = tf.timestamp()
    tf_acc_trn = compute_acc(m, x_test, y_test)
    t2 = tf.timestamp()
    print("\rTrain acc (graph): {:.3f}% - {:.1f}s".format(
        tf_acc_trn.numpy() *100, t2-t1), flush=True)
    t1 = tf.timestamp()
    ypred = m.predict(x_test) >= 0.5
    acc = (y_test.ravel() == ypred.ravel()).sum() / y_test.shape[0]
    t2 = tf.timestamp()
    print("Train acc (keras): {:.3f}% - {:.1f}s".format(
        acc*100, t2-t1, flush=True))
  t_end = tf.timestamp()
  print("Total training time {:.1f} s".format(t_end-t_start))
      

@tf.function 
def graph_train_and_validate(model, train_data, test_data, 
                             epochs,
                             batch_size):
  x_trn, y_trn = train_data
  x_tst, y_tst = test_data  
  n_obs = x_trn.shape[0]

  trn_ds = tf.data.Dataset.from_tensor_slices((x_trn,y_trn))
  trn_shuffled_ds = trn_ds.shuffle(buffer_size=n_obs)
  trn_batched_ds = trn_shuffled_ds.batch(batch_size)
  trn_ready_ds = trn_batched_ds.prefetch(1)
  tf.print(epochs," epochs, batch-size ",batch_size," ds: ", trn_ready_ds)
  loss_func = tf.keras.losses.get(model.loss)
  opt = tf.keras.optimizers.get(model.optimizer)
  
  t_start = tf.timestamp()
  for epoch in range(epochs):
    epoch_loss = 0.
    updates = 0
    for i, (tf_x_batch, tf_y_batch) in enumerate(trn_ready_ds):
      with tf.GradientTape() as tape:
        tf_yhat = model(tf_x_batch)
        tf_loss = loss_func(tf_y_batch, tf_yhat)        
      lst_grads = tape.gradient(tf_loss, model.trainable_weights)
      opt.apply_gradients(zip(lst_grads, model.trainable_weights))
      epoch_loss += tf_loss
      updates += 1
  t_end = tf.timestamp()
  return t_end-t_start

if __name__ == '__main__':
  fn_pickle = 'data/pickled.pkl'
  neg_folder = 'data/neg'  
  pos_folder = 'data/pos'
  neg_test = 'data/test/neg'
  pos_test = 'data/test/pos'
  n_files_each_train = 500
  n_files_each_test = 100
  max_doc_size = 500
  epochs = 3
  bsize = 32
  
  
  if os.path.isfile(fn_pickle):
    print("Loading from pickle '{}'".format(fn_pickle), flush=True)
    with open(fn_pickle, 'rb') as f:
      x_train, y_train, x_test, y_test, d_voc, d_rev_voc = pickle.load(f)
  else:
    print("Creating datasets...", flush=True)     
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
    
    data = x_train, y_train, x_test, y_test, d_voc, d_rev_voc
    print("Saving pickle '{}'".format(fn_pickle))
    with open(fn_pickle, "wb") as f:
      pickle.dump(data, f)
    

  if False: # sanity check  
    print(lst_train_docs[101])
    print(wordize(x_train[101], d_rev_voc))
  
  if False:
    print("\nNon graph training")
    tf.random.set_seed(1234)
    _m = get_model(vocab_size=len(d_voc)+2,
                 embed_size=64, 
                 n_lstm_units=128, 
                 n_lstm_layers=2, 
                 pre_readout=False,
                 is_bidi=True,
                 drop_rate=0.7,
                 masking=False                
                  )
    train_and_validate(_m, (x_train,y_train), (x_test,y_test), 
                       epochs=epochs,
                       batch_size=bsize,
                       in_graph=False)
  
  
  if False:
    print("\nUsing GRAPH batched train")
    tf.random.set_seed(1234)
    _m = get_model(vocab_size=len(d_voc)+2,
                 embed_size=64, 
                 n_lstm_units=128, 
                 n_lstm_layers=2, 
                 pre_readout=False,
                 is_bidi=True,
                 drop_rate=0.7,
                 masking=False                
                  )
    train_and_validate(_m, (x_train,y_train), (x_test,y_test), 
                       epochs=epochs, 
                       batch_size=bsize,
                       in_graph=True)
  
  if True:
    tf.random.set_seed(1234)
    _m = get_model(vocab_size=len(d_voc)+2,
                 embed_size=64, 
                 n_lstm_units=128, 
                 n_lstm_layers=2, 
                 pre_readout=False,
                 is_bidi=True,
                 drop_rate=0.7,
                 masking=False                
                  )
    print("\nFULL GRAPH")
    tf_time = graph_train_and_validate(_m, (x_train,y_train), (x_test,y_test), 
                                               batch_size=bsize,
                                               epochs=epochs)
    print("Done training")
    tf_acc_trn = compute_acc(_m, x_test, y_test)
    print("Accuracy: {:.1f} - time {:.1f} s".format(
        tf_acc_trn.numpy(), tf_time.numpy()))
    
  
  

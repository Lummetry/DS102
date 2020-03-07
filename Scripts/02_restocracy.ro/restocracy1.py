# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:35:17 2019


"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import re
from typing import List
import tensorflow as tf

def corpus_to_words(lst_rev):
  lst_result = []
  for review in lst_rev:
    words = re.split('\W+', review)
    words = [word for word in words if word != '']
    lst_result.append(words)
  return lst_result

def get_vocab(lst_words: List[List]):
  
  dct_vocab = {}
  i = 2
  dct_vocab['<unk>'] = 1
  for review in lst_words:
    for word in review:
      if word not in dct_vocab:
        dct_vocab[word] = i
        i = i + 1
  
  dct_rev_vocab = {idx: word for word, idx in dct_vocab.items()}
  
  
  return dct_vocab, dct_rev_vocab
  
def tokenize(lst_words, dct_vocab, fix_len = None):
  tokens = []
  for review in lst_words:
    review_tokens = []
    for word in review:
      review_tokens.append(word_to_token(word, dct_vocab))
    if fix_len:
      #review_tokens = review_tokens[:fix_len]
      len_review = len(review_tokens)
      review_tokens = review_tokens + [0] * abs(fix_len - len_review) 
    tokens.append(review_tokens[:fix_len])
    
  return tokens

def wordize(sent_tokens, dct_rev_vocab):
  lst_words = []
  for token in sent_tokens:
    if token == 0:
      break
    lst_words.append(token_to_word(token, dct_rev_vocab))
  return " ".join(lst_words)

def word_to_token(word, dct_vocab)  :
  if word in dct_vocab:
    return dct_vocab[word]
  return dct_vocab['<unk>']

def token_to_word(token, dct_rev_vocab):
  return dct_rev_vocab.get(token, dct_rev_vocab[1])


def show_results(model, x, gold):  
  preds = model.predict(x).ravel()
  gold = gold.ravel()
  df = pd.DataFrame({"Predictions": preds, "Truth" : gold, "MAE" : np.abs(preds-gold)})
  print("\n\nResults for model '{}':\n{}".format(model.name,df))
  return

if __name__ == '__main__':
  with open('all_data.pk', 'rb') as file:
    data = pk.load(file)
  lst_prices = []
  lst_reviews = []

  for dct_item in data:
    price = dct_item['price']
    review = dct_item['review']
    
    lst_prices.append(price)
    lst_reviews.append(review)
    
  np_prices = np.array([int(x.split(' ' )[0]) 
                              for x in lst_prices])
  splitted_corpus = corpus_to_words(lst_reviews)
  dct_vocab, dct_rev_vocab = get_vocab(splitted_corpus)
  token_reviews = tokenize(splitted_corpus, 
                           dct_vocab, fix_len=1000)
  
  t = "Ana are mere bune la restaurantul ei\nsi negaunoase "
  splitted_test = corpus_to_words([t])
  token_test = tokenize(splitted_test, dct_vocab)
  print(token_test)
  rebuilt_test = wordize(token_test[0], dct_rev_vocab)
  print(rebuilt_test)
  
  np_reviews = np.array(token_reviews)
  
  np_prices_train = np_prices[:-20]
  np_reviews_train = np_reviews[:-20]
  
  np_prices_test = np_prices[-20:]
  np_reviews_test = np_reviews[-20:]
  
  n_epochs = 100
  
  plt.figure()
  plt.hist(np_prices_train, bins=20, label='train prices')
  plt.hist(np_prices_test, label='dev prices')
  plt.title("Train vs dev sets prices")
  plt.legend()
  plt.savefig("fig1.png")
  plt.show()
  
  
  hists = {}
  
  ###
  ### the most simple model based on embeddings
  ###
  
  tf_input = tf.keras.layers.Input((1000,), name="Net_input")
  lyr_projection = tf.keras.layers.Embedding(14952, 
                                             32, 
                                             name="Projection")
  tf_projection = lyr_projection(tf_input)
  
  lyr_flatten = tf.keras.layers.Flatten(name="1d")
  lyr_regression = tf.keras.layers.Dense(1, name="price")
  
  tf_flatten = lyr_flatten(tf_projection)
  tf_output = lyr_regression(tf_flatten)
  
  model = tf.keras.models.Model(inputs=tf_input, outputs=tf_output,
                                name="Model_no_dropout")
  model.compile(optimizer="adam", loss="mae")
  
  hist = model.fit(x=np_reviews_train, y=np_prices_train, epochs=n_epochs, 
                   verbose=0,
                   validation_data=(np_reviews_test, np_prices_test))
    
  hists['train_no_drop_loss'] = hist.history['loss']
  hists['dev_no_drop_loss'] = hist.history['val_loss']
  
  show_results(model, np_reviews_test, np_prices_test)

  
  del model
  tf.keras.backend.clear_session()    

  ###
  ### add dropout 0.5
  ###
  
  tf_input = tf.keras.layers.Input((1000,), name="Net_input")
  lyr_projection = tf.keras.layers.Embedding(14952, 32, name="Projection")
  tf_projection = lyr_projection(tf_input)
  
  lyr_flatten = tf.keras.layers.Flatten(name="1d")
  lyr_dropout = tf.keras.layers.Dropout(rate=0.5)
  lyr_regression = tf.keras.layers.Dense(1, name="price")
  
  tf_flatten = lyr_flatten(tf_projection)
  tf_dropedout = lyr_dropout(tf_flatten)
  tf_output = lyr_regression(tf_dropedout)
  
  model = tf.keras.models.Model(inputs=tf_input, outputs=tf_output,
                                name="Model_50prc_drop")
  model.compile(optimizer="adam", loss="mae")

  hist = model.fit(x=np_reviews_train, y=np_prices_train, epochs=n_epochs, 
                   verbose=0,
                   validation_data=(np_reviews_test, np_prices_test))
  hists['train_drop_50_loss'] = hist.history['loss']
  hists['dev_drop_50_loss'] = hist.history['val_loss']

  show_results(model, np_reviews_test, np_prices_test)
  

  del model
  tf.keras.backend.clear_session()    

  ###
  ### add dropout 0.9
  ###
  
  tf_input = tf.keras.layers.Input((1000,), name="Net_input")
  lyr_projection = tf.keras.layers.Embedding(14952, 32, name="Projection")
  tf_projection = lyr_projection(tf_input)
  
  lyr_flatten = tf.keras.layers.Flatten(name="1d")
  lyr_dropout = tf.keras.layers.Dropout(rate=0.9)
  lyr_regression = tf.keras.layers.Dense(1, name="price")
  
  tf_flatten = lyr_flatten(tf_projection)
  tf_dropedout = lyr_dropout(tf_flatten)
  tf_output = lyr_regression(tf_dropedout)
  
  model = tf.keras.models.Model(inputs=tf_input, outputs=tf_output,
                                name="Model_90prc_drop")
  model.compile(optimizer="adam", loss="mae")

  hist = model.fit(x=np_reviews_train, y=np_prices_train, epochs=n_epochs, 
                   verbose=0,
                   validation_data=(np_reviews_test, np_prices_test))
  hists['train_drop_90_loss'] = hist.history['loss']
  hists['dev_drop_90_loss'] = hist.history['val_loss']
  
  show_results(model, np_reviews_test, np_prices_test)
  

  del model
  tf.keras.backend.clear_session()    


  ###
  ### add dropout 0.9 and relu
  ###
  
  tf_input = tf.keras.layers.Input((1000,), name="Net_input")
  lyr_projection = tf.keras.layers.Embedding(14952, 32, name="Projection")
  tf_projection = lyr_projection(tf_input)
  
  lyr_flatten = tf.keras.layers.Flatten(name="1d")
  lyr_flatten_act = tf.keras.layers.Activation('relu')
  lyr_dropout = tf.keras.layers.Dropout(rate=0.9)
  lyr_regression = tf.keras.layers.Dense(1, name="price")
  
  tf_flatten = lyr_flatten(tf_projection)
  tf_flatten_act = lyr_flatten_act(tf_flatten)
  tf_dropedout = lyr_dropout(tf_flatten_act)
  tf_output = lyr_regression(tf_dropedout)
  
  model = tf.keras.models.Model(inputs=tf_input, outputs=tf_output,
                                name="Model_90prc_drop")
  model.compile(optimizer="adam", loss="mae")

  hist = model.fit(x=np_reviews_train, y=np_prices_train, epochs=n_epochs, 
                   verbose=0,
                   validation_data=(np_reviews_test, np_prices_test))
  hists['train_drop_90_relu_loss'] = hist.history['loss']
  hists['dev_drop_90_relu_loss'] = hist.history['val_loss']
  
  show_results(model, np_reviews_test, np_prices_test)
  

  del model
  tf.keras.backend.clear_session()    


  ###
  ### add dropout 0.9 and relu and conv
  ###
  
  tf_input = tf.keras.layers.Input((1000,), name="Net_input")
  lyr_projection = tf.keras.layers.Embedding(14952, 32, name="Projection")
  lyr_analysis = tf.keras.layers.Conv1D(16, kernel_size=3, strides=3)
  tf_projection = lyr_projection(tf_input)
  
  lyr_flatten = tf.keras.layers.Flatten(name="1d")
  lyr_flatten_act = tf.keras.layers.Activation('relu')
  lyr_dropout = tf.keras.layers.Dropout(rate=0.9)
  lyr_regression = tf.keras.layers.Dense(1, name="price")
  
  tf_analysis = lyr_analysis(tf_projection)
  tf_flatten = lyr_flatten(tf_analysis)
  tf_flatten_act = lyr_flatten_act(tf_flatten)
  tf_dropedout = lyr_dropout(tf_flatten_act)
  tf_output = lyr_regression(tf_dropedout)
  
  model = tf.keras.models.Model(inputs=tf_input, outputs=tf_output,
                                name="Model_90prc_drop")
  model.compile(optimizer="adam", loss="mae")

  model.summary()
  
  hist = model.fit(x=np_reviews_train, y=np_prices_train, epochs=n_epochs, 
                   verbose=0,
                   validation_data=(np_reviews_test, np_prices_test))
  hists['train_drop_90_relu_c_loss'] = hist.history['loss']
  hists['dev_drop_90_relu_c_loss'] = hist.history['val_loss']
  
  show_results(model, np_reviews_test, np_prices_test)
  

  del model
  tf.keras.backend.clear_session()    


  plt.figure(figsize=(20,15))
  for hkey in hists:
    plt.plot(hists[hkey], label=hkey)
  plt.legend(loc="center right")
  plt.title("Train vs dev history for all models")
  plt.savefig('fig2.png')
  plt.show()
  
  
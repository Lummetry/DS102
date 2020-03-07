# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:34:02 2019

@author: damia
"""

import tensorflow as tf

def get_model(vocab_size, 
              embed_size=64, 
              n_lstm_units=128,
              n_lstm_layers=3,
              pre_readout=False,
              is_bidi=True,
              drop_rate=0.5,
              masking=False,
              ):
  tf_inp = tf.keras.layers.Input((None,))
  tf_x = tf.keras.layers.Embedding(vocab_size, embed_size)(tf_inp)
  for i in range(n_lstm_layers-1):
    layer = tf.keras.layers.LSTM(n_lstm_units, return_sequences=True)
    if is_bidi:
      layer = tf.keras.layers.Bidirectional(layer)
    tf_x = layer(tf_x)
  
  layer = tf.keras.layers.LSTM(n_lstm_units)
  if is_bidi:
    layer = tf.keras.layers.Bidirectional(layer)
  tf_x = layer(tf_x)
  if pre_readout:
    tf_x = tf.keras.layers.Dense(n_lstm_units//2, activation='relu')(tf_x)
  tf_x = tf.keras.layers.Dropout(drop_rate)(tf_x)
  tf_out = tf.keras.layers.Dense(1, activation='sigmoid')(tf_x)
  model = tf.keras.models.Model(tf_inp, tf_out)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  return model
  
  
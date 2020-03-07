# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:44:53 2019

@author: damia
"""
import numpy as np
import os
from PIL import Image
import tensorflow as tf




if __name__ == '__main__':
  fn_all = 'data/all_images.npz'
  shapes = set()
  img_folder = 'data'
  np_images = np.load(fn_all, allow_pickle=True)['a']
  print("images loaded", flush=True)
  
  # transfer learning part
  shape=(384,683,3)
  
  m = tf.keras.applications.InceptionResNetV2(input_shape=shape,
                                               include_top=False)
  print("model loaded", flush=True)
  lyr_max = tf.keras.layers.GlobalMaxPool2D() 

  tf_inp = tf.keras.layers.Input(shape)
  tf_x = m(tf_inp)
  tf_out = lyr_max(tf_x)
  
  model = tf.keras.models.Model(tf_inp, tf_out)

  # compute 5-30 clusterings
  print("inferencing ...", flush=True)
  np_img_feats = model.predict(np_images, batch_size=6, verbose=1)
  
  # find optimal nr of clusters
  
  from sklearn.cluster import KMeans
  
  klfits = []
  for cl in range(5,30,3):
    print("\r Clustering {}".format(cl), end='', flush=True)
    klfits.append(KMeans(n_clusters=cl).fit(np_img_feats))
  
  # explore results
  
  
   
      
      
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:44:53 2019

@author: damia
"""
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
  in_folder = '.....'
  fn_all = 'all_images.npz'
  out_folder = 'thumbs'
  data_folder = 'data'
  files = os.listdir(in_folder)
  out_data = os.path.join(data_folder,fn_all)
  images = []
  for i,_fn in enumerate(files):
    print("\rProcessing {:.1f}%".format((i+1)/len(files)*100), end='',flush=True)
    if '.jpg' in _fn or '.png' in _fn:
      fn = os.path.join(in_folder, _fn)
      fn_thumb = os.path.join(out_folder, _fn)
      img = Image.open(fn)
      img = img.resize(size=(683, 384))
      img.save(fn_thumb)
      images.append(np.array(img))
  
  np_images = np.array(images)
  np.savez_compressed(out_data, a=np_images)
  plt.imshow(np_images[np.random.randint(np_images.shape[0])])
  plt.show()

  
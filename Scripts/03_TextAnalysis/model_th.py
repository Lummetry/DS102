# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:16:01 2019

@author: damia
"""

import torch as th
import numpy as np


class DocumentClassifier(th.nn.Module):
  
  def __init__(self, vocab_size, 
               embed_size=64, 
               n_lstm_units=128, 
               n_lstm_layers=3, 
               pre_read=False,
               is_bidi=True,
               drop_rate=0.5,
               last_layer=True,
               masking=False,
               padding_idx=0):
    super().__init__()
    self.pre_read = pre_read
    self.last_layer = last_layer
    self.masking = masking
    self.embed_layer = th.nn.Embedding(num_embeddings=vocab_size, 
                                       embedding_dim=embed_size,
                                       padding_idx=padding_idx)
    self.num_layers = n_lstm_layers
    self.num_directions = int(is_bidi) + 1
    self.liner_pre_readout =  n_lstm_units * (1 + int(is_bidi)) // 2
    self.n_lstm_units = n_lstm_units
    self.lstm_layer = th.nn.LSTM(input_size=embed_size, 
                                 hidden_size=n_lstm_units, 
                                 num_layers=n_lstm_layers,
                                 bidirectional=is_bidi)
    self.n_rnn_out = n_lstm_units * self.num_directions * (1 if last_layer else n_lstm_layers )
    self.drop = th.nn.Dropout(p=drop_rate)
    if self.pre_read:
      self.linear = th.nn.Linear(self.n_rnn_out, self.liner_pre_readout)
      self.relu = th.nn.ReLU()
      self.drop2 = th.nn.Dropout(p=drop_rate)
    if self.pre_read:
      self.readout = th.nn.Linear(self.liner_pre_readout, 1, bias=False)
    else:
      self.readout =  th.nn.Linear(self.n_rnn_out, 1)
    self.proba = th.nn.Sigmoid()
    print("Model inited with masking={}:\n{}".format(masking, self))
    return
  
  
  def forward(self, inputs, lens, proba=False, is_sorted=False):
    batch_size = inputs.size(0)
    x_embeds = self.embed_layer(inputs)
    x_embeds = x_embeds.permute(1,0,2)
    if self.masking:
      lens = lens.view(-1)
      if is_sorted:
        srt_order = lens.argsort(descending=True)
        x_embeds = x_embeds[srt_order]
        lens = lens[srt_order]
      x_embeds_pack = th.nn.utils.rnn.pack_padded_sequence(x_embeds, 
                                                           lens.view(-1),
                                                           enforce_sorted=is_sorted)
    else:
      x_embeds_pack = x_embeds
    x_seq_pack, (x_h, x_c) = self.lstm_layer(x_embeds_pack)    
    if self.masking:
      x_seq, out_lens = th.nn.utils.rnn.pad_packed_sequence(x_seq_pack)
      # assert  (out_lens == lens).all()
    else:
      x_seq = x_seq_pack
    x_seq1 = x_seq[0,:,self.n_lstm_units:] # backward final step (0)
    x_seq2 = x_seq[-1,:,:self.n_lstm_units] # forward final step (T)
    x_last = th.cat((x_seq2, x_seq1), 1) # order is F,B
    x_post_lstm = x_h.permute(1,0,2)  # BATCH * LAYERS * FEATURES
    if self.last_layer:
      x_post_lstm_ext = x_post_lstm.view(batch_size, 
                                         self.num_layers, 
                                         self.num_directions, 
                                         self.n_lstm_units)
      x_post_lstm2 = x_post_lstm_ext[:,-1,:,:]    
      x = x_post_lstm2.reshape(batch_size,-1)
      # check all backwards (no matter must match)
      # assert (x_seq1 == x[:,self.n_lstm_units:]).all()
      # for forward it dependes where the h, c is calculated in the sequence 
    else:
      x = x_post_lstm.reshape(batch_size,-1)
    x = self.drop(x)
    if self.pre_read:
      x = self.drop2(self.relu(self.linear(x)))
    x_logits = self.readout(x)
    if proba:
      return self.proba(x_logits)
    else:
      return x_logits
    

def batched_test(model, val_data, verbose=False):
  model.eval()
  nr_obs = val_data.dataset.tensors[0].shape[0]
  accs = []
  for i, (x_val_sent, x_val_lens, y_val) in enumerate(val_data):
    if verbose: 
      print("\rValidation {:.1f}% computed".format((i+1)/len(val_data)*100),end='', flush=True)
    with th.no_grad():
      y_hat = model(x_val_sent.cuda(),x_val_lens.cuda(), proba=True).cpu()
      y_pred = (y_hat > 0.5).float().numpy().ravel()
      np_y = y_val.numpy().ravel()
      acc = (y_pred == np_y).sum() 
      accs.append(acc)
  
  return np.sum(accs) / nr_obs


def full_test(model, th_x_sent, th_x_lens, np_y):
  model.eval()
  with th.no_grad():
    yh = model(th_x_sent.cuda(),th_x_lens.cuda(), proba=True)
    y_pred = (yh > 0.5).float().cpu().numpy().ravel()
    acc = (y_pred == np_y.ravel()).sum() / y_pred.shape[0]
  return acc
    
def train_epoch(model, train_data_loader, batch_size,
                optimizer, loss_func, direct_gpu=False):
  if direct_gpu:
    x_data_sent, x_data_lens, y_data = train_data_loader
    n_obs = x_data_sent.shape[0]
    n_iter = n_obs // batch_size + 1
  else:
    n_iter = train_data_loader.dataset.tensors[0].shape[0]    
  losses = th.zeros(n_iter).cuda()
  if direct_gpu:
    for i in range(n_iter):
      print("\rCustom training {:.1f}%".format(i/n_iter*100),
            end='',flush=True)
      b_start = i * batch_size
      b_end = max((i + 1) * batch_size, n_obs)
      x_batch_sent = x_data_sent[b_start:b_end]
      x_batch_lens = x_data_lens[b_start:b_end]
      y_batch = y_data[b_start:b_end]
      batch_loss = train_model(model, (x_batch_sent,x_batch_lens, y_batch), optimizer, loss_func)
      losses[i] = batch_loss
  else:
    for i,train_batch in enumerate(train_data_loader):
      print("\rLoader based training {:.1f}%".format(i/len(train_data_loader)*100),
            end='',flush=True)
      cuda_batch = []
      for train_b in train_batch:
        cuda_batch.append(train_b.cuda())
      batch_loss = train_model(model, cuda_batch, optimizer, loss_func)
      losses[i] = batch_loss
  loss = losses.mean().cpu().item()
  return loss
      
      


def train_model(model, train_data, optimizer, loss_func):
  model.train()
  x_batch_sent, x_batch_lens, y_batch = train_data
  optimizer.zero_grad()
  y_hat = model(x_batch_sent, x_batch_lens)
  th_loss = loss_func(y_hat, y_batch)
  th_loss.backward()
  optimizer.step()
  return th_loss.detach()
    
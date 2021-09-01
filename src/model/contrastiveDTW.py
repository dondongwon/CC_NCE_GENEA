import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb
import copy

import torch
import torch.nn as nn
from transformers import BertModel
import logging
import random
from .layers import *


from soft_dtw_cuda import SoftDTW, SoftDTW_DBS

class MoCo_GlobalDTW(MoCo):
  def __init__(self, enc, self_sup, in_channels=128, time_steps=64, K=512, m=0.99, T=0.1, bn_splits=1, margin = 0, symmetric=False, **kwargs):
    super().__init__(enc, in_channels, time_steps, K, m, T, bn_splits, symmetric, **kwargs)
    self.softdtw = SoftDTW(use_cuda = True)
    self.clusterdict = dict()
    self.clusterdict_q = dict()
    self.clusterdict_labels = dict()
    self.clusterdict_intervals = dict()

    self.margin = margin
    self.select_idx = None
    self.encoder_q = enc
    self.batch_cluster_count = 0
    self.init_epochs = 2

    #global mean
    self.DTW_true = bool(kwargs['DTW'])
    self.kwargs = kwargs
    self.running_mean_list = []
    self.running_mean_counts = 0
    self.running_mean = None
    self.running_sd = None

    #global dict
    self.global_dict = {}
    self.global_dict_q = {}
    self.global_dict_intervals = {}
    self.cluster_count = 0

    self.self_sup = self_sup

  def correct_sample(self, im1):
    with torch.no_grad():
      q = self.encoder_q(im1)
      q = nn.functional.normalize(q, dim=1)  # already normalized

    return q, [torch.zeros(1)[0].to(im1.device)]


    #save cluster dict locally for gradient saving
    

  def bootstrap_cluster_DTW(self, q, y, interval_ids, labels, iter, top_k = 5, bootstrap = True, **kwargs): #anchor: 1 X T X E
    with torch.no_grad():
      # q = self.encoder_q(q.permute(0,2,1))  # queries: NxCxT
      # q = nn.functional.normalize(q, dim=1)  # already normalized
      # q = q.permute(0,2,1)
      if self.self_sup == False:
        q_in = y
      
      if self.self_sup == True:
        q = self.encoder_q(q.permute(0,2,1))
        q = nn.functional.normalize(q, dim=1)
        q = q.permute(0,2,1)
        q_in = q


      if self.select_idx == None:
          sample_indices = torch.nonzero(labels)
          select_idx = torch.randperm(len(sample_indices))[:1]
          anchor = q_in[select_idx]
      else:
          select_idx = self.select_idx[0]
          anchor = q_in[select_idx]

      labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
      sample_indices = torch.nonzero(labels) #update sample_indices to be used later
      remain_size = sample_indices.shape[0]

      if self.DTW_true:
          anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
          sim_scores = self.softdtw(anchor,q_in[sample_indices[:,0]].clone()) # calculate sim scores via dtw

      if self.DTW_true == False:
          anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
          sim_scores = torch.einsum('nct,nct->n', [anchor, q_in[sample_indices[:,0]].clone()])

      scores_size = sim_scores.shape[0]
      if iter == 0:
        mean = torch.mean(sim_scores)
        self.running_mean_list.append(mean)
        self.running_mean_counts += 1

      #Get Mean SD estimates (pt 1)
      if kwargs['epoch'] == self.init_epochs:
        return torch.zeros_like(labels)

      #Get Mean SD estimates (pt 2)
      if kwargs['epoch'] >= self.init_epochs and self.running_mean == None:
        self.running_mean = torch.stack(self.running_mean_list)
        self.running_sd = torch.std(self.running_mean)
        self.running_mean = torch.mean(self.running_mean)
        self.thresh = self.running_mean + self.running_sd

      # implement running mean after first epoch
      # if kwargs['epoch'] > self.init_epochs and iter == 0: #exponential moving average
      #   N = self.running_mean_counts
      #   K = (2/(N+1))
      #   self.running_mean = (self.running_mean * K) + (mean * (1-K))
      #   self.thresh = self.running_mean
      #   self.running_mean_counts += 1

      
      if kwargs['epoch'] >= self.init_epochs:

        top_indices = (sim_scores > self.thresh).nonzero()
        
        if top_indices.shape[0] > 0:
          self.clusterdict[self.batch_cluster_count]= torch.cat((q_in[select_idx].unsqueeze(0), q_in[top_indices])).squeeze()
          self.clusterdict_q[self.batch_cluster_count] = torch.cat((q[select_idx].unsqueeze(0), q[top_indices])).squeeze()
          index_list = torch.cat((select_idx, sample_indices[top_indices].flatten()))
          interval_list = [interval_ids[i] for i in index_list]
          self.clusterdict_intervals[self.batch_cluster_count] = interval_list
          try:
            self.clusterdict_labels[self.batch_cluster_count] = torch.cat((select_idx, sample_indices[top_indices].flatten()))
          except Exception:
            pdb.set_trace()
          self.batch_cluster_count += 1
      
        worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
        self.select_idx = sample_indices[bot_indices]
        labels[sample_indices[top_indices]] = 0
        labels[sample_indices[bot_indices]] = 0
        
        
    return labels


  def global_lifted_embedding_loss(self, q, y, interval_ids, training, global_bool, **kwargs):
    # https://arxiv.org/pdf/1703.07737.pdf
    '''
    #losses for TopK clustering, the im_k is not used at all, only q is used
    '''
    
    batch_size = y.shape[0]
    not_all_assigned = True
    labels = torch.ones(batch_size)
    iter = 0
    self.select_idx = None
    self.cluster_count = 0

    if kwargs['epoch'] < self.init_epochs:
      loss = torch.zeros(1)
      return loss.squeeze(), q


    while not_all_assigned:
        
      if self.self_sup == False:
        labels = self.bootstrap_cluster_DTW(q.transpose(1,2).clone(), y.transpose(1,2).clone(), interval_ids, labels, iter, top_k = batch_size//8, **kwargs)
      if self.self_sup == True:
        labels = self.bootstrap_cluster_DTW(q.transpose(1,2).clone(), y.transpose(1,2).clone(), labels, interval_ids, iter, top_k = batch_size//8, **kwargs)
      
      iter += 1

      if torch.sum(labels) == 0:
          not_all_assigned = False    
    
    loss = []
    if kwargs['epoch'] >= self.init_epochs:
      im_q = self.encoder_q(q)  # queries: NxCxT
      im_q = nn.functional.normalize(im_q, dim=1).permute(0,2,1)  # already normalized

      #intialize global_dict
      if bool(self.global_dict) == False:
        self.global_dict = self.clusterdict.copy()
        self.global_dict_q = self.clusterdict_q.copy()
        self.global_dict_intervals = self.clusterdict_intervals.copy()
      centroids = [cluster_vals[random.randint(0 ,cluster_vals.shape[0]-1)] for cluster_vals in self.global_dict.values()] #do random sampling better
      #centroids = [cluster_vals[0] for cluster_vals in self.global_dict.values()]
      for clusters,v in self.clusterdict.items():
        with torch.no_grad():
          if self.DTW_true == True: 
            sim_scores_global = self.softdtw(v[0].unsqueeze(0),torch.stack(centroids))
          if self.DTW_true == False:
            sim_scores_global = torch.einsum('nct, kct->k', [v[0].unsqueeze(0),torch.stack(centroids)])
        #Need Gradients here now
        top_idx = int(torch.argmax(sim_scores_global))
        # if bool(self.global_dict) == True:
        #   global_vals = self.global_dict.values()
        #   global_tensor_vals = torch.cat(list(self.global_dict.values()))
        #   pos_labels = torch.zeros(global_tensor_vals.shape[0], dtype=torch.bool)

        if sim_scores_global[top_idx] > self.thresh:

          
          #print(self.global_dict[top_idx]].shape)

          with torch.no_grad():
            other_cluster_vals = [v for k,v in self.global_dict_q.items() if top_idx != k]
            other_cluster_vals = torch.cat(other_cluster_vals).clone().detach()
            other_cluster_vals = self.encoder_q(other_cluster_vals.permute(0,2,1))  # queries: NxCxT
            other_cluster_vals = nn.functional.normalize(other_cluster_vals, dim=1).permute(0,2,1)  # already normalized
            try:
              positives = self.global_dict_q[top_idx].detach()
            except Exception:
              pdb.set_trace()
            positives = self.encoder_q(positives.permute(0,2,1))
            positives = nn.functional.normalize(positives, dim=1).permute(0,2,1)


          try:  
            l_pos = torch.einsum('nct,kct->nkt', [im_q[self.clusterdict_labels[clusters]].squeeze(), positives]).cuda()
          except Exception:
            pdb.set_trace()
          if len(other_cluster_vals) > 0: 
            l_neg = torch.einsum('nct, kct->nkt', [im_q[self.clusterdict_labels[clusters]].squeeze(),other_cluster_vals]).cuda()
            logits = torch.cat([l_pos, l_neg], dim=1).cuda()
          else: 
            logits = l_pos

          
          logits = logits - torch.max(logits)

          # apply temperature
          logits /= self.T
          logits-= logits.min(1, keepdim=True)[0]
          logits /= logits.max(1, keepdim=True)[0]

          labels = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2], dtype=torch.float).cuda()
          #labels = torch.zeros_like(logits).cuda()
          labels[:,:l_pos.shape[1],:] = 1
          loss.append(nn.BCELoss().cuda()(logits, labels.float()))
          t = torch.cat([self.global_dict[top_idx], v.detach()]) 
          self.global_dict[top_idx] = t[-20:]
          try:
            t = torch.cat([self.global_dict_q[top_idx], q[self.clusterdict_labels[clusters]].squeeze().permute(0,2,1)]) 
            self.global_dict_q[top_idx] = t[-20:]
            
            t = self.global_dict_intervals[top_idx] + self.clusterdict_intervals[clusters]
            self.global_dict_intervals[top_idx] = t 
          except Exception:
            pdb.set_trace()

        else:
          self.global_dict[len(self.global_dict)]= v.detach()
          self.global_dict_q[len(self.global_dict_q)] = q[self.clusterdict_labels[clusters]].squeeze().detach().permute(0,2,1)
          self.global_dict_intervals[len(self.global_dict_intervals)] = self.clusterdict_intervals[clusters]
      if len(loss) == 0:
        loss = torch.zeros(1)
        
      else:
        loss = torch.stack(loss).mean() 

    self.clusterdict = dict()
    self.clusterdict_q = dict()
    self.clusterdict_intervals = dict()
    self.clusterdict_labels = dict()
    self.batch_cluster_count = 0
    return loss.squeeze(), im_q.permute(0,2,1)


  def forward(self, im1 , y, interval_ids, global_var, **kwargs):
    """
    Input:
        im_q: a batch of query images
    Output:
        loss
    """
    # compute loss
    loss, q = self.global_lifted_embedding_loss(im1, y, interval_ids, self.training, global_var, **kwargs)

    return q, [loss]


class MoCo_GlobalDTW2(MoCo):
  def __init__(self, enc, self_sup, in_channels=128, time_steps=64, K=512, m=0.99, T=0.1, bn_splits=1, margin = 0, symmetric=False, **kwargs):
    super().__init__(enc, in_channels, time_steps, K, m, T, bn_splits, symmetric, **kwargs)
    self.softdtw = SoftDTW(use_cuda = True)
    self.clusterdict = dict()
    self.clusterdict_labels = dict()
    self.margin = margin
    self.select_idx = None
    self.encoder_q = enc
    self.batch_cluster_count = 0
    self.init_epochs = 2

    #global mean
    self.DTW_true = bool(kwargs['DTW'])
    self.kwargs = kwargs
    self.running_mean_list = []
    self.running_mean_counts = 0
    self.running_mean = None
    self.running_sd = None

    #global dict
    self.global_dict = {}
    self.global_enc_dict = {}
    self.cluster_count = 0

    self.self_sup = self_sup

  def correct_sample(self, im1):
    with torch.no_grad():
      q = self.encoder_q(im1)
      q = nn.functional.normalize(q, dim=1)  # already normalized

    return q, [torch.zeros(1)[0].to(im1.device)]


    #save cluster dict locally for gradient saving
    

  def bootstrap_cluster_DTW(self, q, y, labels, iter, top_k = 5, bootstrap = True, **kwargs): #anchor: 1 X T X E
    with torch.no_grad():
      # q = self.encoder_q(q.permute(0,2,1))  # queries: NxCxT
      # q = nn.functional.normalize(q, dim=1)  # already normalized
      # q = q.permute(0,2,1)
      if self.self_sup == False:
        q_in = y
      
      if self.self_sup == True:
        q = self.encoder_q(q.permute(0,2,1))
        q = nn.functional.normalize(q, dim=1)
        q = q.permute(0,2,1)
        q_in = q


      if self.select_idx == None:
          sample_indices = torch.nonzero(labels)
          select_idx = torch.randperm(len(sample_indices))[:1]
          anchor = q_in[select_idx]
      else:
          select_idx = self.select_idx[0]
          anchor = q_in[select_idx]

      labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
      sample_indices = torch.nonzero(labels) #update sample_indices to be used later
      remain_size = sample_indices.shape[0]

      if self.DTW_true:
          anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
          sim_scores = self.softdtw(anchor,q_in[sample_indices[:,0]].clone()) # calculate sim scores via dtw

      if self.DTW_true == False:
          anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
          sim_scores = torch.einsum('nct,nct->n', [anchor, q_in[sample_indices[:,0]].clone()])

      scores_size = sim_scores.shape[0]
      if iter == 0:
        mean = torch.mean(sim_scores)
        self.running_mean_list.append(mean)
        self.running_mean_counts += 1

      #Get Mean SD estimates (pt 1)
      if kwargs['epoch'] == self.init_epochs:
        return torch.zeros_like(labels)

      #Get Mean SD estimates (pt 2)
      if kwargs['epoch'] >= self.init_epochs and self.running_mean == None:
        self.running_mean = torch.stack(self.running_mean_list)
        self.running_sd = torch.std(self.running_mean)
        self.running_mean = torch.mean(self.running_mean)
        self.thresh = self.running_mean + self.running_sd

      # implement running mean after first epoch
      # if kwargs['epoch'] > self.init_epochs and iter == 0: #exponential moving average
      #   N = self.running_mean_counts
      #   K = (2/(N+1))
      #   self.running_mean = (self.running_mean * K) + (mean * (1-K))
      #   self.thresh = self.running_mean
      #   self.running_mean_counts += 1

      
      if kwargs['epoch'] >= self.init_epochs:

        top_indices = (sim_scores > self.thresh).nonzero()
        
        if top_indices.shape[0] > 0:
          self.clusterdict[self.batch_cluster_count]= torch.cat((q[select_idx].unsqueeze(0), q[top_indices])).squeeze()

          try:
            self.clusterdict_labels[self.batch_cluster_count] = torch.cat((select_idx, sample_indices[top_indices].flatten()))
          except Exception:
            pdb.set_trace()
          self.batch_cluster_count += 1
      
        worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
        self.select_idx = sample_indices[bot_indices]
        labels[sample_indices[top_indices]] = 0
        labels[sample_indices[bot_indices]] = 0
        
        
    return labels


  def global_lifted_embedding_loss(self, im_q, y, training, global_bool, **kwargs):
    # https://arxiv.org/pdf/1703.07737.pdf
    '''
    #losses for TopK clustering, the im_k is not used at all, only q is used
    '''
    
    batch_size = y.shape[0]
    not_all_assigned = True
    labels = torch.ones(batch_size)
    iter = 0
    self.select_idx = None
    self.cluster_count = 0

    if kwargs['epoch'] < self.init_epochs:
      loss = torch.zeros(1)
      return loss.squeeze(), im_q


    while not_all_assigned:
        
      if self.self_sup == False:
      # labels = self.cluster_DTW(q.transpose(1,2),  k.transpose(1,2), labels, iter, top_k = 5)
        labels = self.bootstrap_cluster_DTW(im_q.transpose(1,2).clone(), y.transpose(1,2).clone(), labels, iter, top_k = batch_size//8, **kwargs)
      if self.self_sup == True:
        labels = self.bootstrap_cluster_DTW(im_q.transpose(1,2).clone(), y.transpose(1,2).clone(), labels, iter, top_k = batch_size//8, **kwargs)
      
      iter += 1

      if torch.sum(labels) == 0:
          not_all_assigned = False    
    
    loss = []
    if kwargs['epoch'] >= self.init_epochs:
      im_q = self.encoder_q(im_q)  # queries: NxCxT
      im_q = nn.functional.normalize(im_q, dim=1).permute(0,2,1)  # already normalized

      #intialize global_dict
      if bool(self.global_dict) == False:
        self.global_dict = self.clusterdict.copy()
        self.global_enc_dict = self.clusterdict.copy()
      centroids = [cluster_vals[random.randint(0 ,cluster_vals.shape[0]-1)] for cluster_vals in self.global_dict.values()] #do random sampling better
      #centroids = [cluster_vals[0] for cluster_vals in self.global_dict.values()]
      for clusters,v in self.clusterdict.items():
        with torch.no_grad():
          if self.DTW_true == True: 
            sim_scores_global = self.softdtw(v[0].unsqueeze(0),torch.stack(centroids))
          if self.DTW_true == False:
            sim_scores_global = torch.einsum('nct, kct->k', [v[0].unsqueeze(0),torch.stack(centroids)])

        #Need Gradients here now
        top_idx = int(torch.argmax(sim_scores_global))
        # if bool(self.global_dict) == True:
        #   global_vals = self.global_dict.values()
        #   global_tensor_vals = torch.cat(list(self.global_dict.values()))
        #   pos_labels = torch.zeros(global_tensor_vals.shape[0], dtype=torch.bool)
        

        if sim_scores_global[top_idx] > self.thresh:

          other_cluster_vals = [v for k,v in self.global_dict.items() if top_idx != k]
          #print(self.global_dict[top_idx]].shape)
          
          try: 
            l_pos = torch.einsum('nct,kct->nkt', [im_q[self.clusterdict_labels[clusters]].squeeze(), self.global_dict[top_idx].detach()]).cuda()
          except Exception:
            pdb.set_trace()
          if len(other_cluster_vals) > 0: 
            l_neg = torch.einsum('nct, kct->nkt', [im_q[self.clusterdict_labels[clusters]].squeeze(),torch.cat(other_cluster_vals).clone().detach()]).cuda()
            logits = torch.cat([l_pos, l_neg], dim=1).cuda()
          else: 
            logits = l_pos

          
          logits = logits - torch.max(logits)

          # apply temperature
          logits /= self.T
          logits-= logits.min(1, keepdim=True)[0]
          logits /= logits.max(1, keepdim=True)[0]

          # labels: positive key indicators
          #labels = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long).cuda() # TODO: mean???
          #Bx(N+P)xT:
          #BxT:
          #binary cross entropy 
          #Input: (N, *)(N,∗) where *∗ means, any number of additional dimensions

          #Target: (N, *)(N,∗) , same shape as the input


          #labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda()
          labels = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2], dtype=torch.float).cuda()
          #labels = torch.zeros_like(logits).cuda()
          labels[:,:l_pos.shape[1],:] = 1
          loss.append(nn.BCELoss().cuda()(logits, labels.float()))
          t = torch.cat([self.global_dict[top_idx], v.detach()]) 
          self.global_dict[top_idx] = t[-20:]
          # pdb.set_trace()
          # if kwargs['epoch'] == 4:
          #   with torch.no_grad():
          #     #for TSNE
          #     pdb.set_trace()
          #     self.global_enc_dict[top_idx] = torch.cat(self.global_enc_dict[top_idx], im_q[self.clusterdict_labels[clusters]].squeeze()) 
          #     self.global_enc_dict[top_idx] = t[-20:]
          #     pass
        else:
          self.global_dict[len(self.global_dict)]= v.detach()
          # if kwargs['epoch'] == 4:
          #   #for TSNE
          #   with torch.no_grad():
          #     self.global_enc_dict[len(self.global_dict)] = self.encoder_q(v).detach()
        
      if len(loss) == 0:
        loss = torch.zeros(1)
        
      else:
        loss = torch.stack(loss).mean() 

    self.clusterdict = dict()
    self.clusterdict_labels = dict()
    self.batch_cluster_count = 0
    return loss.squeeze(), im_q.permute(0,2,1)


  def forward(self, im1 , y, global_var, **kwargs):
    """
    Input:
        im_q: a batch of query images
    Output:
        loss
    """
    # compute loss
    loss, q = self.global_lifted_embedding_loss(im1, y, self.training, global_var, **kwargs)

    return q, [loss]

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



class ContrastiveExpir(nn.Module):
  def __init__(self, jointenc, enc, in_channels=128, time_steps=64, DTW = True, K=512, m=0.99, T=0.1, bn_splits=1, margin = 0, symmetric=False, **kwargs):
    super().__init__()
    self.encoder = jointenc
    self.contr_encoder = enc 
    self.softdtw = SoftDTW(use_cuda = True)
    self.margin = margin
    self.global_dict = None
    self.time_steps = time_steps
    self.DTW_true = True
    self.self_sup = True
    self.kwargs = kwargs


    #clustering params
    self.select_idx = None
    self.running_mean_list = []
    self.running_mean_counts = 0
    self.running_mean = None
    self.running_sd = None



  def bootstrap_cluster_DTW(self, q, k, labels, iter, top_k = 5, bootstrap = True, **kwargs): #anchor: 1 X T X E
    with torch.no_grad():
        if self.select_idx == None:
            sample_indices = torch.nonzero(labels)
            select_idx = torch.randperm(len(sample_indices))[:1]
            anchor = q[select_idx]
        else:
            select_idx = self.select_idx[0]
            anchor = q[select_idx]

        labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
        sample_indices = torch.nonzero(labels) #update sample_indices to be used later
        remain_size = sample_indices.shape[0]

        if self.DTW_true:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation

            sim_scores = self.softdtw(anchor,q[sample_indices[:,0]].clone()) # calculate sim scores via dtw

        if self.DTW_true == False:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            sim_scores = torch.einsum('nct,nct->n', [anchor, q[sample_indices[:,0]].clone()])

        scores_size = sim_scores.shape[0]

        # Base Cases 

        if kwargs['epoch'] == 0:
          if iter == 0:
              mean = torch.mean(self.softdtw(anchor,q[sample_indices[:,0]].clone()))
              self.running_mean_list.append(mean)
              self.running_mean_counts += 1
          return torch.zeros_like(labels)
        #   # implement running mean

        if kwargs['epoch'] == 1 and self.running_mean == None:
          self.running_mean = torch.stack(self.running_mean_list)
          self.running_sd = torch.std(self.running_mean)
          self.running_mean = torch.mean(self.running_mean)
          self.thresh = self.running_mean

        if iter == 0: #exponential moving average
          mean = self.running_mean
          N = self.running_mean_counts
          K = (2/(N+1))
          self.running_mean = (mean * K) + (self.running_mean * (1-K))
          self.thresh = self.running_mean
          self.running_mean_counts += 1
        
        if kwargs['epoch'] >= 1:

          top_indices = (sim_scores > self.thresh).nonzero()
          
          if top_indices.shape[0] > 0:
              # self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx]) #last elem is anchor seq.
              # self.clusterdict[iter]["vals"] = torch.stack(q[top_indices], q[select_idx])
              try:
                self.clusterdict[self.cluster_count]= torch.cat((q[select_idx].unsqueeze(0), q[top_indices])).squeeze()
                self.cluster_count += 1
              except Exception:
                pdb.post_mortem()
          #else: #버려
              # self.clusterdict[iter]["idx"] = torch.tensor(select_idx)
              # self.clusterdict[iter]["vals"] =  q[select_idx]
              #self.clusterdict[self.cluster_count] =  q[select_idx]
          
          worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
          self.select_idx = sample_indices[bot_indices]
          labels[sample_indices[top_indices]] = 0
          labels[sample_indices[bot_indices]] = 0
          
    return labels



  def update_global_dict(self):
    #update global_dict
    centroids = [cluster_vals[random.randint(0 ,cluster_vals.shape[0]-1)] for cluster_vals in self.global_dict.values()]
    for clusters,v in self.clusterdict.items():
        sim_scores_global = self.softdtw(v[0].unsqueeze(0),torch.stack(centroids))
        top_idx = int(torch.argmax(sim_scores_global))

        if sim_scores_global[top_idx] > self.thresh:
          try:
            #get only 10 seqs as cluster
            t = torch.cat([self.global_dict[top_idx], v]) #last elem is anchor seq
            # idx = torch.randperm(t.shape[0])
            # t = t[idx].view(t.size()) 
            self.global_dict[top_idx] = t[-10:]

          except Exception:
            pdb.post_mortem()
        else:
            self.global_dict[len(self.global_dict)]= v

  def cluster_contrastive(self, x, y, training, global_bool, **kwargs):
    # compute encoding 
    q_copy = x.copy()
    q = self.encoder(x, y, self.time_steps, **kwargs)
    q = self.contr_encoder(q)

    # compute query features
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # batch-wise clustering
    batch_size = y.shape[0]
    not_all_assigned = True
    labels = torch.ones(batch_size)
    iter = 0
    self.select_idx = None
    self.clusterdict = {}
    while not_all_assigned:
      if not self.self_sup:
        labels = self.bootstrap_cluster_DTW(y.transpose(1,2).clone().detach(),  y.transpose(1,2).clone().detach(), labels, iter, top_k = batch_size//8, **kwargs)
      if self.self_sup:
        labels = self.bootstrap_cluster_DTW(q.transpose(1,2).clone().detach(),  q.transpose(1,2).clone().detach(), labels, iter, top_k = batch_size//8, **kwargs)
        iter += 1
        if torch.sum(labels) == 0:
            not_all_assigned = False

    loss = torch.zeros(1).cuda()
    # first epoch gets skipped because it's estimating mean values
    if kwargs['epoch'] == 0:
      return loss.squeeze(), q
        
    # from second epoch
    if kwargs['epoch'] > 0:

      #if global cluster has not been made
      if self.global_dict == None:
        pdb.set_trace()
        self.global_dict = self.clusterdict.copy()
        
        

      self.update_global_dict()


      global_vals = self.global_dict.values()
      global_tensor_vals = torch.cat(list(self.global_dict.values()))
      print("GLOBALDICT:" , len(self.global_dict), "\n")
      pos_labels = torch.zeros(global_tensor_vals.shape[0], dtype=torch.bool)


      for cluster, v in self.global_dict.items():
        
        pos_labels[:v.shape[0]] = True
        neg_labels = ~(pos_labels.clone())

        other_cluster_vals = [v for k,v in self.global_dict.items() if cluster != k]

        l_pos = torch.einsum('nct,nct->nt', [v, v.detach()]).unsqueeze(1)
        # negative logits: NxKxT
        l_neg = torch.einsum('nct, kct->nkt', [v,torch.cat(other_cluster_vals)])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        #stability
        logits = logits - torch.max(logits)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda() # TODO: mean???

        loss += nn.CrossEntropyLoss().cuda()(logits, labels)


      loss = loss/iter #TODO: Change to mean properly (iter is probably wrong)
      self.clusterdict = dict()

    return loss.squeeze(), q


 

  def forward(self, x, y, global_var, self_sup, **kwargs):
    # compute loss
    loss, q = self.cluster_contrastive(x, y, self.training, global_var, **kwargs)
    return q, [loss]

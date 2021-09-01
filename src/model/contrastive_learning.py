import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
from .contrastiveDTW import MoCo_GlobalDTW, MoCo_GlobalDTW2 # MoCo_DTW
from .contrastive_global_expiration import *

import torch
import torch.nn as nn

from functools import partial

default_kwargs = lambda kwargs, key, default: kwargs.get(key) if kwargs.get(key) is not None else default

'''
modelKwargs:
  K: queue size
  m: momentum for key encoder updates
  T: temperature
  bn_splits: number of splits for the batch norm
  symmetric: flag for symmetric loss
'''
class JointLateClusterSoftContrastive_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.encoder = nn.Sequential(self.pose_encoder, self.unet)

    self.contrasive_learning = MoCo(self.encoder, in_channels=in_channels, time_steps=time_steps,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      y1, y2 = self.transforms(y.clone()), self.transforms(y.clone())
      z, i_loss1 = self.contrasive_learning(y1, y2)
    else:
      #if y.shape[1] > 64:
      #  pdb.set_trace()
      if kwargs['sample_flag']:
        shape = y.shape
        y = y.view(-1, 64, shape[-1])
      z, i_loss1 = self.contrasive_learning.sample(y)

    x, i_loss2 = self.decoder(z, labels, **kwargs)

    if kwargs['sample_flag']:
      x = x.view(1, shape[1], -1)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}

class JointLateClusterSoftContrastive2_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.encoder = nn.Sequential(self.pose_encoder, self.unet)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    if kwargs['sample_flag']:
      shape = y.shape
      y = y.view(-1, 64, shape[-1])

    z = self.encoder(y)
    x, i_loss2 = self.decoder(z, labels, **kwargs)

    if kwargs['sample_flag']:
      x = x.view(1, shape[1], -1)

    internal_losses += [torch.zeros(1)[0].to(x.device)]
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}

'''
Audio/language to pose model
w/ MoCo

'''
class JointLateClusterSoftContrastive3_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo(self.unet, in_channels=in_channels, time_steps=time_steps,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, x + torch.randn_like(x)/10)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.sample(x)

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)
    # if kwargs['sample_flag']:
    #   x = x.view(1, shape[-1], -1)#.permute(0, 2, 1) # (1, T, C)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}

'''
Audio/language to pose model
w/ Patchwise contrastive
'''
class JointLateClusterSoftContrastive4_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder
    self.encoder_logits = nn.Conv1d(256, in_channels, kernel_size=1, stride=1, padding=0)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.num_patches = default_kwargs(kwargs, 'num_patches', 5)
    self.window_size = default_kwargs(kwargs, 'window_size', 8)
    self.window_hop = default_kwargs(kwargs, 'window_hop', 5)
    self.patches = TemporalPatches(self.num_patches, self.window_size, self.window_hop)

    negs_from_minibatch = default_kwargs(kwargs, 'negs_from_minibatch', False)
    self.patch_nce = PatchNCELoss(negs_from_minibatch=negs_from_minibatch)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']
    self.lambda_nce = default_kwargs(kwargs, 'lambda_nce', 1)

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]
    x_ = x

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    x = self.encoder_logits(x)
    z_k = x
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z = self.unet(x)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z = self.unet(x)

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)


    '''
    y -> pose_encoder -> unet -> z
    y_cap -> pose_encoder -> unet -> z_cap
    x -> audio_text_encoder -> unet -> z_x
    '''
    ## Get patches
    z_q = self.pose_encoder(x, time_steps)
    z_q = self.encoder_logits(z_q)
    if kwargs['sample_flag']:
      z_q = z_q.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z_k = z_k.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
    #z_k = self.encoder(x_, y, time_steps, **kwargs)
    _, feats_q = self.unet(z_q, return_feats=True, feats=[])
    _, feats_k = self.unet(z_k, return_feats=True, feats=[])

    feats_q, patch_ids = self.patches(feats_q)
    feats_k, _ = self.patches(feats_k, patch_ids)

    ## Get NCE Loss
    i_loss1 = 0

    for idx, (q, k) in enumerate(zip(feats_q, feats_k)):
      if isinstance(self.lambda_nce, list):
        assert len(self.lambda_nce) >= len(feats_q), 'length of lambda_nce should be == {}'.format(len(feats_q))
        lam = self.lambda_nce[idx]
      else:
        lam = self.lambda_nce
      i_loss1 += self.patch_nce(q, k)[1] * lam

    internal_losses += [i_loss1]
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}


'''
Pose to pose model
w/ Patchwise contrastive
'''
class JointLateClusterSoftContrastive5_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    # self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
    #                                                       in_channels=in_channels,
    #                                                       out_feats=out_feats,
    #                                                       num_clusters=num_clusters,
    #                                                       p=p, E=E, **kwargs)

    #self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.num_patches = default_kwargs(kwargs, 'num_patches', 5)
    self.window_size = default_kwargs(kwargs, 'window_size', 8)
    self.window_hop = default_kwargs(kwargs, 'window_hop', 5)
    self.patches = TemporalPatches(self.num_patches, self.window_size, self.window_hop)

    negs_from_minibatch = default_kwargs(kwargs, 'negs_from_minibatch', False)
    self.patch_nce = PatchNCELoss(negs_from_minibatch=negs_from_minibatch)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']
    self.lambda_nce = kwargs.get('lambda_nce') if kwargs.get('lambda_nce') is not None else 1

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]
    x_ = x

    x = self.encoder(y, time_steps) # (B, C, T)
    z_k = x

    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z = self.unet(x)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z = self.unet(x)

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)

    '''
    y -> pose_encoder -> unet -> z
    y_cap -> pose_encoder -> unet -> z_cap
    '''
    ## Get patches
    z_q = self.encoder(x, time_steps)
    if kwargs['sample_flag']:
      z_q = z_q.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z_k = z_k.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
    #z_k = self.encoder(x_, y, time_steps, **kwargs)
    _, feats_q = self.unet(z_q, return_feats=True, feats=[])
    _, feats_k = self.unet(z_k, return_feats=True, feats=[])

    feats_q, patch_ids = self.patches(feats_q[2:])
    feats_k, _ = self.patches(feats_k[2:], patch_ids)

    ## Get NCE Loss
    i_loss1 = 0

    for q, k in zip(feats_q, feats_k):
      i_loss1 += self.patch_nce(q, k)[1] * self.lambda_nce

    internal_losses += [i_loss1]
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}



'''
Audio/language to pose model
w/ MoCo
w/ Patchwise contrastive
'''

'''
Unpaired Style Transfer
A1 -> P1
   -> P2
'''


class JointLateClusterSoftContrastiveNoDTW_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo_DTW(self.unet, in_channels=in_channels, time_steps=time_steps, DTW = False,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, x + torch.randn_like(x)/10)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.sample(x)

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)
    # if kwargs['sample_flag']:
    #   x = x.view(1, shape[-1], -1)#.permute(0, 2, 1) # (1, T, C)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}


class JointLateClusterSoftContrastiveKmeans_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = KMeansContr(self.unet, in_channels=in_channels, time_steps=time_steps, DTW = False,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']
    self.batch_iter = 0

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, x + torch.randn_like(x)/10, batch_iter = self.batch_iter)
      self.batch_iter += 1
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.sample(x)

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)
    # if kwargs['sample_flag']:
    #   x = x.view(1, shape[-1], -1)#.permute(0, 2, 1) # (1, T, C)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}
  

'''
contrastive learning with DTW as the similarity metric
'''
class JointLateClusterSoftContrastiveDTW_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo_DTW(self.unet, in_channels=in_channels, time_steps=time_steps, DTW = True, self_sup = False,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)


    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10,y)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.sample(x)

    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}

class JointLateClusterSoftContrastiveDTWSelfSup_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo_DTW(self.unet, in_channels=in_channels, time_steps=time_steps, self_sup= True,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, y)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.sample(x)

    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}
  
class JointLateClusterSoftContrastiveGlobalDTWSelfSup_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, bootstrap = True, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo_GlobalDTW(self.unet, self_sup = True, in_channels=in_channels, time_steps=time_steps,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, y, global_var = True, **kwargs)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.correct_sample(x)
      if bool(self.contrasive_learning.global_dict):
        self.contrasive_learning.global_dict = {}

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)
    # if kwargs['sample_flag']:
    #   x = x.view(1, shape[-1], -1)#.permute(0, 2, 1) # (1, T, C)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}


class JointLateClusterSoftContrastiveGlobalDTW_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, bootstrap = True, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo_GlobalDTW(self.unet, self_sup = False, in_channels=in_channels, time_steps=time_steps,
                                    **kwargs)

    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)

  def forward(self, x, y, time_steps=None, **kwargs):
    interval_ids = x[-1]
    x = x[:-1]
    x = [x_.cuda() for x_ in x]
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, y, interval_ids, global_var = True, **kwargs)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.correct_sample(x)
      if bool(self.contrasive_learning.global_dict):
        print("Writing Dictionary Clusters")
        
        # torch.save(self.contrasive_learning.global_dict, ${savepath}.format(kwargs['speaker']))
        # torch.save(self.contrasive_learning.global_dict_q, ${savepath}.format(kwargs['speaker']))
        # torch.save(self.contrasive_learning.global_dict_intervals, ${savepath}.format(kwargs['speaker']))
        
        
        enc_q = {}
        with torch.no_grad():
          for k, v in self.contrasive_learning.global_dict_q.items():
            enc_q[k] = self.unet(v.permute(0,2,1))
          # torch.save(enc_q, ${savepath}.format(kwargs['speaker']))
        self.contrasive_learning.global_dict = {}
        self.contrasive_learning.global_dict_q = {}
        self.contrasive_learning.global_dict_intervals = {}
        
        
        # if kwargs['epoch'] == 2:
        #   pdb.set_trace()
          #save self.contrasive_learning.global_dict
          #save self.contrasive_learning.global_enc_dict

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)
    # if kwargs['sample_flag']:
    #   x = x.view(1, shape[-1], -1)#.permute(0, 2, 1) # (1, T, C)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}


class JointLateClusterSoftContrastiveGlobalDTW2_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=1, p = 0, E = 256, bootstrap = True, **kwargs):
    super().__init__()
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.audio_text_encoder = MultimodalMultiscaleEncoder(time_steps=time_steps,
                                                          in_channels=in_channels,
                                                          out_feats=out_feats,
                                                          num_clusters=num_clusters,
                                                          p=p, E=E, **kwargs)

    self.encoder = JointEncoder(self.pose_encoder, self.audio_text_encoder, num_iters=kwargs['num_batches']*5)
    #self.encoder = self.audio_text_encoder

    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.contrasive_learning = MoCo_GlobalDTW(self.unet, self_sup = False, in_channels=in_channels, time_steps=time_steps,
                                    **kwargs)

    num_clusters = 1
    self.decoder = MixGANDecoder(time_steps, in_channels, out_feats, num_clusters, p, E, **kwargs)
    self.transforms = kwargs['transforms']

  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    x = self.encoder(x, y, time_steps, **kwargs) # (B, C, T)
    if kwargs['description'] == 'train' and not kwargs['sample_flag']:
      #y1, y2 = self.transforms(x.clone()), self.transforms(x.clone())
      z, i_loss1 = self.contrasive_learning(x + torch.randn_like(x)/10, y, global_var = True, **kwargs)
    else:
      if kwargs['sample_flag']:
        shape = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 64, shape[1]).permute(0, 2, 1) # (B, C, T)
      z, i_loss1 = self.contrasive_learning.correct_sample(x)
      if bool(self.contrasive_learning.global_dict):
        self.contrasive_learning.global_dict = {}
        if kwargs['epochs'] == 3:
          pdb.set_trace()
          #save self.contrasive_learning.global_dict
          #save self.contrasive_learning.global_enc_dict

      #pdb.set_trace()
    if kwargs['sample_flag']:
      z_ = z.permute(0, 2, 1).reshape(1, -1, z.shape[1]).permute(0, 2, 1)
    else:
      z_ = z
    x, i_loss2 = self.decoder(z_, labels, **kwargs)
    # if kwargs['sample_flag']:
    #   x = x.view(1, shape[-1], -1)#.permute(0, 2, 1) # (1, T, C)

    internal_losses += i_loss1
    internal_losses += i_loss2

    return x, internal_losses, {'z':z}
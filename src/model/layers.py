import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb
import copy
from typing import Optional, Any

import torch
import torch.nn as nn
from transformers import BertModel
# from functions import vq, vq_st
import logging

from soft_dtw_cuda import SoftDTW, SoftDTW_DBS
#from fastdtw_mod import fastdtw

logging.getLogger('transformers').setLevel(logging.CRITICAL)

FLOAT = torch.float # torch.float | torch.double

def num_powers_of_two(x):
  num_powers = 0
  while x>1:
    if x % 2 == 0:
      x /= 2
      num_powers += 1
    else:
      break
  return num_powers

def next_multiple_power_of_two(x, power=5):
  curr_power = num_powers_of_two(x)
  if curr_power < power:
    x = x * (2**(power-curr_power))
  return x

class Transpose(nn.Module):
  def __init__(self, idx):
    super().__init__()
    self.param = torch.nn.Parameter(torch.ones(1))
    self.idx = idx

  def forward(self, x, *args, **kwargs):
    return x.transpose(*self.idx)

class ConvNormRelu(nn.Module):
  def __init__(self, in_channels, out_channels,
               type='1d', leaky=False,
               downsample=False, kernel_size=None, stride=None,
               padding=None, p=0, groups=1):
    super(ConvNormRelu, self).__init__()
    if kernel_size is None and stride is None:
      if not downsample:
        kernel_size = 3
        stride = 1
      else:
        kernel_size = 4
        stride = 2

    if padding is None:
      if isinstance(kernel_size, int) and isinstance(stride, tuple):
        padding = tuple(int((kernel_size - st)/2) for st in stride)
      elif isinstance(kernel_size, tuple) and isinstance(stride, int):
        padding = tuple(int((ks - stride)/2) for ks in kernel_size)
      elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
        assert len(kernel_size) == len(stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(len(kernel_size), len(stride))
        padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, kernel_size))
      else:
        padding = int((kernel_size - stride)/2)


    in_channels = in_channels*groups
    out_channels = out_channels*groups
    if type == '1d':
      self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm1d(out_channels)
      self.dropout = nn.Dropout(p=p)
    elif type == '2d':
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm2d(out_channels)
      self.dropout = nn.Dropout2d(p=p)
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=0.2)
    else:
      self.relu = nn.ReLU()

  def forward(self, x, **kwargs):
    return self.relu(self.norm(self.dropout(self.conv(x))))

class UNet1D(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)

  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super(UNet1D, self).__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, return_bottleneck=False, return_feats=False, feats=[]):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    bn = x

    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)
      if return_feats:
        feats.append(x)

    if return_feats:
      return x, feats
    elif return_bottleneck:
      return x, bn
    else:
      return x


class PoseEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1):
    super(PoseEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None, return_feats=False, feats=[]):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    if return_feats:
      for conv in self.conv:
        x = conv(x)
        feats.append(x)
    else:
      x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)

    if return_feats:
      return x, feats
    else:
      return x

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class MultimodalMultiscaleEncoder(nn.Module):
  '''
  Encoding language and audio jointly with unsupervised alignment
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        text_key=key
    if text_key:
      self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
    self.pos_encoder = PositionalEncoding(256, p)

    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                      p=p)]))
    self.norm = nn.LayerNorm(256)

  def forward(self, x, y, time_steps=None, **kwargs):
    # Late Fusion with Joint
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.
    #if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
    mod_map = {}

    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        mod_map['text'] = i
        #.set_trace() #self.training FALSE

        for te in self.text_encoder:
          x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =0, pos_encoder=self.pos_encoder, **kwargs) ## (B, channels, time)

      if modality.split('/')[0] == 'audio':
        mod_map['audio'] = i
        if x[i].dim() == 3:
          x[i] = x[i].unsqueeze(dim=1)
        x[i] = self.audio_encoder(x[i], time_steps)

    if len(x) >= 2:
      memory = x[mod_map['text']]
      tgt = x[mod_map['audio']]
      tgt = self.pos_encoder(tgt.permute(2, 0, 1)).permute(1, 2, 0)
      x_ = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=0, **kwargs)
      x = torch.cat([x_, x[mod_map['audio']]], dim=1)
      x = self.concat_encoder2(x)
    else:
      x = torch.cat(tuple(x),  dim=1)

    x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)
    return x

class JointEncoder(nn.Module):
  def __init__(self, m1, m2, num_iters):
    super().__init__()
    self.m1 = m1
    self.m2 = m2
    self.thresh = Curriculum(0, 1, num_iters)

  def forward(self, x, y, time_steps=None, **kwargs):

    if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
      return self.m1(y, time_steps) # pose encoder
    else:
      return self.m2(x, y, time_steps, **kwargs) # audio/language encoder

class MixGANDecoder(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters

      ## no residual connections by default
      self.resnet = kwargs.get('resnet') if kwargs.get('resnet') is not None else False

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, input_channels=in_channels)
      self.classify_loss = nn.CrossEntropyLoss()
      self.decoder = nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                 type='1d', leaky=True, downsample=False,
                                                 p=p, groups=self.num_clusters)
                                    for i in range(4)])
      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

      self.labels_cap_soft = None

    def index_select_outputs(self, x, labels):
      '''
      x: (B, num_clusters*out_feats, T)
      labels: (B, T, num_clusters)
      '''
      x = x.transpose(2, 1)
      x = x.view(x.shape[0], x.shape[1], self.num_clusters, -1)
      labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling

      x = (x * labels.unsqueeze(-1)).sum(dim=-2)
      return x

    def forward(self, x, labels, **kwargs):
      internal_losses = []

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      for dec in self.decoder:
        if self.resnet:
          x = dec(x) + x
        else:
          x = dec(x)

      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)

      return x, internal_losses

#Positional Encoding missing in vanilla Transformer
#source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AudioEncoder(nn.Module):
  '''
  input_shape:  (N, C, time, frequency)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=1, kernel_size=None, stride=None, p=0, groups=1):
    super(AudioEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(128, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=(3,8), stride=1, p=p, groups=groups))

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear', align_corners=False)
    x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='nearest')
    x = x.squeeze(dim=-1)
    return x



class BertEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.linear = torch.nn.Linear(768, out_feats)

  def _generate_source_key_padding_mask(self, token_count, mask_val=0):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count) - mask_val
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = mask_val
    return mask.bool().to(token_count.device)

  def output_repeat_text(self, memory, token_duration):
    memory_list = []
    for b in range(memory.shape[1]):
      memory_list_ = [memory[i, b:b+1].repeat(int(token_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
      memory_list.append(torch.cat(memory_list_, dim=0))
    memory = torch.cat(memory_list, dim=1)
    return memory

  def chunk(self, x, pad, max_len=512):
    x_len = x.shape[-1]
    batch = (x_len - 1) // max_len + 1
    if batch > 1:
      new_len = max_len * batch
      x = torch.cat([x, torch.zeros(1, new_len-x_len).float().to(x.device)], dim=-1)
      pad = torch.cat([pad, torch.zeros(1, new_len-x_len).bool().to(x.device)], dim=-1)
      x = x.view(batch, -1)
      pad = pad.view(batch, -1)

    return x, pad, x_len, batch

  def forward(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    token_type_ids = None
    # if len(x.shape) == 3:
    #   sample_flag = True
    # else:
    #   sample_flag = False

    sample_flag = kwargs["sample_flag"]

    ## Create Masks
    assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    token_duration = kwargs['text/token_duration']
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'], mask_val=1)
    #if src_key_padding_mask.shape[1] != x.shape[1]:
    if sample_flag:
      x = x.view(1, -1)
      src_key_padding_mask = src_key_padding_mask.view(1, -1)
      x, src_key_padding_mask, orig_len, batch = self.chunk(x, src_key_padding_mask)

      #src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    memory, pooled_output = self.bert(x.long(), token_type_ids, src_key_padding_mask.long())

    memory = self.linear(memory)

    if sample_flag:
      memory = memory.view(1, -1, memory.shape[-1])[:, :orig_len]
      token_duration = token_duration.view(memory.shape[0], memory.shape[1])[:, :orig_len]

    if 'pos_encoder' in kwargs: ## add positional embedding before repeating -- Useful is used in conjunction with another transformer
      memory = kwargs['pos_encoder'](memory.transpose(1, 0)).transpose(1, 0) ## needs input in the form of (T, B, C)

    if output_repeat:
      memory = self.output_repeat_text(memory.transpose(1, 0), token_duration).transpose(1, 0)

    return memory.transpose(-1, -2)

  def forward_archive(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    token_type_ids = None
    if len(x.shape) == 3:
      sample_flag = True
      x = x.squeeze(0)
    else:
      sample_flag = False

    ## Create Masks
    assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    token_duration = kwargs['text/token_duration']
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'], mask_val=1)
    #if src_key_padding_mask.shape[1] != x.shape[1]:
    # if sample_flag:
    #   x = x.view(1, -1)
    #   src_key_padding_mask = src_key_padding_mask.view(1, -1)
    #   x, src_key_padding_mask, orig_len, batch = self.chunk(x, src_key_padding_mask)

      #src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    memory, pooled_output = self.bert(x.long(), token_type_ids, src_key_padding_mask.long())
    memory = self.linear(memory)

    # if sample_flag:
    #   memory = memory.view(1, -1, memory.shape[-1])[:, :orig_len]
    #   token_duration = token_duration.view(memory.shape[0], memory.shape[1])[:, :orig_len]

    if 'pos_encoder' in kwargs: ## add positional embedding before repeating -- Useful is used in conjunction with another transformer
      memory = kwargs['pos_encoder'](memory.transpose(1, 0)).transpose(1, 0) ## needs input in the form of (T, B, C)

    if output_repeat:
      memory = self.output_repeat_text(memory.transpose(1, 0), token_duration).transpose(1, 0)

    if sample_flag:
      memory = memory.view(1, -1, memory.shape[-1])

    return memory.transpose(-1, -2)


class MultiScaleBertEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=256, **kwargs):
    super().__init__()
    self.word_encoder = BertEncoder(out_feats=out_feats)

    ## Frame Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = out_feats
    self.pos_encoder = PositionalEncoding(self.ninp, p)
    self.frame_pos_encoder = FramesPositionalEncoding(input_channels=E, dropout=0)

    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.frame_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def forward(self, x, y, input_repeat=0, output_repeat=1, **kwargs):
    if kwargs['description'] == 'train':
      is_train = True
    else:
      is_train = False
    memory = self.word_encoder(x, y, input_repeat=0, output_repeat=1, pos_encoder=self.pos_encoder, **kwargs).transpose(-1, -2) ## (B, T) -> (B, T, C)
    memory = self.frame_pos_encoder(memory.transpose(1, 0), kwargs['text/token_duration'], is_train) # (T, B, C) as input -> (T, B, C)
    memory = self.frame_encoder(memory)
    return memory.transpose(1, 0).transpose(-1, -2) # (B, C, T)

class MultimodalTransformerFusion(nn.Module):
  '''
  tgt: audio signal (T, B, C)
  src: text signal (L, B, C), if input_repeat == 0 => L!=T and if input_repeat == 1 => L==T
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=256, nlayer=2,**kwargs):
    super().__init__()
    ## Frame Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.nlayer = nlayer
    self.ninp = out_feats

    decoder_layers = nn.TransformerDecoderLayer(self.ninp, self.nhead, self.nhid)
    decoder_norm = torch.nn.LayerNorm(self.ninp)
    self.memory_decoder = nn.TransformerDecoder(decoder_layers, self.nlayer, decoder_norm) # Norm

  def _generate_source_key_padding_mask(self, token_count, mask_val=0):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count) - mask_val
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = mask_val
    return mask.bool().to(token_count.device)

  def _generate_source_mask(self, token_duration, tgt_len, bsz, input_repeat):
    if input_repeat == 0:
      mask = torch.ones(bsz*self.nhead, tgt_len, token_duration.shape[-1]) # (B, T, L)
    else:
      mask = torch.ones(bsz*self.nhead, tgt_len, tgt_len) # (B, T, T)
    for b in range(token_duration.shape[0]):
      pos = 0
      for i in range(token_duration.shape[1]):
        duration = int(token_duration[b, i].item())
        if input_repeat == 0:
          mask[b*self.nhead:(b+1)*self.nhead, pos:pos+duration, i] = 0
        else:
          mask[b*self.nhead:(b+1)*self.nhead, pos:pos+duration, pos:pos+duration] = 0
        pos = pos + duration
    #mask = mask.float().masked_fill(mask==1, float('-inf')).masked_fill(mask==0, float(0.0)).to(token_duration.device)
    #return mask
    return mask.bool().to(token_duration.device)


  def output_repeat_text(self, memory, token_duration):
    memory_list = []
    for b in range(memory.shape[1]):
      memory_list_ = [memory[i, b:b+1].repeat(int(token_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
      memory_list.append(torch.cat(memory_list_, dim=0))
    memory = torch.cat(memory_list, dim=1)
    return memory

  '''
  tgt: (B, C, T) -> (T, B, C)
  memory: (B, C, L) -> (L, B, C)
  '''
  def forward(self, tgt, memory, y, input_repeat=0, output_repeat=1, src_mask=True, query_text=False, **kwargs):
    tgt = tgt.permute(2, 0, 1)
    memory = memory.permute(2, 0, 1)
    if kwargs['description'] == 'train':
      is_train = True
    else:
      is_train = False

    token_duration = kwargs['text/token_duration']
    token_count = kwargs['text/token_count']
    if token_duration.shape[0] != tgt.shape[1]: ## sample_loop
      token_duration = token_duration.view(1, -1)
      sample_flag = True
    else:
      sample_flag = False

    if src_mask:
      src_mask = self._generate_source_mask(token_duration, tgt.shape[0], tgt.shape[1], input_repeat)
    else:
      src_mask = None
    if input_repeat == 0:
      src_key_padding_mask = self._generate_source_key_padding_mask(token_count)
      if sample_flag:
        src_key_padding_mask = src_key_padding_mask.view(1, -1)
    else:
      src_key_padding_mask = None

    if not query_text:
      ## memory(~key and value) is text, tgt (~query) is audio
      #TODO: what is this?????
      memory = self.memory_decoder(tgt, memory, memory_key_padding_mask=src_key_padding_mask, memory_mask=src_mask[0])
    else:
      memory = self.memory_decoder(memory, tgt, tgt_key_padding_mask=src_key_padding_mask, tgt_mask=src_mask)

    return memory.transpose(1, 0).transpose(-1, -2) # (B, C, T)



class ClusterClassify(nn.Module):
  '''
  input_shape: (B, C, T)
  output_shape: (B, num_clusters, T)
  '''
  def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
    super().__init__()
    self.conv = nn.ModuleList()
    self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                             kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in range(5)])

    self.logits = nn.Conv1d(256*groups, num_clusters*groups, kernel_size=1, stride=1, groups=groups)

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    x = self.logits(x)
    return x


class Confidence(nn.Module):
  '''
  0 < confidence <= 1
  '''
  def __init__(self, beta=0.1, epsilon=1e-8):
    super().__init__()
    self.beta = beta
    self.epsilon = epsilon

  def forward(self, y, y_cap, confidence):
    if isinstance(confidence, int):
      confidence = torch.ones_like(y)
    sigma = self.get_sigma(confidence)
    P_YCAP_Y = self.p_ycap_y(y, y_cap, sigma)
    sigma_ycap = self.get_sigma(P_YCAP_Y)
    return self.get_entropy(sigma_ycap)

  def p_ycap_y(self, y, y_cap, sigma):
    diff = -(y-y_cap)**2
    diff_normalized = diff/(2*sigma**2)
    prob = torch.exp(diff_normalized)
    prob_normalized = prob*(1/(2*math.pi*sigma))
    return prob_normalized

  def get_sigma(self, confidence):
    mask = (confidence < self.epsilon).float()
    confidence = (1 - mask) * confidence + torch.ones_like(confidence)*self.epsilon*mask
    sigma = 1/(2*math.pi*confidence)
    return sigma

  ## entropy of a guassian
  def get_entropy(self, sigma):
    return 0.5*(torch.log(2*math.pi*math.e*(sigma**2)))*self.beta

class Repeat(nn.Module):
  def __init__(self, repeat, dim=-1):
    super().__init__()
    self.dim = dim
    self.repeat = repeat
    #self.temp = torch.nn.Parameter(torch.zeros(1))

  def forward(self, x):
    return x.repeat_interleave(self.repeat, self.dim)


class Curriculum():
  def __init__(self, start, end, num_iters):
    self.start = start
    self.end = end
    self.num_iters = num_iters
    self.iters = 0
    self.diff = (end-start)/num_iters
    self.value = start

  def step(self, flag=True):
    if flag:
      value_temp = self.value
      if self.iters < self.num_iters:
        self.value += self.diff
        self.iters += 1
        return value_temp
      else:
        return self.end
    else:
      return self.value


'''
Contrastive Learning
'''
class MoCo(nn.Module):
  '''
  in_channels: feature size
  time_steps: temporal dimension
  K: queue size
  m: momentum for key encoder updates
  T: temperature
  bn_splits: number of splits for the batch norm
  symmetric: flag for symmetric loss
  '''
  def __init__(self, enc, in_channels=128, time_steps=64, K=4096, m=0.99, T=0.1, bn_splits=1, symmetric=True, **kwargs):
    super().__init__()

    self.K = K
    self.m = m
    self.T = T
    self.symmetric = symmetric

    # create the encoders
    self.encoder_q = enc
    self.encoder_k = copy.deepcopy(enc)

    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
      #param_k.data.copy_(param_q.data)  # initialize
      param_k.requires_grad = False  # not update by gradient

    # create the queue
    self.register_buffer("queue", torch.randn(in_channels, K, time_steps))
    self.queue = nn.functional.normalize(self.queue, dim=0)

    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

  @torch.no_grad()
  def _momentum_update_key_encoder(self):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
      param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

  @torch.no_grad()
  def _dequeue_and_enqueue(self, keys):
    batch_size = keys.shape[0]

    ptr = int(self.queue_ptr)
    #assert self.K % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    if ptr + batch_size > self.queue.shape[1]:
      chunk1 = self.queue.shape[1] - ptr
      #chunk2 = (ptr + batch_size) - self.queue.shape[1]
      self.queue[:, ptr:ptr+chunk1] = keys.transpose(0,1)[:, :chunk1]
      self.queue[:, :batch_size-chunk1] = keys.transpose(0,1)[:, chunk1:]
    else:
      self.queue[:, ptr:ptr + batch_size] = keys.transpose(0,1)  # transpose
    ptr = (ptr + batch_size) % self.K  # move pointer

    self.queue_ptr[0] = ptr

  @torch.no_grad()
  def _batch_shuffle_single_gpu(self, x):
    """
    Batch shuffle, for making use of BatchNorm.
    """
    # random shuffle index
    idx_shuffle = torch.randperm(x.shape[0]).cuda()

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    return x[idx_shuffle], idx_unshuffle

  @torch.no_grad()
  def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
    """
    Undo batch shuffle.
    """
    return x[idx_unshuffle]

  def contrastive_loss(self, im_q, im_k):
    # compute query features
    q = self.encoder_q(im_q)  # queries: NxCxT
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # compute key features
    with torch.no_grad():  # no gradient to keys
      # shuffle for making use of BN
      im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

      k = self.encoder_k(im_k_)  # keys: NxCxT
      k = nn.functional.normalize(k, dim=1)  # already normalized

      # undo shuffle
      k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1xT
    l_pos = torch.einsum('nct,nct->nt', [q, k]).unsqueeze(1)
    # negative logits: NxKxT
    l_neg = torch.einsum('nct,ckt->nkt', [q, self.queue.clone().detach()])
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= self.T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda()

    loss = nn.CrossEntropyLoss().cuda()(logits, labels)


    return loss, q, k

  def forward(self, im1, im2):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
    Output:
        loss
    """

    # update the key encoder
    with torch.no_grad():  # no gradient to keys
      self._momentum_update_key_encoder()

    # compute loss
    if self.symmetric:  # asymmetric loss
      loss_12, q1, k2 = self.contrastive_loss(im1, im2)
      loss_21, q2, k1 = self.contrastive_loss(im2, im1)
      loss = loss_12 + loss_21
      k = torch.cat([k1, k2], dim=0)
      q = q1
    else:  # asymmetric loss
      loss, q, k = self.contrastive_loss(im1, im2)

    self._dequeue_and_enqueue(k)


    return q, [loss]

  def sample(self, im1):
    with torch.no_grad():
      q = self.encoder_q(im1)
      q = nn.functional.normalize(q, dim=1)  # already normalized

    return q, [torch.zeros(1)[0].to(im1.device)]


class PatchNCELoss(nn.Module):
  '''
  PatchNCE adapted from https://github.com/taesungp/contrastive-unpaired-translation/blob/afdc8fb027/models/patchnce.py
  '''
  def __init__(self, nce_T=0.07, negs_from_minibatch=False):
    super().__init__()
    self.nce_T = nce_T
    self.negs_from_minibatch = negs_from_minibatch
    self.ce = nn.CrossEntropyLoss()

  def forward(self, feat_q, feat_k):
    ## Positive Logits
    l_pos = torch.einsum('bctw,bctw->bw', feat_q, feat_k).view(-1, 1)

    ## Negative Logits
    if self.negs_from_minibatch:
      l_neg = torch.einsum('wct,xct->wx', feat_q.reshape(-1, feat_q.shape[1], feat_q.shape[2]), feat_k.reshape(-1, feat_k.shape[1], feat_k.shape[2]))[None, :, :]
    else:
      l_neg = torch.einsum('bctw,bctx->bwx', feat_q, feat_k)
    diagonal = torch.eye(l_neg.shape[-1], device=l_neg.device).bool()[None, :, :]
    l_neg.masked_fill_(diagonal, -10.0)
    l_neg = l_neg.flatten(0, 1)

    ## Get Loss
    logits = torch.cat([l_pos, l_neg], dim=-1)/self.nce_T
    labels = torch.zeros(logits.shape[0]).long().to(l_neg.device)
    loss = self.ce(logits, labels)
    return logits, loss

class TemporalPatches(nn.Module):
  '''
  Get Temporal Patches from a 1d sequence
  input: list of feats of the shape B, C, T
  output: list of feats of the shape B, C, 1, W (window_size)
          list of patch_ids to help extract the exact same patches for other feats

  '''
  def __init__(self, num_patches=5, window_size=8, window_hop=5):
    super().__init__()
    self.num_patches = num_patches
    self.window_size = window_size
    self.window_hop = window_hop

    self.mlp_init = False

  def create_mlp(self, feats):
    #m4(m3(m2(m1(feat_qs[0].permute(0, 3, 1, 2).flatten(0, 1)))).squeeze(-1)).shape
    for mlp_id, feat in enumerate(feats):
      C = feat.shape[1]
      T = self.window_size
      num_layers = int(np.log2(T))
      mlp = nn.ModuleList([])

      for l in range(num_layers):
        mlp.append(ConvNormRelu(C, C, downsample=True))
      if num_layers == 0:
        mlp.append(ConvNormRelu(C, C, kernel_size=1, stride=1)) ##
      mlp.append(Transpose([2, 1]))
      mlp.append(nn.Linear(C, C))
      mlp.append(Transpose([2, 1]))

      setattr(self, 'mlp_{}'.format(mlp_id), nn.Sequential(*mlp))
      getattr(self, 'mlp_{}'.format(mlp_id)).to(feat.device)
    self.mlp_init = True

  def filter_feats(self, feats):
    feats_ = []
    for feat in feats:
      B, C, T = feat.shape[0], feat.shape[1], feat.shape[2]
      if T - self.window_size > self.window_hop:
        feats_.append(feat)
    return feats_

  def forward(self, feats, patch_ids=None):
    return_feats = []
    return_ids = []

    feats = self.filter_feats(feats)

    if not self.mlp_init:
      self.create_mlp(feats)

    for idx, feat in enumerate(feats):
      B, C, T = feat.shape[0], feat.shape[1], feat.shape[2]
      starts = torch.arange(0, T-self.window_size, self.window_hop)
      ends = starts + self.window_size

      ## get random patches
      if patch_ids is None:
        patch_id = torch.randperm(starts.shape[0])[:self.num_patches]
      else:
        patch_id = patch_ids[idx]

      feat_patches = torch.zeros(feat.shape[0], feat.shape[1], self.window_size, patch_id.shape[-1]).to(feat.device) # B x C x T (window) x W (num_patches)
      for w, (s, e) in enumerate(zip(starts[patch_id], ends[patch_id])):
        feat_patches[:, :, :, w] = feat[:, :, s:e]

      feat_patches = getattr(self, 'mlp_{}'.format(idx))(feat_patches.permute(0, 3, 1, 2).flatten(0, 1)) ## BxW, C, T
      feat_patches = feat_patches.view(B, -1, feat_patches.shape[-2], feat_patches.shape[-1]).permute(0, 2, 3, 1) # B, C, T, W

      return_feats.append(feat_patches)
      return_ids.append(patch_id)

    return return_feats, return_ids

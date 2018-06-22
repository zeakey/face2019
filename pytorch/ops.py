import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

class NormedLinear(nn.Module):
  def __init__(self, feat_dim=512, num_class=10572, radius=10):
    super(NormedLinear, self).__init__()
    self.num_class = num_class
    self.feat_dim = feat_dim
    self.radius = float(radius)
    # learning rate of radius
    self.weight = nn.Parameter(torch.randn(self.num_class, self.feat_dim))
  
  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)

  def forward(self, x):
    weight_norm = torch.nn.functional.normalize(self.weight, p=2, dim=1)
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)

    x_norm = x_norm * self.radius
    # see https://github.com/pytorch/pytorch/blob/372d1d67356f054db64bdfb4787871ecdbbcbe0b/torch/nn/modules/linear.py#L55
    return torch.nn.functional.linear(x_norm, weight_norm)

# Normalized Linear with center exclusive
class ExclusiveLinear(nn.Module):
  def __init__(self, feat_dim=512, num_class=10572, norm_data=False, radius=32):
    super(ExclusiveLinear, self).__init__()
    self.num_class = num_class
    self.feat_dim = feat_dim
    self.norm_data = norm_data
    self.radius = float(radius)
    self.weight = nn.Parameter(torch.randn(self.num_class, self.feat_dim))

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)

  def forward(self, x):
    # normalize features and weight
    weight_norm = torch.nn.functional.normalize(self.weight, p=2, dim=1)
    cos = torch.mm(weight_norm, weight_norm.t())
    cos.clamp(-1, 1)
    cos1 = cos.detach() # used to index nearest neighbour
    cos1.scatter_(1, torch.arange(self.num_class).view(-1, 1).long().cuda(), -100) # fill diagonal with -100
    _, indices = torch.max(cos1, dim=0)
    label = torch.zeros((self.num_class, self.num_class)).cuda()
    label.scatter_(1, indices.view(-1, 1).long(), 1) # fill with 1
    exclusive_loss = torch.dot(cos.view(cos.numel()), label.view(label.numel())) / self.num_class # average
    if self.norm_data:
      x = torch.nn.functional.normalize(x, p=2, dim=1)
      x = x * self.radius
    # see https://github.com/pytorch/pytorch/blob/372d1d67356f054db64bdfb4787871ecdbbcbe0b/torch/nn/modules/linear.py#L55
    return torch.nn.functional.linear(x, weight_norm), exclusive_loss

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, min_lambda=5, lambda_base=1000, gamma=0.12, power=-1):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features,out_features))
        # options
        self.min_lambda = min_lambda
        self.lambda_base = lambda_base
        self.gamma = gamma
        self.power = power
        self.m = m
        self.iter = 0
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]
        self.reset_parameters()

    def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, target):
      self.iter = self.iter + 1
      self.lamb = max(self.min_lambda, 
                      self.lambda_base / (1 + self.gamma * self.iter)**-self.power)

      x = input   # size=(B,F)    F is feature len
      w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

      ww = w.renorm(2,1,1e-5).mul(1e5)
      xlen = x.pow(2).sum(1).pow(0.5) # size=B
      wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

      cos_theta = x.mm(ww) # size=(B,Classnum)
      cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
      cos_theta = cos_theta.clamp(-1,1)

      cos_m_theta = self.mlambda[self.m](cos_theta)
      theta = cos_theta.acos()
      k = (self.m * theta / 3.14159265).floor()
      n_one = k*0.0 - 1
      phi_theta = (n_one**k) * cos_m_theta - 2*k

      cos_theta = cos_theta * xlen.view(-1,1)
      phi_theta = phi_theta * xlen.view(-1,1)

      index = cos_theta * 0.0
      index.scatter_(1, target.view(-1, 1), 1)
      index = index.byte()

      output = cos_theta * 1.0 #size=(B,Classnum)
      output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
      output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
      return (output, self.lamb) # size=(B,Classnum,2)

# AngleLinear with Center Exclusive
class ExclusiveAngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, min_lambda=5, lambda_base=1000, gamma=0.12, power=-1):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features,out_features))
        # options
        self.min_lambda = min_lambda
        self.lambda_base = lambda_base
        self.gamma = gamma
        self.power = power
        self.m = m
        self.iter = 0
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]
        self.reset_parameters()

    def reset_parameters(self):
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, target):
      self.iter = self.iter + 1
      self.lamb = max(self.min_lambda,
                      self.lambda_base / (1 + self.gamma * self.iter)**-self.power)

      x = input   # size=(B,F)    F is feature len
      w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

      ww = w.renorm(2,1,1e-5).mul(1e5)
      xlen = x.pow(2).sum(1).pow(0.5) # size=B
      wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

      cos_theta = x.mm(ww) # size=(B,Classnum)
      cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
      cos_theta = cos_theta.clamp(-1,1)

      cos_m_theta = self.mlambda[self.m](cos_theta)
      theta = cos_theta.acos()
      k = (self.m * theta / 3.14159265).floor()
      n_one = k*0.0 - 1
      phi_theta = (n_one**k) * cos_m_theta - 2*k

      cos_theta = cos_theta * xlen.view(-1,1)
      phi_theta = phi_theta * xlen.view(-1,1)

      index = cos_theta * 0.0
      index.scatter_(1, target.view(-1, 1), 1)
      index = index.byte()

      output = cos_theta * 1.0 #size=(B,Classnum)
      output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
      output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

      # weight exclusive
      weight_norm = torch.nn.functional.normalize(self.weight, p=2, dim=1)
      cos = torch.mm(weight_norm, weight_norm.t())
      cos.clamp(-1, 1)
      cos1 = cos.detach() # used to index nearest neighbour
      cos1.scatter_(1, torch.arange(self.num_class).view(-1, 1).long().cuda(), -100) # fill diagonal with -100
      _, indices = torch.max(cos1, dim=0)
      label = torch.zeros((self.num_class, self.num_class)).cuda()
      label.scatter_(1, indices.view(-1, 1).long(), 1) # fill with 1
      exclusive_loss = torch.dot(cos.view(cos.numel()), label.view(label.numel())) / self.num_class # average

      return (output, exclusive_loss, self.lamb) # size=(B,Classnum,2)

class SampleExclusiveLoss(nn.Module):
  def __init__(self):
    super(SampleExclusiveLoss, self).__init__()
    self.it = 0

  def forward(self, x, target):
    self.it += 1
    batch_size = x.size()[0]
    target = target.view(-1,1) #size=(B,1)
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    cos_sim = torch.matmul(x_norm, torch.t(x_norm))
    cos_sim1 = cos_sim.detach() # only used to find nearest neighbour
    label = target.view(-1, 1).repeat(1, batch_size) # [batch_size x batch_size]
    label = (label == label.t())
    cos_sim1[label] = -100
    cos_sim1.scatter_(1, torch.arange(batch_size).view(-1, 1).long().cuda(), -100) # fill diagonal with -100
    _, indices = torch.max(cos_sim1, dim=0) # find the largest of each row
    label = torch.zeros((batch_size, batch_size)).cuda()
    label.scatter_(1, indices.view(-1, 1).long(), 1) # fill with 1
    loss = torch.sum(torch.mul(label, cos_sim)) / batch_size # NOTE here we use cos_sim (not cos_sim1) that is traced by the gradient graph
    return loss

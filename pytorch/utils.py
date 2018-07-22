import torch
import numpy as np
from PIL import Image
import os, sys
from os.path import isfile
from scipy.io import savemat
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnelingLR(_LRScheduler):
  def __init__(self, optimizer, min_lr=0.01, max_lr=0.1, cycle_length=250, cycle_length_decay=2,
               cycle_magnitude_decay=0.5, last_epoch=-1):
    self.max_lr = max_lr # unreferenced
    self.min_lr = min_lr
    self.cycle_length = cycle_length
    self.cycle_length_decay = cycle_length_decay
    self.cycle_magnitude_decay = cycle_magnitude_decay
    # initiate
    self.iter = -1
    self.cycle = 0
    self.base = float(1.0)
    self.current_cycle_length = self.cycle_length
    super(CosineAnnelingLR, self).__init__(optimizer, last_epoch)
  def step(self, epoch=None):
    self.iter = self.iter + 1
    if self.iter == self.current_cycle_length:
      self.iter = 0
      self.current_cycle_length = int(self.current_cycle_length * self.cycle_length_decay)
      self.base = self.base * self.cycle_magnitude_decay
    if epoch is None:
      epoch = self.last_epoch + 1
    self.last_epoch = epoch
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr
  def get_lr(self):
    return [self.min_lr + (base_lr * self.base - self.min_lr) *
           (1 + math.cos(math.pi * self.iter / self.current_cycle_length)) / 2
           for base_lr in self.base_lrs]

def test_lfw(model, imglist, test_transform, cache_fn):
  model.eval() # switch to evaluate mode
  features = np.zeros((len(imglist), 512 * 2), dtype=np.float32)
  with torch.no_grad():
    for idx, i in enumerate(imglist):
      assert isfile(i), "Image %s doesn't exist." % i
      im = Image.open(i).convert('RGB')
      data0 = test_transform(im)
      data1 = test_transform(im.transpose(Image.FLIP_LEFT_RIGHT))
      # add extra axis ahead
      data0 = data0.unsqueeze(0)
      data1 = data1.unsqueeze(0)
      data = torch.cat((data0, data1), dim=0)
      data = data.cuda()
      feature = model(data)
      feature = feature.cpu()
      feature = feature.numpy().flatten()
      features[idx, :] = feature
    from test_lfw import fold10
    lfw_acc = fold10(features, cache_fn=cache_fn+'lfw-acc.txt')
    savemat(cache_fn+'-features.mat', dict({"features": features}))
    return lfw_acc

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename):
    torch.save(state, filename)

def str2bool(x):
  x = str(x).lower()
  if x == "1" or x == "true" or x == "yes" or x == "y":
    return True
  elif x == "0" or x == "false" or x == "no" or x == "n":
    return False
  else:
    raise ValueError("Invalid bool value %s" % x)

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()

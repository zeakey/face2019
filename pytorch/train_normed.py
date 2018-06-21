from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
torch.backends.cudnn.bencmark = True

import os, sys, random, datetime, time
from os.path import isdir, isfile, isdir, join, dirname, abspath
import argparse
import numpy as np
from PIL import Image
from scipy.io import savemat

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, 'tmp')
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)

parser = argparse.ArgumentParser(description='PyTorch Implementation of HED.')
parser.add_argument('--bs', type=int, help='batch size', default=600)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=0.1)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=18)
parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
parser.add_argument('--wd', type=float, help='weight decay', default=5e-4)
parser.add_argument('--maxepoch', type=int, help='max epoch', default=30)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--train', type=int, help='train or not', default=1)
parser.add_argument('--cuda', type=int, help='cuda', default=1)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--checkpoint', type=str, help='checkpoint prefix', default="normed")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# datasets
parser.add_argument('--casia', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()

assert isfile(args.lfwlist)
assert isdir(args.lfw)
assert args.cuda == 1
args.checkpoint = join(TMP_DIR, args.checkpoint)
if args.train == 0:
  args.train = False
elif args.train == 1:
  args.train = True
else:
  raise ValueError("args.train must be ether 0 or 1")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
if args.train:
  print("Pre-loading training data...")
  train_dataset = datasets.ImageFolder(
    args.casia,
    transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ])
  )
  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.bs, shuffle=True,
    num_workers=12, pin_memory=True
  )
  print("Done!")

# transforms for LFW testing data
test_transform = transforms.Compose([
  transforms.ToTensor(),
  normalize
])

# model and optimizer
from models import Normed
print("Loading model...")
model = Normed(num_class=10575)
print("Done!")

# optimizer related
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

if args.cuda:
  print("Transporting model to GPU(s)...")
  model.cuda()
  print("Done!")

def train_epoch(train_loader, model, optimizer, epoch):
  # recording
  losses = AverageMeter()
  top1 = AverageMeter()
  batch_time = AverageMeter()
  train_record = np.zeros((len(train_loader), 2), np.float32) # loss top1acc
  # exclusive loss weight
  #exclusive_weight = float(epoch + 1) ** 2 / float(1000)
  exclusive_weight = 10
  # switch to train mode
  model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    start_time = time.time()
    if args.cuda:
      data = data.cuda()
      label = label.cuda(non_blocking=True)
    prob = model(data)
    loss = criterion(prob, label)
    # measure accuracy and record loss
    prec1, prec5 = accuracy(prob, label, topk=(1, 5))
    losses.update(loss.item(), data.size(0))
    top1.update(prec1[0], data.size(0))
    # clear cached gradient
    optimizer.zero_grad()
    # backward gradient
    loss.backward()
    # update parameters
    optimizer.step()
    batch_time.update(time.time() - start_time)
    if batch_idx % args.print_freq == 0:
      print("Epoch %d/%d Batch %d/%d, (sec/batch: %.2fsec): loss=%.3f, acc1=%.3f" % \
      (epoch, args.maxepoch, batch_idx, len(train_loader), batch_time.val, losses.val,  top1.val))
    train_record[batch_idx, :] = np.array([losses.avg, top1.avg / float(100)])
  return train_record

def test_lfw(model, imglist, epoch):
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
      if args.cuda:
        data = data.cuda()
      feature = model(data)
      if args.cuda:
        feature = feature.cpu()
      feature = feature.numpy().flatten()
      features[idx, :] = feature
    from test_lfw import fold10
    lfw_acc = fold10(features, cache_fn=args.checkpoint + "-epoch%d-lfw-acc.txt" % epoch)
    savemat(args.checkpoint + "-epoch%d-features.mat" % epoch, dict({"features": features}))
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

def main():
  lfw_acc_history = np.zeros((args.maxepoch, ), np.float32)
  for epoch in range(args.maxepoch):
    scheduler.step() # will adjust learning rate
    if args.train:
      if epoch == 0:
        train_record = train_epoch(train_loader, model, optimizer, epoch)
      else:
        train_record = np.vstack((train_epoch(train_loader, model, optimizer, epoch), train_record))
      save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
      }, filename=args.checkpoint + "-epoch%d.pth" % epoch)
    # prepare data for testing
    with open(args.lfwlist, 'r') as f:
      imglist = f.readlines()
    imglist = [join(args.lfw, i.rstrip()) for i in imglist]
    lfw_acc_history[epoch] = test_lfw(model, imglist, epoch)
    print("Epoch %d best LFW accuracy is %.5f." % (epoch, lfw_acc_history.max()))
  if args.train:
    savemat(args.checkpoint + '-record(max-acc=%.5f).mat' % lfw_acc_history.max(),
            dict({"train_record": train_record,
                  "lfw_acc_history": lfw_acc_history}))
    plt.plot(train_record[:, 0]) # loss
    plt.plot(train_record[:, 1]) # top1acc
    plt.plot(np.arange(0, train_record.shape[0], train_record.shape[0] / args.maxepoch), lfw_acc_history)
    plt.legend(['loss', 'Training-Acc', 'LFW-Acc (max=%.5f)' % lfw_acc_history.max()])
  else:
    savemat(args.checkpoint + '-record(max-acc=%.5f).mat' % lfw_acc_history.max(),
            dict({"lfw_acc_history": lfw_acc_history}))
    plt.plot(lfw_acc_history)
    plt.legend(['LFW-Accuracy (max=%.5f)' % lfw_acc_history.max()])
  plt.savefig(args.checkpoint + 'record.pdf')

if __name__ == '__main__':
  main()


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

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, 'tmp')
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)

parser = argparse.ArgumentParser(description='PyTorch Implementation of HED.')
parser.add_argument('--bs', type=int, help='batch size', default=512)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=0.1)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--maxepoch', type=int, help='max epoch', default=30)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=18)
parser.add_argument('--gamma', type=float, help='gamma', default=0.1)
parser.add_argument('--wd', type=float, help='weight decay', default=5e-4)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--train', type=int, help='train or not', default=1)
parser.add_argument('--angle_linear', type=int, help='use angle linear or not', default=1)
parser.add_argument('--cuda', type=int, help='cuda', default=1)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="checkpoint")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# datasets
parser.add_argument('--casia', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()

assert isfile(args.lfwlist)
assert isdir(args.lfw)

if args.cuda == 0:
  args.cuda = False
elif args.cuda == 1:
  args.cuda = True
else:
  raise ValueError("args.cuda must be ether 0 or 1")
  
if args.train == 0:
  args.train = False
elif args.train == 1:
  args.train = True
else:
  raise ValueError("args.cuda must be ether 0 or 1")

if args.angle_linear == 0:
  args.angle_linear = False
elif args.angle_linear == 1:
  args.angle_linear = True
else:
  raise ValueError("args.cuda must be ether 0 or 1")

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
  print("Done! Data have been preloaded^_^")

# transforms for LFW testing data
test_transform = transforms.Compose([
  transforms.ToTensor(),
  normalize
])

# model and optimizer
from models import Sphereface20
print("Loading model...")
model = Sphereface20(num_class=10575)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

if args.stepsize > 0:
  scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

if args.cuda:
  print("Transporting model to GPU(s)...")
  model.cuda()

def train_epoch(train_loader, model, criterion, optimizer, epoch):
  losses = AverageMeter()
  top1 = AverageMeter()
  batch_time = AverageMeter()
  # switch to train mode
  model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    start_time = time.time()
    if args.cuda:
      data = data.cuda()
      label = label.cuda(non_blocking=True)
    output, lamb = model(data, label)
    loss = criterion(output, label)
    # measure accuracy and record loss
    prec1, prec5 = accuracy(output, label, topk=(1, 5))
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
      print("Epoch %d/%d Batch %d/%d, (batch time: %.2fsec): Loss=%.3f, acc1=%.3f, lambda=%.3f" % \
      (epoch, args.maxepoch, batch_idx, len(train_loader), batch_time.val, losses.val, top1.val, lamb))

def test_lfw(model, imglist, filename):
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
      output = model(data)
      if args.cuda:
        output = output.cpu()
      output = output.numpy().flatten()
      features[idx, :] = output
    from test_lfw import fold10
    lfw_acc = fold10(features, cache_fn=filename+".txt")
    savemat(filename, dict({"features": features}))
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
      train_epoch(train_loader, model, criterion, optimizer, epoch)
      save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
      }, filename=join(TMP_DIR, args.checkpoint + "epoch%d" % (epoch) + ".pth"))
    # prepare data for testing
    with open(args.lfwlist, 'r') as f:
      imglist = f.readlines()
    imglist = [join(args.lfw, i.rstrip()) for i in imglist]
    lfw_acc_history[epoch] = test_lfw(model, imglist,
                filename=join(args.checkpoint + "epoch%d" % (epoch) + ".pth-features.mat"))
    print("Epoch %d best LFW accuracy is %.3f." % (epoch, lfw_acc_history.max()))

if __name__ == '__main__':
  main()


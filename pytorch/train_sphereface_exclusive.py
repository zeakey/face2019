# sphereface + center-exclusive
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
from utils import accuracy, test_lfw, AverageMeter, save_checkpoint

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, 'tmp')
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)

parser = argparse.ArgumentParser(description='PyTorch Implementation of HED.')
parser.add_argument('--bs', type=int, help='batch size', default=600)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=0.1)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=6)
parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
parser.add_argument('--wd', type=float, help='weight decay', default=5e-4)
parser.add_argument('--maxepoch', type=int, help='maximal training epoch', default=30)
parser.add_argument('--center_weight', type=float, help='center loss weight', default=5)
parser.add_argument('--exclusive_weight', type=float, help='exclusive loss weight', default=15)
parser.add_argument('--m', type=int, help='m', default=4)
# model parameters
parser.add_argument('--radius', type=float, help='normed data radius', default=10)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--train', type=int, help='train or not', default=1)
parser.add_argument('--cuda', type=int, help='cuda', default=1)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--checkpoint', type=str, help='checkpoint prefix', default="sphereface_center_exclusive")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# datasets
parser.add_argument('--casia', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--num_class', type=int, help='num classes', default=10572)
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()

assert isfile(args.lfwlist)
assert isdir(args.lfw)
assert args.cuda == 1
assert args.center_weight >= 0 and args.exclusive_weight >= 0

args.checkpoint = join(TMP_DIR, args.checkpoint) + "-exclusive_weight%.3f-m=%d" % \
                                     (args.exclusive_weight, args.m)
# manually asign random seed
torch.manual_seed(666)

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

print("Loading model...")
from models import Sphere20CenterExclusive
model = Sphere20CenterExclusive(num_class=args.num_class, m=args.m)
print("Done!")

# optimizer related
from ops import AngleCenterLoss
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

if args.cuda:
  print("Transporting model to GPU(s)...")
  model.cuda()
  print("Done!")

def train_epoch(train_loader, model, optimizer, epoch):
  # recording
  loss_cls = AverageMeter()
  loss_exc = AverageMeter()
  top1 = AverageMeter()
  batch_time = AverageMeter()
  train_record = np.zeros((len(train_loader), 3), np.float32) # loss exc_loss top1-acc
  # switch to train mode
  model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    it = epoch * len(train_loader) + batch_idx
    start_time = time.time()
    if args.cuda:
      data = data.cuda()
      label = label.cuda(non_blocking=True)
    prob, exclusive_loss, lamb= model(data, label)
    #=====================================================================
    # adjust you loss weight (s) here
    # center_weight = float(args.center_weight * (1 - np.exp(-it * 0.001)))
    # center_weight = float(args.center_weight)
    lamb1 = float(lamb)
    if lamb1 == 0:
      exclusive_weight = 0
    else:
      exclusive_weight = args.exclusive_weight * float(5 / lamb1)
    #=====================================================================
    last_lamb = float(lamb)
    cls_loss = criterion(prob, label)
    loss = cls_loss + exclusive_weight * exclusive_loss
    # measure accuracy and record loss
    prec1, prec5 = accuracy(prob, label, topk=(1, 5))
    loss_cls.update(cls_loss.item(), data.size(0))
    loss_exc.update(exclusive_loss.item(), data.size(0))
    top1.update(prec1[0], data.size(0))
    # clear cached gradient
    optimizer.zero_grad()
    # backward gradient
    loss.backward()
    # update parameters
    optimizer.step()
    batch_time.update(time.time() - start_time)
    if batch_idx % args.print_freq == 0:
      print("Epoch %d/%d Batch %d/%d, (sec/batch: %.2fsec): loss_cls=%.3f (* 1), loss-exc=%.5f (* %f), lamb=%.3f, acc1=%.3f" % \
      (epoch, args.maxepoch, batch_idx, len(train_loader), batch_time.val, loss_cls.val, \
      loss_exc.val, exclusive_weight, lamb, top1.val))
    train_record[batch_idx, :] = np.array([loss_cls.avg, loss_exc.avg, top1.avg / float(100)])
  return train_record

def main():
  lfw_acc_history = np.zeros((args.maxepoch, ), np.float32)
  for epoch in range(args.maxepoch):
    scheduler.step() # will adjust learning rate
    if args.train:
      if epoch == 0:
        train_record = train_epoch(train_loader, model, optimizer, epoch)
      else:
        train_record = np.vstack((train_record, train_epoch(train_loader, model, optimizer, epoch)))
      save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
      }, filename=args.checkpoint + "-epoch%d.pth" % epoch)
    # prepare data for testing
    with open(args.lfwlist, 'r') as f:
      imglist = f.readlines()
    imglist = [join(args.lfw, i.rstrip()) for i in imglist]
    lfw_acc_history[epoch] = test_lfw(model, imglist, test_transform, args.checkpoint+'-epoch%d' % epoch)
    print("Epoch %d best LFW accuracy is %.5f." % (epoch, lfw_acc_history.max()))
  if args.train:
    savemat(args.checkpoint + '-record(max-acc=%.5f).mat' % lfw_acc_history.max(),
            dict({"train_record": train_record,
                  "lfw_acc_history": lfw_acc_history}))
    iter_per_epoch = train_record.shape[0] / args.maxepoch # iterations per epoch
    plt.plot(train_record[:, 0] / 10, 'r') # loss cls / 10
    plt.plot(train_record[:, 1], 'g') # loss exclusive
    plt.plot(train_record[:, 2], 'c') # top1 acc
    plt.plot(np.arange(0, train_record.shape[0], iter_per_epoch), lfw_acc_history, 'm')
    max_acc_epoch = np.argmax(lfw_acc_history)
    plt.plot(max_acc_epoch * iter_per_epoch, lfw_acc_history[max_acc_epoch], 'm*', markersize=12)

    plt.legend(['cross entropy loss (*0.1)', 'exclusive loss', 'center loss', 'Training-Acc', 'LFW-Acc (max=%.5f)' % lfw_acc_history.max()])
  else:
    savemat(args.checkpoint + '-record(max-acc=%.5f).mat' % lfw_acc_history.max(),
            dict({"lfw_acc_history": lfw_acc_history}))
    plt.plot(lfw_acc_history)
    plt.legend(['LFW-Accuracy (max=%.5f)' % lfw_acc_history.max()])
  plt.ylim(0, 4)
  plt.grid(True)
  plt.title("center-loss $\\times$ %.3f + exclusive-loss $\\times$ %.3f" % (args.center_weight, args.exclusive_weight))
  plt.savefig(args.checkpoint + '-record.pdf')

if __name__ == '__main__':
  main()


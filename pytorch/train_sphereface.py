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
import argparse, datetime
import numpy as np
from PIL import Image
from scipy.io import savemat
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import accuracy, test_lfw, AverageMeter, save_checkpoint

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
# model parameters
parser.add_argument('--m', type=int, help='m', default=4)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--train', type=int, help='train or not', default=1)
parser.add_argument('--angle_linear', type=int, help='use angle linear or not', default=1)
parser.add_argument('--cuda', type=int, help='cuda', default=1)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="sphereface20")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# datasets
parser.add_argument('--casia', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--num_class', type=int, help='num classes', default=10572)
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()
args.checkpoint = join(TMP_DIR, args.checkpoint) + "-m=%d-" % args.m + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
print("Checkpoint directory: %s" % args.checkpoint)
if not isdir(args.checkpoint):
  os.makedirs(args.checkpoint)

# check and assertations
assert isfile(args.lfwlist) and isdir(args.lfw)

# manually asign random seed
torch.manual_seed(666)

# data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
  args.casia,
  transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
  ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.bs, shuffle=True,
    num_workers=12, pin_memory=True
)
# transforms for LFW testing data
test_transform = transforms.Compose([
  transforms.ToTensor(),
  normalize
])
# model
print("Loading model...")
from models import Sphereface20
model = Sphereface20(num_class=args.num_class, m=args.m).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

def train_epoch(train_loader, model, criterion, optimizer, epoch):
  losses = AverageMeter()
  top1 = AverageMeter()
  batch_time = AverageMeter()
  # switch to train mode
  model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    start_time = time.time()
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
      }, filename=join(args.checkpoint, "epoch%d" % (epoch) + ".pth"))
    # prepare data for testing
    with open(args.lfwlist, 'r') as f:
      imglist = f.readlines()
    imglist = [join(args.lfw, i.rstrip()) for i in imglist]
    lfw_acc_history[epoch] = test_lfw(model, imglist, test_transform, join(args.checkpoint, 'epoch%d' % epoch))
    print("Epoch %d best LFW accuracy is %.5f." % (epoch, lfw_acc_history.max()))

if __name__ == '__main__':
  main()


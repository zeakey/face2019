# center loss: nearly the same with the original paper
# we use angular distance instead of Euclidean distance.
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils import accuracy, test_lfw, AverageMeter, save_checkpoint, str2bool
import math

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
parser.add_argument('--gamma', type=float, help='gamma', default=0.1)
parser.add_argument('--wd', type=float, help='weight decay', default=5e-4)
parser.add_argument('--maxepoch', type=int, help='maximal training epoch', default=30)
# model parameters
parser.add_argument('--radius', type=float, help='radius', default=15)
parser.add_argument('--l2filter', type=str, help='filter samples based on l2', default="True")
parser.add_argument('--center_weight', type=float, help='center loss weight', default=0.1)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--train', type=str, help='set to false to test lfw acc only', default="true")
parser.add_argument('--cuda', type=int, help='cuda', default=1)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--checkpoint', type=str, help='checkpoint prefix', default="centerloss-original")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
parser.add_argument('--parallel', action='store_true')
# datasets
parser.add_argument('--casia', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--num_class', type=int, help='num classes', default=10572)
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()

assert isfile(args.lfwlist) and isdir(args.lfw)
assert args.cuda == 1

args.train = str2bool(args.train)
args.l2filter = str2bool(args.l2filter)
args.checkpoint = join(TMP_DIR, args.checkpoint) + "-center_weight-%.1f-radius%.1f-" % (args.center_weight, args.radius) + \
                                     datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
print("Checkpoint directory: %s" % args.checkpoint)

if not isdir(args.checkpoint):
  os.makedirs(args.checkpoint)

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
    num_workers=24, pin_memory=True
  )
  print("Done!")

# transforms for LFW testing data
test_transform = transforms.Compose([
  transforms.ToTensor(),
  normalize
])

class CenterLoss(nn.Module):
  def __init__(self, feat_dim=512, num_class=10572):
    super(CenterLoss, self).__init__()
    self.num_class = num_class
    self.feat_dim = feat_dim
    self.weight = nn.Parameter(torch.randn(self.num_class, self.feat_dim))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)

  def forward(self, x, target):
    # normalize features and weight
    normed_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1) # 10572 x feat_dim
    normed_data = torch.nn.functional.normalize(x, p=2, dim=1) # bs x feat_dim
    # center loss
    centers = torch.index_select(normed_weight, dim=0, index=target) # bs x feat_dim
    # we would like to "minimize" the angle between sample and corresponding center,
    # so we have to "maximize" its cosine similarity
    cos = torch.sum(torch.mul(normed_data, centers), dim=1)
    # NOTE the minus, we maximize the cosine term
    center_loss = -torch.mean(cos)
    if center_loss > 1 or center_loss < -1:
      raise ValueError("?")
    return center_loss

from models import Resnet20
from ops import NormedLinear

class Model(nn.Module):
  def __init__(self, dim=512, num_class=10572, norm_data=True, radius=15):
    super(Model, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = NormedLinear(dim, num_class, radius)
  def forward(self, x):
    feature = self.base(x) # batch_size x 512
    if self.training:
      prob = self.fc6(feature)
      # return predicted probabilities and feature
      return prob, feature
    else:
      return feature

# model and optimizer
print("Loading model...")
model = Model(num_class=args.num_class, norm_data=True, radius=args.radius)
print("Done!")

# optimizer related
criterion = nn.CrossEntropyLoss(reduce = not args.l2filter)
center_loss = CenterLoss(feat_dim=512, num_class=args.num_class)
center_loss.cuda()

# optimizer of the base model
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
# optimizer of the center loss module
# we should use a larger weight decay, because the centers will always get far away the origin.
optimizer_center = torch.optim.SGD(center_loss.parameters(), lr= 5 * args.lr, weight_decay=args.wd * 2, momentum=args.momentum)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
scheduler_center = lr_scheduler.StepLR(optimizer_center, step_size=args.stepsize, gamma=args.gamma)

if args.cuda:
  print("Transporting model to GPU(s)...")
  model.cuda()
  print("Done!")
if args.parallel:
  model = nn.DataParallel(model)

def train_epoch(train_loader, model, optimizer, epoch):
  # recording
  loss_cls = AverageMeter()
  loss_center = AverageMeter()
  top1 = AverageMeter()
  batch_time = AverageMeter()
  train_record = np.zeros((len(train_loader), 4), np.float32) # loss, exc_loss, top1-acc, lr
  # switch to train mode
  model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    it = epoch * len(train_loader) + batch_idx
    # exclusive loss weight
    warmup = 0
    if epoch < warmup:
      center_weight = 0
    else:
      center_weight = args.center_weight
    start_time = time.time()
    if args.cuda:
      data = data.cuda()
      label = label.cuda(non_blocking=True)
    prob, feature = model(data)
    ##########################################
    if args.l2filter:
      bs = feature.size(0)
      feature_l2 = torch.norm(feature, p=2, dim=1).detach()
      feature_l2 = feature_l2.cpu().numpy()
      assert feature_l2.min() > 0
      if False:
        bad, hard = int(bs / 20), int(bs / 2)
        bad_examples = (feature_l2 <= np.sort(feature_l2)[bad]) # bad examples will be eliminated
        # hard examples will be emphasized
        hard_examples = np.logical_and(feature_l2 > np.sort(feature_l2)[bad], feature_l2 < np.sort(feature_l2)[hard])
        normal_examples = np.logical_not(np.logical_or(bad_examples, hard_examples))
        weight = feature_l2.copy()
        weight[normal_examples] = 1
        weight[bad_examples] = 0
        weight[hard_examples] /= weight[hard_examples].max()
        weight[hard_examples] = 1 / weight[hard_examples]
      else:
        num_decay = int(feature_l2.size / 15)
        decay_examples = feature_l2 < np.sort(feature_l2)[num_decay]
        normal_examples = np.logical_not(decay_examples)
        weight = feature_l2.copy()
        weight[normal_examples] = 1
        weight[decay_examples] -= weight[decay_examples].min()
        weight[decay_examples] /= weight[decay_examples].max()
        # weight[decay_examples] = 1 / weight[decay_examples]
      loss = criterion(prob, label)
      loss = torch.mul(loss, torch.from_numpy(weight).cuda()).mean()
    else:
      loss = criterion(prob, label)
    # calculate center loss
    centerloss = center_loss(feature, label)
    ##########################################
    # measure accuracy and record loss
    prec1, prec5 = accuracy(prob, label, topk=(1, 5))
    loss_cls.update(loss.item(), data.size(0))
    loss_center.update(centerloss.item(), data.size(0))
    top1.update(prec1[0], data.size(0))
    # collect losses
    loss = loss + centerloss * center_weight
    # clear cached gradient
    optimizer.zero_grad()
    optimizer_center.zero_grad()
    # backward gradient
    loss.backward()
    # center_loss.backward()
    # update parameters
    optimizer.step()
    optimizer_center.step()
    ##########################################
    batch_time.update(time.time() - start_time)
    if batch_idx % args.print_freq == 0:
      # analyse center loss parameters and fc parameters
      fc6weight = model.state_dict()['fc6.weight'].detach()
      centers = center_loss.state_dict()['weight'].detach()
      fc6weight = torch.nn.functional.normalize(fc6weight, p=2, dim=1)
      centers = torch.nn.functional.normalize(centers, p=2, dim=1)
      cos = torch.sum(torch.mul(fc6weight, centers), dim=1)
      cos = torch.mean(cos)
      print("Epoch %d/%d Batch %d/%d, (sec/batch: %.2fsec): loss_cls=%.3f (* 1), loss-center=%.5f (* %.4f), acc1=%.3f,"
      "lr=%.3f, fc6-centers-cos=%f" % \
      (epoch, args.maxepoch, batch_idx, len(train_loader), batch_time.val, loss_cls.val,
      loss_center.val, center_weight, top1.val, scheduler.get_lr()[0], cos))
      if args.l2filter:
        plt.scatter(feature_l2, weight)
        plt.title("%dhard-%dbad" % (np.count_nonzero(np.logical_and(weight != 1, weight != 0)), np.count_nonzero(weight==0)))
        plt.savefig(join(args.checkpoint, "Iter%d-feature-l2-vs-weight.jpg" % it))
        plt.close()
    train_record[batch_idx, :] = np.array([loss_cls.avg, loss_center.avg, top1.avg / float(100), scheduler.get_lr()[0]])
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
    # prepare data for testing
    with open(args.lfwlist, 'r') as f:
      imglist = f.readlines()
    imglist = [join(args.lfw, i.rstrip()) for i in imglist]
    lfw_acc_history[epoch] = test_lfw(model, imglist, test_transform, join(args.checkpoint, 'epoch%d' % epoch))
    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'optimizer' : optimizer.state_dict(),
    }, filename=join(args.checkpoint, "epoch%d-lfw%f.pth" % (epoch, lfw_acc_history[epoch])))
    # NOTE we also save the center_loss module, because it has learnable parameters (the centers)
    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': center_loss.state_dict(),
      'optimizer' : optimizer_center.state_dict(),
    }, filename=join(args.checkpoint, "epoch%d-center_loss-lfw%f.pth" % (epoch, lfw_acc_history[epoch])))
    print("Epoch %d best LFW accuracy is %.5f." % (epoch, lfw_acc_history.max()))
  if args.train:
    savemat(join(args.checkpoint, 'record(max-acc=%.5f).mat' % lfw_acc_history.max()),
            dict({"train_record": train_record,
                  "lfw_acc_history": lfw_acc_history}))
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax in axes:
      ax.grid(True)
      ax.hold(True)
    axes[0].plot(train_record[:, 0], 'r') # loss cls
    axes[0].set_title("CELoss")

    axes[1].plot(train_record[:, 1], 'r') # loss exclusive
    axes[1].set_title("ExLoss")

    axes[2].plot(train_record[:, 2], 'r') # top1 acc
    axes[2].set_title("Trn-Acc")

    axes[3].plot(train_record[:, 3], 'r') # LR
    axes[3].set_title("LR")

    axes[4].plot(lfw_acc_history.argmax(), lfw_acc_history.max(), 'r*', markersize=12)
    axes[4].plot(lfw_acc_history, 'r')
    axes[4].set_title("LFW-Acc")

    plt.suptitle("radius=%.1f, max LFW-Acc=%.3f" % (args.radius, lfw_acc_history.max()))
  else:
    savemat(join(args.checkpoint + 'record(max-acc=%.5f).mat' % lfw_acc_history.max()),
            dict({"lfw_acc_history": lfw_acc_history}))
    plt.plot(lfw_acc_history)
    plt.legend(['LFW-Accuracy (max=%.5f)' % lfw_acc_history.max()])
  plt.grid(True)
  plt.title("center-loss$\\times$%.1f" % args.center_weight)
  plt.savefig(join(args.checkpoint, 'record.pdf'))

if __name__ == '__main__':
  main()


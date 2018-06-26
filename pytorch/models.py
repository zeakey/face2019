import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
#from ops import AngleLinear1 as AngleLinear
from ops import AngleLinear, CenterExclusiveAngleLinear, NormedLinear, CenterlossExclusiveLinear
#==================================================================#
# the base architecture
#==================================================================#
class Resnet20(nn.Module):
    def __init__(self, dim=512, bn=False):
        super(Resnet20, self).__init__()
        self.dim = dim
        self.bn = bn
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6, self.dim)

        if self.bn:
          self.bn1_1 = nn.BatchNorm2d(64)
          self.bn1_2 = nn.BatchNorm2d(64)
          self.bn1_3 = nn.BatchNorm2d(64)
          self.bn2_1 = nn.BatchNorm2d(128)
          self.bn2_2 = nn.BatchNorm2d(128)
          self.bn2_3 = nn.BatchNorm2d(128)
          self.bn2_4 = nn.BatchNorm2d(128)
          self.bn2_5 = nn.BatchNorm2d(128)
          self.bn3_1 = nn.BatchNorm2d(256)
          self.bn3_2 = nn.BatchNorm2d(256)
          self.bn3_3 = nn.BatchNorm2d(256)
          self.bn3_4 = nn.BatchNorm2d(256)
          self.bn3_5 = nn.BatchNorm2d(256)
          self.bn3_6 = nn.BatchNorm2d(256)
          self.bn3_7 = nn.BatchNorm2d(256)
          self.bn3_8 = nn.BatchNorm2d(256)
          self.bn3_9 = nn.BatchNorm2d(256)
          self.bn4_1 = nn.BatchNorm2d(512)
          self.bn4_2 = nn.BatchNorm2d(512)
          self.bn4_3 = nn.BatchNorm2d(512)

    def forward(self, x):
        if self.bn:
            x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
            x = x + self.relu1_3(self.bn1_3(self.conv1_3(self.relu1_2(self.bn1_2(self.conv1_2(x))))))

            x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
            x = x + self.relu2_3(self.bn2_3(self.conv2_3(self.relu2_2(self.bn2_2(self.conv2_2(x))))))
            x = x + self.relu2_5(self.bn2_5(self.conv2_5(self.relu2_4(self.bn2_4(self.conv2_4(x))))))

            x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
            x = x + self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(x))))))
            x = x + self.relu3_5(self.bn3_5(self.conv3_5(self.relu3_4(self.bn3_4(self.conv3_4(x))))))
            x = x + self.relu3_7(self.bn3_7(self.conv3_7(self.relu3_6(self.bn3_6(self.conv3_6(x))))))
            x = x + self.relu3_9(self.bn3_9(self.conv3_9(self.relu3_8(self.bn3_8(self.conv3_8(x))))))

            x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
            x = x + self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(x))))))
        else:
            x = self.relu1_1(self.conv1_1(x))
            x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

            x = self.relu2_1(self.conv2_1(x))
            x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
            x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

            x = self.relu3_1(self.conv3_1(x))
            x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
            x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
            x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
            x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

            x = self.relu4_1(self.conv4_1(x))
            x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        return x

#==================================================================#
# baseline model with cross-entropy loss
#==================================================================#
class Baseline(nn.Module):
  def __init__(self, dim=512, num_class=10572):
    super(Baseline, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = nn.Linear(dim, num_class)
  def forward(self, x):
    x = self.base(x)
    if self.training:
      x = self.fc6(x)
    return x

#==================================================================#
# cross-entropy loss with normalized data and weights
#==================================================================#
class Normed(nn.Module):
  def __init__(self, dim=512, num_class=10572, radius=10):
    super(Normed, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.radius = radius
    self.fc6 = NormedLinear(dim, num_class, radius=self.radius)
  def forward(self, x):
    x = self.base(x)
    if self.training:
      x = self.fc6(x)
    return x

#==================================================================#
# sphereface20 with angular margin inner-product (AngleLinear) layer
#==================================================================#
class Sphereface20(nn.Module):
  def __init__(self, dim=512, num_class=10572):
    super(Sphereface20, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = AngleLinear(dim, num_class, gamma=0.06)
  def forward(self, x, target=None):
    x = self.base(x)
    if self.training:
      x, lamb = self.fc6(x, target)
      return x, lamb
    else:
      return x

#==================================================================#
# Sphereface20 (with margin inner-product) + sample exclusive
#==================================================================#
class SampleExclusive(nn.Module):
  def __init__(self, dim=512, num_class=10572):
    super(SampleExclusive, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = AngleLinear(dim, num_class, gamma=0.06)
  def forward(self, x, target=None):
    feature = self.base(x)
    if self.training:
      prob, lamb = self.fc6(feature, target)
      return prob, feature, lamb
    else:
      return feature

#==================================================================#
# Sphereface20 (with margin inner-product) + center exclusive
#==================================================================#
class Sphere20CenterExclusive(nn.Module):
  def __init__(self, dim=512, num_class=10572, m=4):
    super(Sphere20CenterExclusive, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = CenterExclusiveAngleLinear(dim, num_class, gamma=0.06, m=m)
  def forward(self, x, target=None):
    feature = self.base(x)
    if self.training:
      prob, exclusive_loss, lamb = self.fc6(feature, target)
      return prob, exclusive_loss, lamb
    else:
      return feature

#==================================================================#
# center exclusive
#==================================================================#
class CenterExclusive(nn.Module):
  def __init__(self, dim=512, num_class=10572, norm_data=True, radius=10):
    super(CenterExclusive, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = ExclusiveLinear(dim, num_class, norm_data=norm_data, radius=radius)
  def forward(self, x):
    feature = self.base(x)
    if self.training:
      prob, exclusive_loss = self.fc6(feature)
      return prob, exclusive_loss
    else:
      return feature

#==================================================================#
# centerloss + center-exclusve
#==================================================================#
class CenterLossExclusive(nn.Module):
  def __init__(self, dim=512, num_class=10572, norm_data=True, radius=10):
    super(CenterLossExclusive, self).__init__()
    self.num_class = num_class
    self.base = Resnet20()
    self.fc6 = ExclusiveLinear(dim, num_class, norm_data=norm_data, radius=radius)
  def forward(self, x):
    feature = self.base(x) # batch_size x 512
    if self.training:
      prob, exclusive_loss = self.fc6(feature)
      # the returned feature with be used to calculate center-loss
      return prob, feature, exclusive_loss
    else:
      return feature

#==================================================================#
# centerloss + center-exclusve
# integrate center-loss and exclusive-loss into one operator
# and reuse the linear weights as centers
#==================================================================#
class CenterLossExclusive1(nn.Module):
  def __init__(self, dim=512, num_class=10572, norm_data=True, radius=10, bn=False):
    super(CenterLossExclusive1, self).__init__()
    self.num_class = num_class
    self.base = Resnet20(bn=bn)
    self.fc6 = CenterlossExclusiveLinear(dim, num_class,
                        norm_data=norm_data, radius=radius)
  def forward(self, x, target=None):
    feature = self.base(x) # batch_size x 512
    if self.training:
      assert target is not None
      prob, exclusive_loss, centerloss = self.fc6(feature, target)
      return prob, feature, exclusive_loss, centerloss
    else:
      return feature

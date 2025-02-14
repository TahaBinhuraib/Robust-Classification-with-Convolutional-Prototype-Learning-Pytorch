import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ModifiedResNet(nn.Module):
    def __init__(self, num_hidden_units, s, *args, **kwargs):
        super(ModifiedResNet, self).__init__()
        self.scale = s
        self.resnet = torchvision.models.resnet18(pretrained=True)
        #self.resnet = torchvision.models.resnet34(pretrained=True)
        num_ftrs = self.resnet.fc.out_features

        self.reduce_to_hidden = nn.Linear(num_ftrs, num_hidden_units)
        self.dce_loss = dce_loss(10, num_hidden_units)

    def forward(self, x):
        x = self.resnet(x)
        x1 = F.relu(self.reduce_to_hidden(x))
        centers, x = self.dce_loss(x1)
        output = F.log_softmax(self.scale * x, dim=1)
        return x1, centers, x, output


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):

        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter( # Prototypes 
            torch.randn(self.feat_dim, self.n_classes), requires_grad=True
        )
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):

        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = features - torch.t(centers)[labels]

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance

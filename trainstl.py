from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.classifier import Net
from models.real_nvp import RealNVP
# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
args = parser.parse_args()

torch.manual_seed(args.seed)

										   
from data import initialize_data, data_transforms,data_brightnesstransform,data_huetransform,data_contrasttransform,data_saturationtransform,data_translate,data_rotate,data_center,data_shear


trainset=torch.utils.data.ConcatDataset([datasets.CIFAR10(root='.',transform=data_transforms,train=True,download=True),
    datasets.CIFAR10(root='.',transform=data_brightnesstransform,train=True),
    datasets.CIFAR10(root='.',transform=data_huetransform,train=True),
    datasets.CIFAR10(root='.',transform=data_contrasttransform,train=True),
    datasets.CIFAR10(root='.',transform=data_saturationtransform,train=True),
    datasets.CIFAR10(root='.',transform=data_translate,train=True),
    datasets.CIFAR10(root='.',transform=data_rotate,train=True),
    datasets.CIFAR10(root='.',transform=data_center,train=True),
    datasets.CIFAR10(root='.',transform=data_shear,train=True)])
   
testset=datasets.CIFAR10(root='.',transform=data_transforms,train=False,download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

num_inputs=64*64*3
num_outputs=10

class MultiNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MultiNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 1000)
        self.non_linear = nn.Tanh()
        self.linear2 = nn.Linear(1000,num_outputs)
        self.generator=RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
        self.generator.load_state_dict(torch.load('stlgen'))


    def forward(self, input):
        input,_=self.generator(input)
        input = input.view(-1, num_inputs) # reshape input to batch x num_inputs
        layer1output = self.linear1(input)
        non_linearoutput=self.non_linear(layer1output)
        output=self.linear2(non_linearoutput)
        return output

multinetwork = MultiNet(num_inputs, num_outputs).cuda()
optimizer3 = optim.Adam(multinetwork.parameters(), lr=args.lr, betas=(0,0.9))


def train3(epoch):
    multinetwork.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer3.zero_grad()
        output = multinetwork(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer3.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test3():
    multinetwork.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = multinetwork(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))





if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train3(epoch)
        test3()

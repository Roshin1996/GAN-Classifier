from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

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
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--g", type=int, default=1, help="generator")
parser.add_argument("--c", type=int, default=1, help="classifier")
args = parser.parse_args()

torch.manual_seed(args.seed)

val_loss=[]




reqtrans=transforms.Compose([transforms.Resize(args.img_size),transforms.CenterCrop(args.img_size),transforms.ToTensor()])
	
val_dataset = datasets.STL10(root='./data/stl10',
                                           transform=reqtrans,
                                           split='test',
                                           download=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=args.batch_size, 
                                           shuffle=False)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from models.classifier import Net
from models.real_nvp import RealNVP
model = Net()
model.load_state_dict(torch.load('model_{}.pth'.format(args.c)))
model=model.cuda()
generator=RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
generator.load_state_dict(torch.load('./savedmodels/checkpoint_num_epochs_200_dataset_stl10_batch_size_64/generator_{}'.format(args.g)))
generator=generator.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    count=0
    with torch.no_grad():
        for data, target in val_loader:
            print(count)
            data, target = Variable(data).cuda(), Variable(target).cuda()
            data,_=generator(data)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            count+=1

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100.*correct/len(val_loader.dataset)))

    val_loss.append(validation_loss)


if __name__ == '__main__':
    validation()
    

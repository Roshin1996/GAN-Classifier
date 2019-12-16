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
args = parser.parse_args()

torch.manual_seed(args.seed)

train_loss=[]
val_loss=[]




reqtrans=transforms.Compose([transforms.Resize(args.img_size),transforms.CenterCrop(args.img_size),transforms.ToTensor()])
	
train_dataset = datasets.STL10(root='./data/stl10',
                                           transform=reqtrans,
                                           split='train',
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size, 
                                           shuffle=True)
										   
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
model = Net().cuda()
generator=RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
generator.load_state_dict(torch.load('./savedmodels/generator_3'))
generator=generator.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    epoch_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        data,_=generator(data)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss.append(epoch_loss/len(train_loader))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data,_=generator(data)
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100.*correct/len(val_loader.dataset)))

    val_loss.append(validation_loss)


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validation()
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
        if epoch%10==0:
            for param_group in optimizer.param_groups:
                param_group['lr']*=0.1

    

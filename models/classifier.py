import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 10 # GTSRB as 43 classes

num_outputs = 10 # same for both CIFAR10 and MNIST, both have 10 classes as outputs
num_inputs=32*32*3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 1000)
        self.non_linear = nn.Tanh()
        self.linear2 = nn.Linear(1000,num_outputs)

    def forward(self, input):
        input = input.view(-1, num_inputs) # reshape input to batch x num_inputs
        layer1output = self.linear1(input)
        non_linearoutput=self.non_linear(layer1output)
        output=self.linear2(non_linearoutput)
        return output

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5)
        self.nonlinear1=nn.Tanh()
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=128,kernel_size=5)
        self.nonlinear2=nn.Tanh()
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.linear1=nn.Linear(128*21*21,64)
        self.nonlinear3=nn.Tanh()
        self.linear2=nn.Linear(64,10)
        

    def forward(self, input):
        conv1_op=self.conv1(input)
        nonlinear1_op=self.nonlinear1(conv1_op)
        maxpool1_op=self.maxpool1(nonlinear1_op)
        conv2_op=self.conv2(maxpool1_op)
        nonlinear2_op=self.nonlinear2(conv2_op)
        maxpool2_op=self.maxpool2(nonlinear2_op)
        maxpool2_op=maxpool2_op.view(-1,128*21*21)
        linear1_op=self.linear1(maxpool2_op)
        nonlinear3_op=self.nonlinear3(linear1_op)
        output=self.linear2(nonlinear3_op)
        return output


class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=5)
        self.bn1=nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=3)
        self.bn2=nn.BatchNorm2d(100)
        self.conv3=nn.Conv2d(100,200,kernel_size=3)
        self.bn3=nn.BatchNorm2d(200)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    

    def forward(self, x):
        x=self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x=self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x=self.bn2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x=self.bn3(x)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
		
		

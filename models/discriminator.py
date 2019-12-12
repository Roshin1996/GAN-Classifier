import torch.nn as nn
import numpy as np
from models.spectralnorm import SpectralNorm

class spectralDiscriminator(nn.Module):
	def __init__(self,args):
		super(spectralDiscriminator,self).__init__()
		
		self.d=args.dataset
		self.s=args.img_size/8
		self.cnn = nn.Sequential(
			SpectralNorm(nn.Conv2d(args.channels, 64, 4, stride=2, padding=1)),
			nn.LeakyReLU(0.1),
			SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
			nn.LeakyReLU(0.1),
			SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
			nn.LeakyReLU(0.1),
		)
		
		self.fc = nn.Sequential(
			SpectralNorm(nn.Linear(int(self.s*self.s*256), 128)),
			nn.LeakyReLU(0.1),
			SpectralNorm(nn.Linear(128, 1))
		)
			
	def forward(self, img):
		x = self.cnn(img)
		x = x.view(-1, int(self.s*self.s*256))
		x = self.fc(x)
		return x

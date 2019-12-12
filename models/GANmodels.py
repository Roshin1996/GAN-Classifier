from models.discriminator import spectralDiscriminator
from models.real_nvp import RealNVP
import torch
import numpy as np

class RealNVPGAN:
    def __init__(self,args):
        self.args=args
        self.generator=RealNVP(num_scales=2, in_channels=args.channels, mid_channels=64, num_blocks=8)
        self.discriminator=spectralDiscriminator(args)

        
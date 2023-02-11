import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from collections import defaultdict

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10


class SCPHERE(torch.nn.Module):
    
    def __init__(self, 
        n_gene, 
        n_batch=None, 
        z_dim=2, 
        activation=F.relu, 
        latent_dist='vmf', 
        observation_dist='nb', 
        batch_invariant=False):
        

        super(SCPHERE, self).__init__()
        
        self.n_gene = n_gene
        self.n_batch = n_batch
        self.z_dim = z_dim
        self.activation = activation
        self.latent_dist = latent_dist
        self.observation_dist = observation_dist
        self.batch_invariant = batch_invariant
        self.h_dim = 32
        
        self.x = 
        
        ## encoder ##
        self.fc_e0 = nn.Linear(self.n_gene, h_dim*4)
        self.fc_e1 = nn.Linear(h_dim*4, h_dim*2)
        self.fc_e2 = nn.Linear(h_dim*2, h_dim)
        
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented
        
        ## decoder ##
        self.fc_d0 = nn.Linear(z_dim, h_dim*4)
        self.fc_d1 = nn.Linear(h_dim*4, h_dim*2)
        self.fc_logits = nn.Linear(h_dim*2, self.x.shape[-1])
        
        
        
        
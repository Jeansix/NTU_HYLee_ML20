from data import *
from utils import *
from model import *
from train import *
from inference import *
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# hyperparameters
batch_size = 64
z_dim = 100
lr = 1e-4
n_epoch = 10
# working directory
workspace_dir = './'
if __name__ == "__main__":
    save_dir = os.path.join(workspace_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)

    same_seeds(0)

    # dataloader
    dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    G = Generator(in_dim=z_dim).cuda()
    D = Discriminator(3).cuda()

    # loss criterion
    criterion = nn.BCELoss()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    same_seeds(0)
    # train
    train(G, D, criterion, opt_D, opt_G, dataloader, n_epoch, z_dim, save_dir)
    #inference
    inference(z_dim, save_dir)

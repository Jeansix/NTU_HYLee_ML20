import torch
import os
import torchvision
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt

workspace_dir = './'


def inference(z_dim, save_dir):
    G = Generator(z_dim)
    # load pretrained model
    G.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))
    G.eval()
    G.cuda()
    n_output = 20
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(save_dir, f'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    # show image
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))  # 把channel放到最后一维
    plt.show()


if __name__ == "__main__":
    z_dim = 100
    save_dir = os.path.join(workspace_dir, 'logs')
    inference(z_dim, save_dir)

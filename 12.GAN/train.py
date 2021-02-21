import torch
from torch.autograd import Variable
import torchvision
import os
import matplotlib.pyplot as plt

workspace_dir = "./"


def train(G, D, criterion, opt_D, opt_G, dataloader, n_epoch, z_dim, save_dir):
    """

    :param G: generator
    :param D: discriminator
    :param criterion: loss function
    :param opt_D: optimizer for discriminator
    :param opt_G: optimizer for generator
    :param dataloader: encapsulated dataloader
    :param n_epoch: total epochs
    :param z_dim: dim for vector z
    :param save_dir: directory for saving photos
    :return:
    """
    # set mode to train
    G.train()
    D.train()
    # 随便sample100张图片，作为测试
    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()
            bs = imgs.size(0)

            # Train D,fix G
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            # label
            r_label = torch.ones((bs)).cuda()  # 1 for real images
            f_label = torch.zeros((bs)).cuda()  # 0 for fake images

            # dis
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # compute loss
            r_loss = criterion(r_logit, r_label)  # discriminator给真实的image高分(接近1)
            f_loss = criterion(f_logit, f_label)  # discriminator给生成的的image低分(接近0)
            loss_D = (r_loss + f_loss) / 2

            # update model
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G,fix D
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)
            # compute loss
            loss_G = criterion(f_logit, r_label)  # 要让它很真实样本越像越好，骗过discriminator

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()
            # log
            print(f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)}\
             Loss_D: {loss_D.item(): .4f}Loss_G: {loss_G.item(): .4f}', end='')

        # set mode to evaluation
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # show generated image
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()
        if (e + 1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join(workspace_dir, f'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, f'dcgan_d.pth'))

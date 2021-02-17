import torch


def train(img_dataloader, model, optimizer, criterion, n_epoch):
    """
    训练模型
    :param img_dataloader: 用DataLoader类封装image
    :param model: 定义的模型
    :param optimizer: 定义的优化器
    :param criterion: 定义的损失函数
    :param n_epoch: epoch总数
    :return:
    """
    for epoch in range(n_epoch):
        train_loss = 0
        for data in img_dataloader:
            img = data
            img = img.cuda()  # 加载到gpu

            output_encode, output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()  # reset
            loss.backward()  # back propagation
            optimizer.step()  # update

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch + 1))
            train_loss += loss.item()
        print('epoch [{}/{}], loss:{:.5f}'.format(epoch + 1, n_epoch, train_loss))
    torch.save(model.state_dict(), './checkpoints/last_checkpoint.pth')

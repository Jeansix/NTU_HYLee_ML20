import copy
import torch.nn
from utils import *


def train(model, optimizer, train_loader, val_loader, test_loader, loss_function, epochs, device, model_path,
          int2word_dict_cn, int2word_dict_en):
    """
    训练模型
    :param model: 定义的模型
    :param optimizer: 定义的优化器
    :param train_loader: 封装好的train data
    :param val_loader: 封装好的validation data
    :param test_loader: 封装好的test data
    :param loss_function: 定义的损失函数
    :param epochs: 总epoch数
    :param device: cpu or cuda
    :param model_path: 模型保存路径
    :param int2word_dict_cn: 数字到中文单词的映射字典
    :param int2word_dict_en: 数字到英文单词的映射字典
    :return: best_model:训练得到bleu_score最好的模型
    """
    model.train()  # 模式设置为train
    train_losses = []
    val_losses = []
    bleu_scores = []
    best_bleu_score = 0.0
    for epoch in range(epochs):
        # set mode to train
        model.train()
        # set train_loss to 0
        train_loss = 0.0
        batch_size = 1
        for (sources, targets) in train_loader:
            # load sources and targets to gpu
            batch_size = sources.shape[0]
            sources = sources.to(device)
            targets = targets.to(device)
            # generate outputs
            outputs, preds = model(sources, targets, schedual_sampling(epoch, epochs, 'train'), 'train')
            outputs = outputs[:, 1:].reshape(-1, outputs.size(2))  # skip <BOS>
            targets = targets[:, 1:].reshape(-1)
            # calculate loss
            loss = loss_function(outputs, targets)
            train_loss += loss.item()
            # reset optimizer
            optimizer.zero_grad()
            # calculate gradient
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # clip 防止梯度爆炸
            # update parameters
            optimizer.step()

        train_losses.append(train_loss / len(train_loader) * batch_size)
        # set mode to evaluate
        model.eval()
        with torch.no_grad():
            val_loss, bleu_score, result = test(model, val_loader, loss_function, device, int2word_dict_cn,
                                                int2word_dict_en)
            if bleu_score > best_bleu_score:
                # update max score and best model
                best_bleu_score = bleu_score
                save_model(model, path=model_path, step=epoch)
                best_model = copy.deepcopy(model)
            val_losses.append(val_loss)
            bleu_scores.append(bleu_score)
            test_loss, test_bleu, _ = test(model, test_loader, loss_function, device, int2word_dict_cn,
                                           int2word_dict_en)
        print("epoch: {}/{} | train_loss: {:.3f} | val_loss: {:.3f} val_bleu: {:.3f}  \
        | test_loss: {:.3f} test_bleu: {:.3f}".format(epoch, epochs, train_losses[-1], val_loss, bleu_score, test_loss,
                                                      test_bleu))
    return best_model


def test(model, dataloader, loss_func, device, int2word_dict_cn, int2word_dict_en):
    """
    评估模型
    :param model: 当前模型
    :param dataloader: 当前封装好的data loader
    :param loss_func: 定义的损失函数
    :param device: cpu or cuda
    :param int2word_dict_cn: 数字到中文单词的映射字典
    :param int2word_dict_en: 数字到英文单词的映射字典
    :return: val_loss: 当前data loader的损失
    :return: bleu_score: 当前bleu_score
    :return: result: 保存结果三元组(source,pred,target)
    """
    model.eval()
    loss_sum = 0
    bleu_score = 0.0
    num = 0
    result = []
    for (sources, targets) in dataloader:
        num += sources.size(0)  # batch_size
        sources = sources.to(device)
        targets = targets.to(device)

        outputs, preds = model(sources, targets, schedual_sampling(1, 1, 'test'), 'test')
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_func(outputs, targets)
        loss_sum += loss.item()

        # 将预测结果转化为文字
        targets = targets.view(sources.size(0), -1)
        preds = idx2sentences(preds, int2word_dict_cn)
        sources = idx2sentences(sources, int2word_dict_en)
        targets = idx2sentences(targets, int2word_dict_cn)

        for idx in range(len(targets)):
            result.append((sources[idx], preds[idx], targets[idx]))
        # 计算bleu分数
        bleu_score += computebleu(preds, targets)

    return loss_sum / len(dataloader), bleu_score / num, result


def schedual_sampling(epoch, epochs, mode):
    if mode == 'train':
        return epoch / epochs
    else:
        return 1

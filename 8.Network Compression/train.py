from model import *
from data import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def run_epoch_pruning(dataloader, net, optimizer, criterion, update=True):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空optimizer
        optimizer.zero_grad()
        inputs, labels = batch_data
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()
        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


def run_epoch_distillation(dataloader, teacher_net, student_net, optimizer, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空optimizer
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = hard_labels.cuda()
        with torch.no_grad():
            soft_labels = teacher_net(inputs)
        if update:
            logits = student_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


def test_pruning():
    net = StudentNet().cuda()
    net.load_state_dict(torch.load('student_custom_small.bin'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)
    train_dataloader = get_dataloader('training', batch_size=32)
    valid_dataloader = get_dataloader('validation', batch_size=32)
    now_width_mult = 1
    for i in range(5):
        now_width_mult *= 0.95
        new_net = StudentNet(width_mult=now_width_mult).cuda()
        params = net.state_dict()
        net = network_slimming(net, new_net)
        now_best_acc = 0
        for epoch in range(5):
            print("============Epoch{:>3d}============".format(epoch))
            net.train()
            train_loss, train_acc = run_epoch_pruning(train_dataloader, net, optimizer, criterion, update=True)
            net.eval()
            valid_loss, valid_acc = run_epoch_pruning(valid_dataloader, net, optimizer, criterion, update=False)
            if valid_acc > now_best_acc:
                now_best_acc = valid_acc
                torch.save(net.state_dict(), f'custom_small_rate_{now_width_mult}.bin')
                print(
                    'rate {:6.4f} epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
                        now_width_mult,
                        epoch, train_loss, train_acc, valid_loss, valid_acc))


def test_distillation():
    # define student net and teacher net
    teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
    student_net = StudentNet(base=16).cuda()
    teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
    optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)
    train_dataloader = get_dataloader('training', batch_size=32)
    valid_dataloader = get_dataloader('validation', batch_size=32)
    teacher_net.eval()
    now_best_acc = 0
    for epoch in range(200):
        print("============Epoch{:>3d}============".format(epoch))
        student_net.train()
        train_loss, train_acc = run_epoch_distillation(train_dataloader, teacher_net, student_net, optimizer,
                                                       update=True)
        student_net.eval()
        valid_loss, valid_acc = run_epoch_distillation(valid_dataloader, teacher_net, student_net, optimizer,
                                                       update=False)
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), 'student_model.bin')
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))


if __name__ == "__main__":
    test_pruning()

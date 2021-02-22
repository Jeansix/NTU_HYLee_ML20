import torch


def train_epoch(feature_extractor, label_predictor, domain_classifier, class_criterion, domain_criterion, opt_F, opt_C,
                opt_D, source_dataloader, target_dataloader, lamb):
    F_loss, D_loss = 0.0, 0.0  # loss for feature_extractor and domain_classifier

    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        domain_label[:source_data.shape[0]] = 1  # source data的domain label为1

        # step1. train domain classifier
        feature = feature_extractor(mixed_data)
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        D_loss += loss.item()
        loss.backward()
        opt_D.step()

        # step2. train feature extractor and label predictor
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        F_loss += loss.item()
        loss.backward()
        opt_F.step()
        opt_C.step()

        opt_D.zero_grad()
        opt_F.zero_grad()
        opt_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')
    return D_loss / (i + 1), F_loss / (i + 1), total_hit / total_num


def train(feature_extractor, label_predictor, domain_classifier, class_criterion, domain_criterion, opt_F, opt_C,
          opt_D, source_dataloader, target_dataloader, extractor_dir, predictor_dir, lamb, n_epoch):
    for epoch in range(n_epoch):
        D_loss, F_loss, train_acc = train_epoch(feature_extractor, label_predictor, domain_classifier, class_criterion,
                                                domain_criterion, opt_F, opt_C,
                                                opt_D, source_dataloader, target_dataloader, lamb)
        torch.save(feature_extractor.state_dict(), extractor_dir)
        torch.save(label_predictor.state_dict(), predictor_dir)
        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, D_loss,
                                                                                               F_loss, train_acc))

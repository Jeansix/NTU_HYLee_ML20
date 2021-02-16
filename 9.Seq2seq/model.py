import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert self.encoder.n_layers == self.decoder.n_layers  # 判断层数相同

    def forward(self, inputs, target, teacher_forcing_ratio, mode):
        """
        :param inputs:[batch_size, input_len]
        :param target:[batch_size,target_len]
        :param teacher_forcing_ratio:有多少概率用正确答案来训练
        :param mode:'train' or 'test'
        :return: outputs:[batch_size, target_len, cn_vocab_size]
        :return: preds:[batch_size, target_len, 1]
        """
        batch_size = target.shape[0]
        target_len = target.shape[1]
        cn_vocab_size = self.decoder.cn_vocab_size
        # 为输出开辟固定大小空间
        outputs = torch.zeros(batch_size, target_len, cn_vocab_size).to(self.device)
        # 将输入放进encoder
        encoder_output, encoder_hidden = self.encoder(inputs)
        # hidden =  [num_layers * directions, batch size,hid dim]  --> [num_layers,directions,batch size,hid dim]
        hidden = encoder_hidden.reshape(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, ::]), dim=2)  # ？？

        # 取<BOS>
        decoder_input = target[:, 0]  # first token
        preds = []
        if mode == 'train':
            for t in range(1, target_len):
                output, hidden = self.decoder(decoder_input, hidden, encoder_output)
                outputs[:, t] = output
                # 决定是否用正确答案来训练
                teacher_force = random.random() <= teacher_forcing_ratio and t < target_len
                # 取出概率最大的单词(在dim=1找最大下标)
                top1 = torch.argmax(output, 1)
                decoder_input = top1 if teacher_force else target[:, t]  # 更新decoder_input
                preds.append(top1.unsqueeze(1))  # ??
            preds = torch.cat(preds, 1)  # <EOS>
        elif mode == 'test':
            for t in range(1, target_len):
                output, hidden = self.decoder(decoder_input, hidden, encoder_output)
                outputs[:, t] = output
                top1 = torch.argmax(output, 1)
                decoder_input = top1
                preds.append(top1.unsqueeze(1))
            preds = torch.cat(preds, 1)
        else:
            print("mode is error!")

        return outputs, preds


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, embed_dim, hidden_dim, n_layers, dropout, bidirectional=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(en_vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [batch_size, sea_len, embed_dim]
        :return: out:[batch_size, seq_len, hidden*2]
        :return: hidden:[num_layers*2, batch_size, hidden]
        """

        embed_x = self.embedding(x)
        out, hidden = self.rnn(embed_x)
        return out, hidden


class Decoder(nn.Module):  # decoder只能是单向
    def __init__(self, cn_vocab_size, embed_dim, hidden_dim, n_layers, dropout, is_attn):
        super(Decoder, self).__init__()
        self.cn_vocab_size = cn_vocab_size
        self.input_dim = embed_dim
        self.hidden_dim = hidden_dim * 2  # why *2??
        self.n_layers = n_layers
        self.is_attn = is_attn  # 是否引入注意力机制
        self.embedding = nn.Embedding(cn_vocab_size, embed_dim)
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # classification layer
        self.embedding2vocab = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.cn_vocab_size)
        )

    def forward(self, x, hidden, encoder_outputs):
        """
        :param x:
        :param hidden: [num_layers,batch_size, hidden_dim]
        :param encoder_outputs:[batch_size,seq_len,hidden_dim]
        :return pred:[batch_size, cn_vocab_size]
        :return hidden:[num_layers, batch_size, hidden]
        """
        x = x.unsqueeze(1)
        input_embed = self.embedding(x)
        # 核心是将hidden加入attention以后，还原成原来的形式
        if self.is_attn:
            hidden = hidden.permute(1, 2, 0)  # [batch_size, hidden_dim, num_layers]
            attn_weights = torch.bmm(encoder_outputs, hidden)  # [batch_size, seq_len, num_layers]
            soft_attn_weight = nn.functional.softmax(attn_weights, dim=1)
            hidden = torch.bmm(soft_attn_weight.transpose(1, 2),
                               encoder_outputs)  # [batch_size, num_layers, hidden_dim]
            hidden = hidden.transpose(0, 1).contiguous()  # [num_layers, batch_size,hidden_dim]
        out, hidden = self.rnn(input_embed, hidden)  # out:[batch_size, seq_len, hidden]
        pred = self.embedding2vocab(out.squeeze(1))

        return pred, hidden

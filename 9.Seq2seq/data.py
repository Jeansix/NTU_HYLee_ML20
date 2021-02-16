import torch.utils.data as data
import torch


class EN2CNDataset(data.Dataset):
    def __init__(self, en, cn):
        self.cn = cn
        self.en = en

    def __len__(self):
        return len(self.en)

    def __getitem__(self, item):
        return torch.tensor(self.en[item]), torch.tensor(self.cn[item])

import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        images = self.image_list[item]
        return images

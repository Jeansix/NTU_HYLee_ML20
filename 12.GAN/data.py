from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)  # 读进来直接是BGR 格式数据格式在 0~255
        img = self.BGR2RGB(img)  # because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换成RGB格式


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    dataset = FaceDataset(fnames, transform)
    return dataset


if __name__ == "__main__":
    workspace_dir = './'
    dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    plt.imshow(dataset[10].numpy().transpose(1, 2, 0))# 把channel放到最后一维
    plt.show()

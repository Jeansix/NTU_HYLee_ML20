import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

source_transform = transforms.Compose([
    # 转灰阶，因为Canny不吃RGB
    transforms.Grayscale(),
    # 转np.array后再做cv2.Canny,因为cv2不吃skimage.Image
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # 重新将np.array转成skimage.Image
    transforms.ToPILImage(),
    # 水平翻转（augmentation）
    transforms.RandomHorizontalFlip(),
    # 旋转15度以内
    transforms.RandomRotation(15),
    # 转成tensor供模型使用
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    # 转灰阶：将三维转为一维
    transforms.Grayscale(),
    # 缩放：source data是32x32
    transforms.Resize((32, 32)),
    # 水平翻转（augmentation）
    transforms.RandomHorizontalFlip(),
    # 旋转15度以内
    transforms.RandomRotation(15),
    # 转成tensor供模型使用
    transforms.ToTensor(),

])

if __name__ == "__main__":
    source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
    target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
    print(source_dataset.__len__())
    print(target_dataset.__len__())

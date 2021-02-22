### 📕Data

##### Data Form

source data + label：真实图片，共5000张

target data：涂鸦图片，共100000张

##### Data Processing

```python
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

```

##### Dataset encapsulation

Use Dataset(ImageFolder) and DataLoader to encapsulate data for training, validation and testing.

### 🔮Model

##### DaNN

核心：让Soucre Data和Target Data经过Feature Extractor都在同一个Distribution上

![Alt text](https://github.com/Jeansix/NTU_HYLee_ML20/blob/master/13.Transfer Learning/images/model.png)

![Alt text](https://github.com/Jeansix/NTU_HYLee_ML20/blob/master/13.Transfer Learning/images/train.JPG)

##### Feature Extractor

 典型的VGG-like卷积网络，学习如何产生feature来骗过domain classifier

##### Label Predictor

MLP，判断source domain的data来自哪个class

##### Domain Classifier

MLP，判断feature来自哪个domain

#### 🚆Train

交替训练Domain Classifier 和Feature Extractor / Label Predictor 

loss为原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。

```txt
类似GAN的做法，将Feature Extractor和Label Predictor看作G，Domain Classifier看作D
step1.for k steps：保持G不变，训练D让式子domain BCE的值达到最大
step2.保持D不变，训练G使式子class CE - lamb * domain BCE的值达到最大 
step3.重复step1和step2，直到G和D达到平衡
```



### 🔨Inference

```python
1.load pretrained model feature_extractor and label_predictor
2.generate feature for classification based on the two model
3.make label prediction on feature extracted from target data 
```


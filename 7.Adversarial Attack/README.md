### 📕Data

##### Data Form

```txt
attack data
├─ images
│	   ├─ 000.png
│	   ...(200 in total)
│	   ├─ 199.png
└─ categories.csv(CategoryId,CategoryName)
│   
└─ labels.csv(只用关心TrueLabel这个属性，表示picture所属class)

image->label(TrueLabel)->category(CategoryName)
```

##### Dataset encapsulation

Use Dataset and DataLoader to encapsulate data for attacking.

### 🔮Model

##### step0. Load pretrained model weight

因为网络不是很稳定，所以将pretrained参数设为False。下载好weight以后，再将它加载进来。

```python
self.model=models.vgg16(pretrained=False)
self.model.to(device)
pre=torch.load(fpath,map_location=device) # fpath is path for pretrained weigths
self.model.load_state_dict(pre) # load pretrained weight to the model
```



##### step1. Transform dataset

Apply **transforms** on dataset to tailor pictures to the pretrained models.

```python
self.mean = [0.485, 0.456, 0.406]
self.std = [0.229, 0.224, 0.225]
transform=transforms.Compose([
    		# 将输入PIL图像的大小调整为给定大小。所需的插值为3。
            transforms.Resize((224, 224), interpolation=3),
    		# 将PIL Image转换为tensor，并且通过除以255归一化至[0,1] 
            transforms.ToTensor(),
    		# 用平均值和标准偏差归一化张量图像。此变换将标准化输入的RGB三个通道。
    		# 即input[channel] = (input[channel] - mean[channel]) / std[channel]
            transforms.Normalize(self.mean, self.std, inplace=False)
        ])
```



##### step2. Core algorithm

核心算法是FGSM：在原来的图片$X_0$上加一个很小的扰动$\Delta$x，得到新的图片X'=$X_0+\Delta x$，让新图片通过网络得到的分类结果从原来的正确变为错误。这个noise加在梯度的方向。（梯度方向是函数值下降最快的方向，相当于设置了一个很大的learning rate，一次update就可以达到攻击目标。）

```python
def fgsm_attack(self, image, epsilon, data_grad):
        # locate gradient
        sign_data_grad=data_grad.sign() # 只在意gradient的方向，不在意大小！
        # add noise in gradient direction
        delta_image=image+epsilon*sign_data_grad
        return delta_image
```

FGSM属于白盒攻击，攻击者知道模型所使用的算法，以及算法所使用的参数（可以调整epsilon来观察攻击效果）。如果给定一个网络参数，白盒攻击是最成功的方法。

##### step3. Do attack

将样本分为三类：

    wrong: initially predicted as a wrong class
    (true):
    	fail: attack failed
    	success: attack succeeded
我们关心的是原先分类器预测正确的样本，在加入扰动之后，分类器是否还能将它预测正确。

当epsilon设置为0.1时，有6/200个样本在加入扰动后分类器依旧可以预测正确。

当epsilon设置为0.01时，有54/200个样本在加入扰动后分类器依旧可以预测正确。

反过来说，给的扰动越小，分类器越不容易判别错误(attack succeeded)，符合一般认知。






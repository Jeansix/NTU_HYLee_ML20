### 📕Data

##### Data Form

71313张动漫人物脸的图片

##### Data Processing

```txt
1.将BGR格式转换成RGB格式
2.data resize
3.data transform
```

##### Dataset encapsulation

Use Dataset(EN2CNDataset) and DataLoader to encapsulate data for training, validation and testing.

### 🔮Model

##### DCGAN

DCGAN=Generator+Discriminator

![Alt text](https://github.com/Jeansix/NTU_HYLee_ML20/blob/master/12.GAN/images/model.png)

##### Generator

input (N, in_dim)
output (N, 3, 64, 64)

 将一个100 维的噪音向量扩展成64 * 64 * 3 的矩阵输出

整个过程采用了卷积+上采样，目的是将经过池化层以后缩小的矩阵扩大到一定的大小。

UnPooling：特点是在Maxpooling的时候保留最大值的位置信息，之后在unPooling阶段使用该信息扩充Feature Map

![Alt text](https://github.com/Jeansix/NTU_HYLee_ML20/blob/master/12.GAN/images/unpooling.jpg)

UnSampling：没有使用MaxPooling时的位置信息，而是直接将内容复制来扩充Feature Map。

![Alt text](https://github.com/Jeansix/NTU_HYLee_ML20/blob/master/12.GAN/images/unsampling.jpg)

而反卷积可以实现UnPooling和unSampling，只要设置合理的参数。

![Alt text](https://github.com/Jeansix/NTU_HYLee_ML20/blob/master/12.GAN/images/deconvolution.JPG)

##### Discriminator

input (N, 3, 64, 64)
output (N, )

用于通过卷积操作判断产生的图片是来自真实样本（r_imgs)，还是generator产生的样本(f_imgs)

##### Advantage

- 使用卷积和去卷积代替池化层
- 在生成器和判别器中都添加了批量归一化操作
- 去掉了全连接层，使用全局池化层替代
- 生成器的输出层使用Tanh 激活函数，其他层使用RELU
- 判别器的所有层都是用LeakyReLU 激活函数



#### 🚆Train

- define criterion

- define optimizer

- change mode to enable optimizer to update parameters

  `model.train()`

- start training!

  ```txt
  step1.for k steps：保持G不变，训练D让式子logD(x) + log(1 - D(G(Z))的值达到最大
  step2.保持D不变，训练G使式子logD(G(z))的值达到最大 
  step3.重复step1和step2，直到G和D达到平衡
  ```
  
  

### 🔨Inference

```python
1.load pretrained model G
2.generate images based on randomly sampled z
3.show image
```

ref：

DCGAN: https://blog.csdn.net/qq_33594380/article/details/84135797

反卷积，上采样，上池化：https://blog.csdn.net/e01528/article/details/84667302


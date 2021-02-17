### 📕Data

##### Data Form

NumPy 为 ndarray 对象引入了一个简单的文件格式：.npy

npy 文件用于存储重建 ndarray 所需的数据、图形、dtype 和其他信息。

trainX.npy

valX.npy

valY.npy

##### Data Processing

归一化；维度之间换顺序

##### Dataset encapsulation

利用Dataset类封装成ImageDataset

### 🔮Model

##### Auto encoder for CNN

- Encoder：Convolution->pooling->Convolution->pooling->Convolution->pooling->code
- Decoder：code->deconvolution->unpooling->deconvolution->unpooling->deconvolution->unpooling

deconvolution的本质就是convolution，nn.ConvTranspose2d封装了此反卷积操作

#### 🚆Train

- define criterion

- define optimizer

- change mode to enable optimizer to update parameters

  `model.train()`

- start training!

  ```python
  for each epoch:
      set train_loss to 0
  	load data to gpu
  	optimizer.zero_grad()
  	# generate outputs
  	outputs=model(inputs)
  	# calculate loss
  	loss=criterion(outputs,labels)
  	loss.backward()# calculate gradient
  	optimizer.step() # update parameters
  	update loss
  	checkpoint
  ```

  

### 🔨Test

```python
# test=inference+predict
load model parameters
model.eval()
inference:经过encoder得到vector
predict:对inference后的结果进行clustering,然后分类
   Dimension Reduction:Latents Shape: (data_size, 4096)
			   First Reduction Shape: (data_size, 200)
			   Second Reduction Shape: (data_size, 2)
```



### 🖊Analysis

Problem 1(作图) 降到两维后，不同类点之间有一定的距离。证明了二维的情况下，deep auto encoder的点是分开的，而PCA的点混在一起。

Problem 2 选取一些图片，画出原图和reconstruct之后的图片

Problem 3 挑选10个checkpoint的model

随着epoch的增加，reconstruction error减小，accuracy上升
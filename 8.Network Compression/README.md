Model Compression 有很多种门派：

- Knowledge Distillation: 让小 model 萃取大 model 的知识
- Network Pruning: 将已经学习好的的大 model 做剪枝（weight/neuron）,让整个模型变小
- Weight Quantization: 用更好的方式来表现 model 中的参数，以此降低运算消耗/容量。
- Design Architecture: 将原始的 layer 用更小的参数来表现。 (例如 Convolution ￫ Depthwise & Pointwise Convolution)

### Design Architecture

##### 核心

使用Depthwise & Pointwise Convolution Laye来叠model，总共有7层CNN。

- DW(Depthwise Convolution Layer)：一张feature map各自用一个filter处理
- PW(Pointwise Convolution Layer)：把所有feature map的单个pixel资讯合在一起(就是1个pixel的Fully Connected Layer)。

将原本的Convolution Layer换成Dw & Pw後，Accuracy通常不会降很多。同时，模型的参数显著减少了。

##### 实现

```python
# 一般的Convolution, weight大小 = in_chs * out_chs * kernel_size^2
nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)

# Group Convolution, Group数目可以自行控制，表示要分成几群。其中in_chs和out_chs必须要可以被groups整除。(不然沒办法分群。)
nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups)

# Depthwise Convolution, 輸入chs=输出chs=Groups数目, weight大小 = in_chs * kernel_size^2
nn.Conv2d(in_chs, out_chs=in_chs, kernel_size, stride, padding, groups=in_chs)

# Pointwise Convolution, 也就是1 by 1 convolution, weight大小 = in_chs * out_chs
nn.Conv2d(in_chs, out_chs, 1)
```



### Knowledge Distillation

##### What?

利用大model预测的logits给小model当作标准，从而告诉小的model如何学习.

##### Why?

- 当data不是很干净的时候，对一般model来说是noise，会干扰学习。通过去学习其他大的model预测的logits会比较好。
- label和label之间可能存在关联，这可以引导小的model去学习。（如数字8和6，9，0就有关系）
- 弱化已经学习不错的target，避免让其gradient干扰其他没有学习好的task。

##### How?

 $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{Original Loss})$

改一下损失函数，分成hard_loss和soft_loss两部分。hard_loss针对hard_labels,soft_loss针对teacher_outputs。

```python
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的cross entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 将logits的log_softmax对目标概率(teacher的logits/T后softmax)做KL Divergence
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                    F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss
```

##### Steps

1. 封装dataset
2. 得到相应的dataloader
3. 载入teacher_net:torchvision提供的ResNet18，把num_classes改成11后load进去即可
4. 定义optimizer，loss，开始训练

### Network Pruning

##### What?

对已经学习完的model中的Neuron进行删减，让网络更瘦。Neuron Pruning需要衡量Neuron的重要性，将不重要的Neuron删减掉。

衡量方法：通过batchnorm layer的𝛾因子來決定neuron的重要性。 (by paper - Network Slimming)

##### How?

- 将StudentNet(width_mult=α)的neuron经过筛选后移植到StudentNet(width_mult=β)。(α>β)
- 筛选只需要抓出每一个block的batchnorm的γ即可。

Model的中间两层比较特殊:

| Layer                           | Output of Channels |
| ------------------------------- | ------------------ |
| Input                           | in_chs             |
| Depthwise(in_chs)               | in_chs             |
| BatchNorm(in_chs)               | in_chs             |
| Pointwise(in_chs, **mid_chs**)  | **mid_chs**        |
| **Depthwise(mid_chs)**          | **mid_chs**        |
| **BatchNorm(mid_chs)**          | **mid_chs**        |
| Pointwise(**mid_chs**, out_chs) | out_chs            |

利用第二个BatchNorm来做筛选的時候，跟他的Neuron有直接关系的是该层的Depthwise&Pointwise以及上层的Pointwise。
因此再做neuron筛选时记得要将这四个(包括自己, bn)也要同时prune掉。

Model其他的层：

|      | name      | meaning                     | code                              | weight shape |
| ---- | --------- | --------------------------- | --------------------------------- | ------------ |
| 0    | cnn.{i}.0 | Depthwise Convolution Layer | nn.Conv2d(x, x, 3, 1, 1, group=x) | (x, 1, 3, 3) |
| 1    | cnn.{i}.1 | Batch Normalization         | nn.BatchNorm2d(x)                 | (x)          |
| 2    |           | ReLU6                       | nn.ReLU6                          |              |
| 3    | cnn.{i}.3 | Pointwise Convolution Layer | nn.Conv2d(x, y, 1),               | (y, x, 1, 1) |
| 4    |           | MaxPooling                  | nn.MaxPool2d(2, 2, 0)             |              |

我们要抓取的$\gamma$系数在在cnn.{i}.1.weight内。

### Weight Quantization

核心思想：use less bit to represent a value.在实作中，将32-bit的tensor转换成更少bit的ndarray存起来。

定义`encode`和`decode`函数，前者将模型参数压缩成低bit保存到文件中，后者将文件中保存的参数还原成torch.tensor，再存入state_dict中。

- to 16 bit：用`np.float16`进行转化压缩即可

- to 8 bit：因为沒有8-bit的float，所以我们先对每个weight记录最小值和最大值，进行min-max正则化后乘上255再四舍五入，就可以用np.uint8存取了。

  转化公式：$W' = round(\frac{W - \min(W)}{\max(W) - \min(W)} \times (2^8 - 1)$)
### 📕Data

##### Data Form

```txt
food-11
├─ training
│	   ├─ label_no.jpg
│	   ...(9866 in total)
└─ validation
│      ├─ label_no.jpg
│      ...(3430 in total)
└─ testing
       ├─ no.jpg
       ...(3347 in total)
```

##### Dataset encapsulation

Use Dataset and DataLoader to encapsulate data for training and testing.



### 🔮Model

A multilayer-CNN classifier for food classification.



### 💡Method

##### Saliency map

计算loss对image的偏微分值，代表在model parameter和label都固定的情况下，稍微改变image的某个pixel value会对loss发生什么变化。这个变化的剧烈程度称为该pixel的重要性，可以衡量图中哪些位置是model在判断时的重要依据。

##### Filter explain

若想要知道一个filter认出了什么，有两件事：

- Filter activation：图片中哪些位置会activate该filter
- Filter visualization：怎样的image可以最大程度activate该filter

实现：加入`hook_handle`,告诉pytorch当forward过了第cnnid层cnn后，要先调用`hook`这个定义好的function后才可以继续forward下一层cnn。在`hook`中，会把该层的activation map/output记录下来。

###### Filter activation

观察x经过被指定filter的activation map

```python
	model(x.cuda())
	filter_activations = layer_activations[:,filterid,:,:].detach().cpu()
```



###### Filter visualization

修改x，最大化filter activation

```python
 	x = x.cuda()
    x.requires_grad_()
    optimizer = Adam([x],lr=lr)
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
         # 类比loss,但这里是maximize,所以要加-号
        objective = -layer_activations[:,filterid,:,:].sum()
        objective.backward()
        #算 filter activation 对 input image 的偏微分
        optimizer.step()
    # 修改input image来最大化filter activation
    filter_visualization = x.detach().cpu().squeeze()[0]
```

因为最近正好在复习强化学习，感觉这两个步骤很类似于强化学习中的policy evaluation和policy improvement，也很类似于期望最大化算法，是一个**迭代优化**的过程。



### 🔨Test

```python
# 指定需要分析的图像下标
img_indices = [83, 4218, 4707, 8598]
# 得到图像数据
images, labels = train_set.getbatch(img_indices)

# 画出图像
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
     # 在 pytorch 的世界，image tensor 各 dimension 为 (channels, height, width)
     # 但在 matplolib 的世界，想要把一個 tensor 画出來，形状为 (height, width, channels)
     # 因此，要通过permute来作dimension之间的转换
    axs[0][i].imshow(img.permute(1, 2, 0))
plt.show()
plt.close()

```


### 📕Data

train.npy： 是原先trainingset中的某些类的data，干净的数据，大小为(40000, 32, 32, 3)（40000个32*32图片样本，共rgb三个通道）

test.npy：是原先testingset中的所有data混入了outlier data(anomaly)

要侦测的anomaly为training data 中没有出现的类别。

label的部分，1为outlier data而0为inlier data

### 🔮Method

- KNN

  假设training data共有n群，先用K-means计算training data中的n个centroid，再用这n个centroid对training data分群。可以观察到，inlier data与所分到群的centroid的距离应当小于outlier data到此的距离。

- PCA

  先计算training data的principle component，将testing data投影在这些component上，再将投影重建回原先space的向量。对重建图片和原图计算MSE，inlier data的数值小于outlier data。

- AE

  通过encoder-decoder模型重构出的图片和原图计算损失，inlier data的数值小于outlier data。根据encoder-decoder的不同实现，可以分为：

  - fc_autoencoder

    encoder和decoder都是线性的

  - conv_autoencoder

    encoder和decoder是卷积网络

  - VAE

    为每个样本构造专属的正态分布，然后采样来重构。

### 🔨Test

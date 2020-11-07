# <center> Regression</center>

### 🔈Introduction



### 📕Data

##### Training Data

每一天（日期区分，每个月放了20天的数据）的记录包含了两个维度：

- 横-24个小时

- 纵-18项观测数据/features: 

  ```txt
  AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR
  ```

  

##### Testing Data

每一条（id区分，共240条）记录包含了两个维度：

- 横-9个小时
- 纵-18项观测数据/features

目标是将前九小时的观测数据作为features，第十小时的PM2.5作为answer。



### 🌷Feature

- 每个月作为一个样本，样本大小为`18*480`,18是观测数据的维度，480是每个月选取20天和每天24小时的乘积

  <img src="D:\university\大四前进（1）\机器学习\NTU_HYLee_ML20\Regression\extract_feature1.png" alt="extract_feature1" style="zoom:75%;" />

- 维护一个9小时的滑动窗口，

  1个月有480小时，每9小时形成一组data，共有471组data，

  12个月一共有`12*471`组data,每组data的维度是`18*9`，

  <img src="D:\university\大四前进（1）\机器学习\NTU_HYLee_ML20\Regression\extract_feature2.png" alt="extract_feature2" style="zoom:75%;" />

  对应的标签有`471*12`个，为第10小时的PM2.5


# <center> Classification</center>

### 🔈Introduction



### 📕Data

##### Training Data

X_train: each line represents a training sample. There are 510 features setting  0 or 1.

Y_train: labels for X_train

X_test: each line represents a testing sample

```python
def read_file(fpath,islabel=False):
    with open(fpath) as f:
        next(f) # skip the first index line
        if islabel is False:
            x=np.array([line.strip('\n').split(',')[1:] for line in f],dtype=float)
        else:
            x=np.array([line.strip('\n').split(',')[1] for line in f],dtype=float)
    return x
```

The read_file function read data from file, after which a matrix of (54265,510) ,  (54256,) ,  (27622,510) are generated respectively.

### 💡Model 

##### Discriminative Classifier

###### 1 Principle

Discriminative classifiers model the posterior p(ylx) directly, or model the **decision boundary** between the classes. **Logistic Regression** works on multi-class classification. To train this linear discriminative classifier, **mini-batch gradient descent** is employed to minimize loss. In every iteration, choose batch_size of samples to compute loss and update the weight matrix. In every epoch, the dataset is reshuffled and similar procedures are redone. The decision boundary can be learned through this process.

###### 2 Sigmoid

Sigmoid function is used to score each class. The higher the score, the more likely the sample is in this class. We have function $\sigma(w^Tx+b)=\frac{1}{1+e^{(-w^Tx+b)} }$ . To avoid overflow and underflow, numpy.clip(a, a_min, a_max, out=None) is employed to refine the value between max and min.

```python
def sigmoid(z):
    return np.clip(1/(1.0+np.exp(-z)),1e-8, 1 - (1e-8))
```

###### 3 Gradient Descent

**Cross-entropy loss** is employed to calculate loss:  $R(W)=-\frac{1}{N}\sum_{n=1}^{N}{(y^{(n)})}^Tlog {\hat y}^{(n)}$

```python
# 2 Classes
def cross_entropy_loss(Y_pred, Y_label):
    cross_entropy=-np.dot(Y_label,np.log(Y_pred))-np.dot(1-Y_label,np.log(1-Y_pred))
    return cross_entropy
```

And the gradient is: $\frac{\partial(R(W))}{\partial(W)}=-\frac{1}{N}\sum_{n=1}^{N}x^{(n)}{(y^{(n)}-{\hat y}^{(n)})}^T$

```python
def gradient(self,X,Y_label):
        z=np.matmul(X,self.w)+self.b
        Y_pred=sigmoid(z)
        error=Y_label-Y_pred
        w_grad=-np.sum(error*X.T,1)
        b_grad=-np.sum(error)
        return w_grad,b_grad
```

Update parameter: $W_{t+1}←W_t-(-\alpha(\frac{1}{N}\sum_{n=1}^{N}x^{(n)}{(y^{(n)}-{\hat y_{w_t}}^{(n)})}^T))$

```python
# Compute the gradient
w_grad, b_grad = self.gradient(X, Y)
                
# gradient descent update
# learning rate decay with time
self.w-=lr/np.sqrt(step) * w_grad
self.b-=lr/np.sqrt(step) * b_grad

step = step + 1
```

Meanwhile, the learning rate is also tailored in purpose of convergence.

###### 4 Summary

It can be concluded that discriminative classifier tackles classification problem directly. The input dataset yields a decision boundary as output through training. The drawback is it can't generate new samples and relationships between variables are not explicit.

##### Generative Classifier

###### 1 Principle

Generative classifiers learn a model of the **joint probability**, p( x, y), of the inputs x and the label y, and make their predictions by using **Bayes rules** to calculate p(ylx), and then picking the most likely label y. First, the dataset is assumed to follow gaussian distribution. Then, **parametric density estimation** is employed to calculate mu, sigma. Finally, use mu and sigma to calculate w and b, which is different from the value achieved from discriminative classifier. Then, sigmoid is similarly employed to predict the class.

###### 2 $\mu$

$\mu=\overline X=\frac{1}{N}\sum X_{(a)}$

```python
_,dim=X_train.shape
# divide into two classes
X_train_0=np.array([x for x,y in zip(X_train,Y_train) if y==0])
X_train_1=np.array([x for x,y in zip(X_train,Y_train) if y==1])
# calculate mu for each class
mean_0=np.mean(X_train_0,axis=0)
mean_1=np.mean(X_train_1,axis=0)
```

###### 3 $\Sigma$

$S=\frac{1}{N}\sum {(X_{(a)}-\overline X)}^T(X_{(a)}-\overline X)$

```python
# calculate cov for each class
cov_0=np.zeros((dim,dim))
cov_1=np.zeros((dim,dim))
        
for x in X_train_0:
	cov_0+=np.dot(np.asmatrix(x-mean_0).T,np.asmatrix(x-mean_0))/X_train_0.shape[0]
for x in X_train_1:
    cov_1+=np.dot(np.asmatrix(x-mean_1).T,np.asmatrix(x-mean_1))/X_train_1.shape[0]
            
# shared cov
cov=(cov_0*X_train_0.shape[0]+cov_1*X_train_1.shape[0])/(X_train_0.shape[0]+X_train_1.shape[0])
```

###### 4 w,b

$z=(\mu_1-\mu_2)^T\Sigma^{-1}x-\frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1+\frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2+\log\frac{N1}{N2}$

```python
# compute inverse of cov
u,s,v=np.linalg.svd(cov,full_matrices=False)
inv=np.matmul(v.T*1/s,u.T)
        
self.w=np.dot(inv,mean_0-mean_1)
     self.b=-0.5*np.dot(mean_0,np.dot(inv,mean_0))+0.5*np.dot(mean_1,np.dot(inv,mean_1))+np.log(float(X_train_0.shape[0])/X_train_1.shape[0])
```

###### 5 Summary

It can be concluded that generative classifier explicitly models the actual distribution of each class. By substituting samples into the estimated distributions, it can find out the most likable class.



##### Comparison

###### 1 Math

From the perspective of math, ‌training classifiers involve estimating P(Y|X). To achieve this, the methods differ in the two models.

> ###### **Generative classifiers**
>
> - Assume some functional form for **P(Y), P(X|Y)**
> - Estimate parameters of **P(X|Y), P(Y)** directly from training data
> - Use Bayes rule to calculate **P(Y |X)**
>
> ###### **Discriminative Classifiers**
>
> - Assume some functional form for **P(Y|X)**
> - Estimate parameters of **P(Y|X)** directly from training data

###### 2 Training

For discriminative classifiers, they learn a decision boundary by employing mini-batch gradient descent.  Each iteration contributes to the optimization of the decision boundary. 

For generative classifiers,  they first assume the distribution form. Then, they learn sigma, mu of the distribution by employing parametric density estimation. 

Based on the differences, it can be concluded that:

- When the generative **modeling** **assumptions do not hold**, the performance of the generative model will decrease sharply. In this case, discriminative classifiers perform better.

- When training **data is scarce**, generative models are preferred. Discriminative classifiers needs great amounts of data to learn the decision boundary,  while generative classifiers requires fewer if the assumption holds.

- Naive Bayes is a learning algorithm with **greater bias, but lower variance**, than Logistic Regression. Because it pays less attention to the training data and oversimplifies the model by asserting a strong assumption.  On the contrary, Logistic Regression only learns from training data and does not generalize on the data which it hasn’t seen before. Each update of weight matrix depends only on previous label, and not future labels.

  

###### 3 Parameters

For generative classifiers, we have 2*M($\mu_1,\mu_2$)+$\frac{M·(M+1)}{2}(\Sigma)+1(\pi)$ parameters. M is the dimension of samples.

For discriminative classifiers, we have M+1 parameters.

It can be concluded that **generative classifiers needs more parameters**, but they **can generate synthetic example data** with them accordingly. 

###### 

###### 4 Cost

For discriminative classifiers, the **time complexity for training** the model is very high.

For generative classifiers, it **takes time to estimate the parameters  of O($M^2$) scale.** M is the dimension of samples.

###### 5 Application

Generative model involve modeling, discriminative models directly solve classification. Thus, generative models are more elegant, have explanatory power. They are more suitable for situations where samples are scarce,  distribution form is known and generating new samples is required. Discriminative models are more desirable for pure classification problem with clear boundary and large training samples.


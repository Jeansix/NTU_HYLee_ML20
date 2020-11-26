# <center>CNN</center>

### 📕Data

```
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

##### Load data

###### Dataset

- Numpy +opencv

- ImageFolder 

###### DataLoader

### 🔮Model

##### Definition

Convolution(`Conv2d`) -> Batch Normalization(`BatchNorm2d`)->Activate function(`ReLU`) -> Max Pooling (`MaxPooling`) -> ... -> Flatten(`view`)-> Fully Connected(`Linear`) -> Output

Some useful tips:

```txt
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
torch.nn.MaxPool2d(kernel_size, stride, padding)
feature map:(n+2p-f)/s+1,
			 n-original size;
			 p-padding;
			 f-filter size;
			 s-stride
```

##### Framework

There is a specific framework for pytorch to train models. First, loss function and optimizer should be selected.

- In this statement, optimizer gets all the parameters  (tensors) of the model to  be updated. The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) . Meanwhile ,learning rate is set.

  `optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)`

- In this statement, cross entropy loss is applied for the multi-classification problem.

  `loss = nn.CrossEntropyLoss()`

- When `loss.backward()`is called, all it does is compute gradient of the parameters in loss that have `requires_grad = True` and store them in `parameter.grad` attribute for every parameter. Before that, optimizer should be reset because the previous accumulated gradients is still recorded and will exert an effect.

  `optimizer.zero_grad()`

  `loss.backward()`

- In this statement, the optimizer iterates over all parameters (tensors) it is supposed to update and use their internally stored `grad` to update their values.

  `optimizer.step()` 

  
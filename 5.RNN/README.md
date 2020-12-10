### 📕Data

##### Data Form

`training_label.txt`

```
0 +++$+++ i feel icky , i need a hug
```



`training_nolabel.txt`

```
i would loooooooove to come !!! xd but i ' m not from america i wish i was 3 !! have funnnnnnnnnnnnnnnn !!
```



`testing_data.txt`

```txt
id,text
0,my dog ate our dinner . no , seriously ... he ate it .
```

##### Data Processing

```txt
build word2vec model => get_w2v_model
make embedding => make_embedding
add embedding for <PAD> and <UNK> => add_embedding
pad sentences to certain length =>pad_sequence
text serialization => sentece_word2idx
```



##### Dataset encapsulation

Use Dataset and DataLoader to encapsulate data for training and testing.

### 🔮Model

##### Framework



<img src="D:\university\大四前进（1）\机器学习\NTU_HYLee_ML20\5.RNN\model.JPG" alt="3d" style="zoom: 50%;" />

##### LSTM Part

**-nn.LSTM(input_size, hidden_size, num_layers)**: input_size indicates the dimension of features, hidden_size shows how many hidden cells lie in the hidden state, num_layers records the number of hidden layers. Batch_first=true means the input for the neural network begins with batch_size. 

The **input** for LSTM layer has two parameters: **input(batch_size,seq_len,input_size)** and **h0(batch_size,num_layers\*num_directions,hidden_size)**.

The LSTM layer yields two **outputs**: **out(batch_size,seq_len,hidden_size\*num_directions)** and **h_n(batch_size,num_layers\*num_directions,hidden_size)**

<img src="D:\PRML\asm2\3d.png" alt="3d" style="zoom: 50%;" />

#### 🚆Train

- change mode to enable optimizer to update parameters

  model.train()

- define criterion

- define optimizer

- start training!

  ```python
  for each epoch:
  	set total_loss,total_acc to 0
  	load inputs and labels to gpu
  	optimizer.zero_grad()
  	# generate outputs
  	outputs=model(inputs)
  	outputs=outputs.squeeze()
  	# calculate loss
  	loss=criterion(outputs,labels)
  	loss.backward()# calculate gradient
  	optimizer.step() # update parameters
  	call evaluation to calculate accuracy
  	update total_acc
  	update total_loss
  	# valid data
  	model.eval()
  	with torch.no_grad():
  		repeat procedure of train_data 
  ```

  

### 🔨Test

```python
model.eval()
	with torch.no_grad():
		load inputs to gpu
		# generate outputs
		outputs=model(inputs)
		outputs=outputs.squeeze()
		assign outputs to 1 or 0
		transfer outputs to list
```


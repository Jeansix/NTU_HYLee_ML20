### 🎮Gym

Gym可以建立一个LunarLander环境，这个环境模拟了登月小艇降落在月球表面时的情形。

环境同时包括了agent和environment,agent不断地和environment互动。

`env.observation_space`：返回一个八维向量，作为observation/state

`env.action_space`：返回Discrete(4)，说明agent可以采取四种离散的行动（0-不采取任何行动；1-主引擎向左喷射；2-主引擎向下喷射；3-主引擎向右喷射）

`env.step(action)`:返回(observation,reward,done,_),表示采取了action以后,agent对于environment的observation，这一个action获得的reward，和到这一步为止这个episode是否结束了（当登月小艇成功着陆，或者不幸坠毁时，一个episode就结束了，done为True）

`reward:`

| case               | reward  |
| ------------------ | ------- |
| 小艇坠毁           | -100    |
| 小艇在黄旗之间着陆 | 100~140 |
| 主引擎向下喷射     | -0.3    |
| 小艇完全静止       | 100     |

一个random agent的例子：

```python
# 任何操作之前，先重置环境
env.reset()

#展示图形界面
img = plt.imshow(env.render(mode='rgb_array'))

#采样一次
done = False
while not done:
    #随机采取一个动作
    action = env.action_space.sample()
    #更新一步，这一步agent和environment进行了交互
    observation, reward, done, _ = env.step(action)
	#更新图形界面
    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
```



### 🔮Model

##### step 1. policy network

input:observation,一个八维向量

output:离散的四个动作之一

目的就是用神经网络来拟合输入到输出的映射关系

##### step 2. policy gradient agent

agent搭配policy network来采取行动

`learn(log_probs, rewards)`:根据记录的log probabilities和rewards来更新policy network

`sample(state)`:根据当前enviroment的state,利用policy network得出应该采取的行动。按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。回传抽样得到的action和此次抽样的log probabilities。

### 🚆Train

change mode to enable optimizer to update parameters

`model.train()`

```python
for each episode:
	#reset state
	state = env.reset()
    total_reward,total_step=0,0
    while True:
        # sample action based on current state
        action, log_prob = agent.sample(state)
        # a step forward
        next_state, reward, done, _ = env.step(action)
        # make records
        log_probs.append(log_prob) # add log_probs
        total_reward += reward# add reward to the episode's total reward
        total_step += 1#count steps
        state = next_state# update state
        if done:
            final_rewards.append(reward)
            total_rewards.append(total_reward)
            # set rewards in one episode the same
            rewards.append(np.full(total_step, total_reward))
            break
    rewards = np.concatenate(rewards, axis=0)
    #normalization
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)# avoid divided by 0
    # calculate loss and update parameters by the agent based on log probabilities and 
    # rewards for each episode
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
```



- np.concatenate((a1,a2,...),axis=0)

  -功能：数组拼接

  -(a1,a2,...):需要被连接的数组，除了axis对应的轴，数组其他维度必须一样

  -axis:连接方向

  -e.g.

  ```python
  a = np.array([[1, 2], [3, 4]])#(2*2)
  b = np.array([[5, 6]])#(1*2)
  np.concatenate((a, b), axis=0)# 2*2,1*2
  array([[1, 2],
         [3, 4],
         [5, 6]])#3*2
  
  np.concatenate((a, b.T), axis=1)#2*2,2*1
  array([[1, 2, 5],
         [3, 4, 6]])#2*3
  
  np.concatenate((a, b), axis=None)
  array([1, 2, 3, 4, 5, 6])
  ```

  

- torch.stack((a1,a2,...),dim=0)

  -功能： 沿一个新维度对输入张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度，而stack（）函数指定的dim参数，就是新增维度的（下标）位置

  -(a1,a2,...):需要被连接的tensor

  -dim:拼接维度

  -e.g.

  ```python
  """
  torch.stack的参数dim的功能是标识出，新增的那一维位置，如果dim=0，则新增的维度为第一维，第一维就是原来的tensor拼接而成。而第二维、第三维则是原来三个tensor的继承。以次类推dim为1和2的情况。
  """
  import torch
  a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  b = torch.tensor([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
  c = torch.stack([a, b], dim=0)
  print(c)
  tensor([[[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9]],
   
          [[11, 22, 33],
           [44, 55, 66],
           [77, 88, 99]]])
  
  
  c = torch.stack([a, b], dim=1)
  print(c)
  tensor([[[ 1,  2,  3],
           [11, 22, 33]],
   
          [[ 4,  5,  6],
           [44, 55, 66]],
   
          [[ 7,  8,  9],
           [77, 88, 99]]])
  
  c = torch.stack([a, b], dim=2)
  print(c)
  tensor([[[ 1, 11],
           [ 2, 22],
           [ 3, 33]],
   
          [[ 4, 44],
           [ 5, 55],
           [ 6, 66]],
   
          [[ 7, 77],
           [ 8, 88],
           [ 9, 99]]])
  ```

  

### 🔨Test

change mode to fix parameters

`model.eval()`

```python
agent.network.eval()
# 任何操作之前，先重置环境
state=env.reset()
#展示图形界面
img = plt.imshow(env.render(mode='rgb_array'))
total_reward=0
done=False
while not done:
    # 让agent选取一个动作
    action,_=agent.sample(state)
    #前进一步
    new_state,reward,done,_=env.step(action)
    #更新当前状态，总的回报
    state=new_state
    total_reward+=reward
    #更新图形界面
    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
```


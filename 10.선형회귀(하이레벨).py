#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import Adam
from torch.nn import Linear, MSELoss, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../data/cars.csv', index_col="Unnamed: 0")
df


# In[3]:


df.corr()


# ## High level API 1
# - Data를 matrix형태로 줘야함
# - 왜 그런지는 다중 선형회귀에서 이해할 수 있음

# In[18]:


x = torch.FloatTensor(df[['speed']].values) # 특성 데이터
y = torch.FloatTensor(df[['dist']].values) # 라벨


# In[19]:


linear = Linear(1,1) # (특성 데이터의 갯수, 라벨의 갯수)


# In[20]:


list(linear.parameters()) # linear.parameters() 안에서 w와 b의 값이 주어짐 (Random하게)


# In[34]:


linear.weight


# In[35]:


linear.bias


# In[30]:


loss_fn = MSELoss()
optimizer = Adam(linear.parameters(), lr=0.1)
for step in range(1000):
    optimizer.zero_grad()
    hx = linear.forward(x) # hx = w*x+b # linear(x) <-- special 함수로 재정의 되어있어서 가능
    cost = loss_fn(hx, y)
    cost.backward()
    optimizer.step()
    print(step, cost.item())


# In[37]:


list(linear.parameters())


# In[38]:


linear.weight


# In[39]:


linear.bias


# In[41]:


linear.forward(torch.FloatTensor([10]))


# In[42]:


linear(torch.FloatTensor([10]))


# In[44]:


pred = linear(x).detach().numpy()


# In[46]:


plt.scatter(df['speed'], df['dist'])
plt.plot(df['speed'], pred, 'r--')
plt.show()


# ### 번외 Python의 special 함수

# In[31]:


class Test:
    def __init__(self):
        self.d = {}
        self.a = 10
    def __repr__(self):
        return f'a={self.a}'
    def __setitem__(self, key, value):
        print('setitem call')
        self.d[key] = value


# In[32]:


obj = Test()
print(obj) # obj.__repr__()


# In[33]:


obj['aa']=100 # obj.__setitem__('aa', 100)


# ## High level API 2
# 

# In[48]:


model = Sequential()
model.add_module('nn1', Linear(1,1))


# In[49]:


loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.1)


# In[51]:


for step in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x) # w*x+b # hx = model(x)
    cost = loss_fn(hx, y)
    cost.backward()
    optimizer.step()
    print(step, cost.item())


# In[54]:


model[0].weight


# In[55]:


model[0].bias


# In[57]:


model( torch.FloatTensor([10]))


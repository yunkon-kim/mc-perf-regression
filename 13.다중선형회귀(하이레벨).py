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


df = pd.read_csv('../data/test.csv', header=None) # Quiz1 , Quiz2, Midterm, Final
df.columns = ['q1', 'q2', 'mid', 'final']
df


# In[3]:


#df.iloc[행, 열]
x = torch.FloatTensor(df.iloc[:,:-1].values)
y = torch.FloatTensor(df.iloc[:,[-1]].values)


# In[8]:


model = Sequential()
model.add_module('nn1', Linear(3,1)) # (특성데이터 갯수, 라벨의 갯수)
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.1)


# In[14]:


for step in range(1000):
    optimizer.zero_grad()
    hx = model.forward(x) # matmul (x, w) + b # model(x) 가능
    cost = loss_fn(hx, y)
    cost.backward()
    optimizer.step()
    print(step, cost.item())


# In[22]:


model[0].weight


# In[16]:


model[0].bias


# In[23]:


model(torch.FloatTensor([[80,90,90]]))


# In[25]:


model(torch.FloatTensor([[80,90,90], [70,50,50]]))


# In[37]:


pred = model(x)
pred = pred.detach().numpy()


# In[38]:


x_axis = torch.arange(0, len(df['q1']))
plt.scatter(x_axis, df['final'])
plt.plot(x_axis, pred, 'r--') # r-- : Red, dashed line
plt.show()


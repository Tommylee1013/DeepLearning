#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

epochs = 10000
learning_rate = 0.05

train_inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_out = np.array([0, 1, 1, 0])

W1 = np.random.randn(2,2)
W2 = np.random.randn(2,1)
b1 = np.random.randn(1,2)
b2 = np.random.randn(1,1)

def tanh(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator / denominator

def BP(w, y, x = 1) :
    Y = (1 - y) * (1 + y)
    return w * Y * x

def MSE(t, y) :
    return (y - t) ** 2

errors = []
for epoch in range(epochs):
    for batch in range(4):
        idx = random.randint(0,3)
        xin = train_inp[idx].reshape(1,2)
        ans = train_out[idx]
        
        net1 = tanh(np.matmul(xin,W1) + b1)
        net2 = tanh(np.matmul(net1,W2) + b2)
        
        loss = MSE(ans, net2)
        
        delta_W1 = np.zeros((2,2))
        delta_W2 = np.zeros((2,1))
        delta_b1 = np.zeros((1,2))
        delta_b2 = np.zeros((1,1))
        
        weight = ans - net2
        
        delta_W1 = - BP(weight, net2) * BP(W2.T, net1, x = xin.T)
        delta_W2 = - BP(weight, net2, x = net1.T)
        delta_b1 = - BP(weight, net2) * BP(W2.T, net1)
        delta_b2 = - BP(weight, net2)
        
        W1 = W1 - learning_rate * delta_W1
        W2 = W2 - learning_rate * delta_W2
        
        b1 = b1 - learning_rate * delta_b1
        b2 = b2 - learning_rate * delta_b2
        
    if epoch % 500 == 0 : 
        print("epoch[%i/%i] loss: %.4f" % (epoch, epochs, float(loss)))
        
    errors.append(loss)

loss =  np.array(errors)
plt.plot(loss.reshape(epochs))
plt.xlabel("epoch")
plt.ylabel("loss")

for idx in range(4):
    xin = train_inp[idx]
    ans = train_out[idx]
    
    net1 = tanh(np.matmul(xin,W1)+b1)
    net2 = tanh(np.matmul(net1,W2)+b2)

    pred = net2
    
    print("input: ", xin, ", answer: ", ans, ", pred: {:.4f}".format(float(pred)))
    
np.savetxt("20180594_layer2_weight",(W1, W2, b1, b2),fmt="%s")


# In[4]:


import numpy as np
import random
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

epochs = 10000
learning_rate = 0.05

train_inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_out = np.array([0, 1, 1, 0])

W1 = np.random.randn(2,1)
b1 = np.random.randn(1,1)

def tanh(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator / denominator

def BP(w, y, x = 1) :
    Y = (1 - y) * (1 + y)
    return w * Y * x

def MSE(t, y) :
    return (y - t) ** 2

errors = []
for epoch in range(epochs):
    for batch in range(4):
        idx = random.randint(0,3)
        xin = train_inp[idx].reshape(1,2)
        ans = train_out[idx]
        
        net1 = tanh(np.dot(xin, W1) + b1)

        loss = MSE(ans, net1)
        
        delta_W1 = np.zeros((2,1))
        delta_b1 = np.zeros((1,2))
        
        weight = ans - net1
        
        delta_W1 = - BP(weight, net1)
        delta_b1 = - BP(weight, net1)
        
        W1 = W1 - learning_rate * delta_W1
        
        b1 = b1 - learning_rate * delta_b1
        
    if epoch % 500 == 0 : 
        print("epoch[%i/%i] loss: %.4f" % (epoch, epochs, float(loss)))
        
    errors.append(loss)

loss =  np.array(errors)
plt.plot(loss.reshape(epochs))
plt.xlabel("epoch")
plt.ylabel("loss")

for idx in range(4):
    xin = train_inp[idx]
    ans = train_out[idx]
    
    net1 = tanh(np.matmul(xin,W1)+b1)

    pred = net1
    
    print("input: ", xin, ", answer: ", ans, ", pred: {:.4f}".format(float(pred)))

np.savetxt("20180594_layer1_weight",(W1, b1),fmt="%s")


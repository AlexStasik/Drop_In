#!/usr/bin/env python
# coding: utf-8

# In[30]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[31]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[32]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[33]:


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# In[34]:


scale = 1.

mask = list()
for i, param in enumerate(net.parameters()):
    if i<4:
        m = np.ones((param.detach().numpy()).shape)
    else:
        m = np.random.binomial(1, scale, size=(param.detach().numpy()).shape)
    mask.append(torch.tensor(m))


# In[35]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[36]:


losses = list()


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for p, m in zip(net.parameters(), mask):
            p.grad *= m
        
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            losses.append(running_loss / 2000)
            running_loss = 0.0


            
print('Finished Training')

np.save('loss_hist/losses_'+str(int(1/scale))+'.npy', np.array(losses))


# In[ ]:


outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()


# In[ ]:


print(inputs.grad)


# In[ ]:


grads = []
for param in net.parameters():
    grads.append(param.grad.view(-1))


# In[ ]:


paras = []
for param in net.parameters():
    paras.append(param.view(-1))


# In[ ]:


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# In[ ]:


dataiter = iter(testloader)
images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[ ]:


net = Net()
net.load_state_dict(torch.load(PATH))


# In[15]:


outputs = net(images)


# In[16]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# In[17]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[18]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[19]:


from scipy import stats
import pickle

paras = list()
dists = list()
for param in net.parameters():
    paras.append((param.view(-1)).detach().numpy())
    weights = np.squeeze(paras[-1].flatten())
    dists.append(stats.gaussian_kde(weights))
    
pickle.dump( dists, open( "dists.pkl", "wb" ) )


# In[20]:


# for p, d in zip(paras, dists):
#     x_plot = np.linspace(p.min(), p.max(), 100, endpoint=True)
#     plt.figure()
#     plt.hist(p, bins=100, density=True)
#     plt.plot(x_plot, d.pdf(x_plot))
#     plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self, scale_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, int(120/scale_size))
        self.fc2 = nn.Linear(int(120/scale_size), int(84/scale_size))
        self.fc3 = nn.Linear(int(84/scale_size), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def main(scale=1, scale_size=1, ):
    
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
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    
    net = Net(scale_size)
    
    
    mask = list()
    for i, param in enumerate(net.parameters()):
        if i<4:
            m = np.ones((param.detach().numpy()).shape)
        else:
            m = np.random.binomial(1, scale, size=(param.detach().numpy()).shape)
        mask.append(torch.tensor(m))
        
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    
    
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

    np.save('loss_hist/losses_'+str(int(scale))+'_'+str(scale_size)+'.npy', np.array(losses))
    
    
    
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    grads = []
    for param in net.parameters():
        grads.append(param.grad.view(-1))
        
    paras = []
    for param in net.parameters():
        paras.append(param.view(-1))
        
        
    PATH = './cifar_net_'+str(int(scale))+'_'+str(scale_size)+'.pth'
    torch.save(net.state_dict(), PATH)
    
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    net = Net(scale_size)
    net.load_state_dict(torch.load(PATH))
    
    outputs = net(images)
    
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    
    
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    
    
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

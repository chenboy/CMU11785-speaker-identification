# coding: utf-8

# In[11]:

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as utils

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import os
from random import randint

# In[2]:

#read all f
npyFile = os.listdir("/home/bowei/MFCC_overlap")
model_path = 'model.pt'

# In[3]:

#get file for each speaker
dic = {}
for i in npyFile:
    speaker = i.split("-")[0]
    if speaker not in dic:
        dic[speaker] = []
    dic[speaker].append(i)

# In[6]:

#map the speaker to index
file = {}
indexMap = {}
speakerNum = 0
for k, v in dic.items():
    for i in v:
        file[i] = k
    if k not in indexMap:
        indexMap[k] = speakerNum
        speakerNum += 1

print(len(file))
print(speakerNum)

# In[7]:

#for each audio, use the first 2w dimension, split it to 10 parts
X = []
Y = []
for i in npyFile:
    data = np.load("/home/bowei/MFCC_overlap/" + i, encoding='bytes')

    speaker = i.split("-")[0]
    for n in range(10):
        X.append(data[2000 * n:2000 * (n + 1):].transpose())
        Y.append(speaker)

# In[8]:

#number of samples
print(len(X))
print(len(Y))

# In[12]:


class MyDataset(Dataset):
    def __init__(self, trainX, trainY, transform=None):
        X = trainX
        Y = trainY
        self.buildInput(X, Y)

    def buildInput(self, X, Y):
        size = len(X)
        inputs1 = []
        inputs2 = []

        label1 = []
        label2 = []
        outputs = []

        #get index for same speaker
        count = {}
        for i in range(size):
            if Y[i] not in count:
                count[Y[i]] = []

            count[Y[i]].append(i)

        for i in range(size):
            truth = Y[i]
            inputs1.append(X[i])
            inputs2.append(X[i])

            label1.append(indexMap[Y[i]])
            label2.append(indexMap[Y[i]])
            outputs.append(1)

            #for each audio, select two audio from same speaker
            for n in range(2):
                index = randint(0, len(count[Y[i]]) - 1)
                v = count[Y[i]][index]

                inputs1.append(X[i])
                inputs2.append(X[v])

                label1.append(indexMap[Y[i]])
                label2.append(indexMap[Y[v]])
                outputs.append(1)

            #for each audio, select two audio from different speaker
            for n in range(2):
                v = randint(0, size - 1)
                while v in count[Y[i]]:
                    v = randint(0, size - 1)
                inputs1.append(X[i])
                inputs2.append(X[v])

                label1.append(indexMap[Y[i]])
                label2.append(indexMap[Y[v]])
                outputs.append(-1)

        #i1,i2 is input audio, l1,l2 is speaker label, o is whether they are from same speaker
        self.i1 = inputs1
        self.i2 = inputs2
        self.l1 = label1
        self.l2 = label2
        self.o = outputs

    def __getitem__(self, index):
        return self.i1[index], self.i2[index], self.l1[index], self.l2[
            index], self.o[index]

    def __len__(self):
        return len(self.i1)


# In[13]:


def collate_fn(batch):
    inputs1 = []
    inputs2 = []
    labels1 = []
    labels2 = []
    outputs = []

    for i1, i2, l1, l2, o in batch:
        inputs1.append(i1)
        inputs2.append(i2)
        labels1.append(l1)
        labels2.append(l2)
        outputs.append(o)

    I1s = torch.from_numpy(np.array(inputs1))
    I2s = torch.from_numpy(np.array(inputs2))

    L1s = torch.from_numpy(np.array(labels1))
    L2s = torch.from_numpy(np.array(labels2))
    Os = torch.from_numpy(np.array(outputs))

    return I1s, I2s, L1s, L2s, Os


# In[14]:

dataset = MyDataset(X, Y)
dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# In[25]:


#flatten the cnn output
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


#cnn part
net = nn.Sequential(
    #nn.Dropout(p=0.2),
    nn.Conv1d(in_channels=12, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=3),

    #nn.Dropout(p=0.5),
    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6),
    nn.ReLU(),
    #nn.MaxPool1d(kernel_size=2, padding=1, stride=1),
    nn.MaxPool1d(kernel_size=6),
    nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
    nn.ReLU(),
    #nn.MaxPool1d(kernel_size=2, padding=1, stride=1),
    nn.MaxPool1d(kernel_size=3),
    Flatten(),
    nn.Linear(2304, 5000),
    nn.Sigmoid(),
    nn.Linear(5000, 1500),
    nn.Sigmoid())

#classifier for speaker
speakerClassifier = nn.Linear(1500, speakerNum)

#identify whether they are from same speaker
identifier = nn.Sequential(
    nn.Linear(1, 50),
    nn.Sigmoid(),
    nn.Linear(50, 2),
)

# In[26]:


#init weight for CNN
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal(m.weight.data)


net.apply(weight_init)

# In[28]:

#cross entropy loss
criterion = nn.CrossEntropyLoss()

# In[29]:

net.cuda()
speakerClassifier.cuda()
identifier.cuda()
criterion.cuda()

# In[30]:

optimizer = torch.optim.SGD(
    list(net.parameters()) + list(speakerClassifier.parameters()) +
    list(identifier.parameters()),
    lr=0.003,
    momentum=0.8)

print("training classifier...")
if os.path.exists(model_path):
    net.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), model_path),
            map_location=lambda storage, loc: storage))
# In[ ]:
else:
    for epoch in range(20):
        sumLoss = 0
        count = 0
        n_samples = 0
        n_correct = 0
        for i1, i2, l1, l2, o in dataloader:
            v1 = Variable(i1).float().cuda()
            v2 = Variable(i2).float().cuda()
            l1 = Variable(l1).cuda()
            l2 = Variable(l2).cuda()
            o = Variable(o).cuda()

            #print(v1.shape)
            feature = net(v1)

            result = speakerClassifier(feature)
            n_samples += result.shape[0]
            pred = result.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(l1.data.view_as(pred)).sum()

            loss1 = criterion(result.contiguous().view(-1, speakerNum),
                              l1.view(-1))
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            l_1 = loss1.cpu().data.numpy()
            sumLoss += l_1
            count += 1
            #print(str(l_1))
            if count % 100 == 0:
                print("accuracy : ", float(n_correct) / float(n_samples))
        print(sumLoss / count)
    torch.save(net.state_dict(), model_path)

# In[ ]:

print("training cosine loss func...")
lossFunc = nn.CosineEmbeddingLoss()

# In[ ]:

for epoch in range(30):
    sumLoss = 0
    total_samples = 0
    correct_samples = 0
    num_batch = 0
    for i1, i2, l1, l2, o in dataloader:
        v1 = Variable(i1).float().cuda()
        v2 = Variable(i2).float().cuda()
        l1 = Variable(l1).cuda()
        l2 = Variable(l2).cuda()
        o = Variable(o).cuda()

        feature1 = net(v1)
        feature2 = net(v2)
        dist = F.cosine_similarity(feature1, feature2)
        correct_samples += torch.sum(dist * (
            o > 0).type(torch.cuda.FloatTensor) > 0.5)[0].cpu().data.numpy()[0]
        correct_samples += torch.sum(dist * (
            o < 0).type(torch.cuda.FloatTensor) < 0.5)[0].cpu().data.numpy()[0]
        total_samples += feature1.shape[0]

        loss = lossFunc(feature1, feature2, o)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l = loss.cpu().data.numpy()[0]
        sumLoss += l
        num_batch += 1
    torch.save(net.state_dict(), model_path)
    print(float(correct_samples) / float(total_samples))
    print(sumLoss / num_batch)

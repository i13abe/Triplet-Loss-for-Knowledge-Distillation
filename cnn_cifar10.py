import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F
import sys

argvs = sys.argv


torch.cuda.manual_seed_all(int(sys.argv[1]))
torch.manual_seed(int(sys.argv[1]))
np.random.seed(int(sys.argv[1]))
data_train = dsets.CIFAR10(".", download=True, train=True)
x_train = []
y_train = []
x_test = []
y_test = []

data_test = dsets.CIFAR10(".", download=True, train=False)



for i in range(len(data_train)):
	x_train.append(np.array(data_train[i][0]))
	y_train.append(data_train[i][1])
	
for i in range(len(data_test)):
	x_test.append(np.array(data_test[i][0]))
	y_test.append(data_test[i][1])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.transpose(0,3,1,2)

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.transpose(0,3,1,2)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255.0
x_test /= 255.0


n_class = 10
class_num_train = 5000




datasize = len(y_train)
datasize_test = len(y_test)
epoch=500
batchsize=100


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128, 10)




    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Network()
net.cuda()
softmax_cross_entropy = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.3)

train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]


for epoch in range(1, epoch+1):
	print('epoch', epoch)
	perm = np.random.permutation(datasize)
	sum_score = 0
	sum_loss = 0
	net.train()
	for i in range(0, datasize, batchsize):
		x_batch = x_train[perm[i:i+batchsize]]
		y_batch = y_train[perm[i:i+batchsize]]
		x_batch = torch.from_numpy(x_batch).float().cuda()
		y_batch = torch.from_numpy(y_batch).long().cuda()
		optimizer.zero_grad()
		y = net(x_batch)
		loss = softmax_cross_entropy(y, y_batch)           
		loss.backward()                           
		optimizer.step()


	sum_score = 0
	sum_loss = 0
	net.eval()
	for i in range(0, datasize, batchsize):
		x_batch = x_train[i:i+batchsize]
		y_batch = y_train[i:i+batchsize]
		x_batch = torch.from_numpy(x_batch).float().cuda()
		y_batch = torch.from_numpy(y_batch).long().cuda()
		y = net(x_batch)
		loss = softmax_cross_entropy(y, y_batch)
		sum_loss += float(loss.cpu().data.item()) * batchsize
		_, predict = y.max(1)
		sum_score += predict.eq(y_batch).sum().item()
	print("train  mean loss={}, accuracy={}".format(sum_loss / datasize, sum_score / datasize))
	train_loss.append(sum_loss / datasize)
	train_acc.append(sum_score / datasize)


	sum_score = 0
	sum_loss = 0
	net.eval()
	for i in range(0, datasize_test, batchsize):
		x_batch = x_test[i:i+batchsize]
		y_batch = y_test[i:i+batchsize]
		x_batch = torch.from_numpy(x_batch).float().cuda()
		y_batch = torch.from_numpy(y_batch).long().cuda()
		y = net(x_batch)
		loss = softmax_cross_entropy(y, y_batch)
		sum_loss += float(loss.cpu().data.item()) * batchsize
		_, predict = y.max(1)
		sum_score += predict.eq(y_batch).sum().item()
	print("test  mean loss={}, accuracy={}".format(sum_loss / datasize_test, sum_score / datasize_test))
	test_loss.append(sum_loss / datasize_test)
	test_acc.append(sum_score / datasize_test)
	scheduler.step()

fo = open('cnn_cifar10.txt', 'a')
sys.stdout = fo

print("No.", argvs[2])
print("train accuracy:", train_acc[epoch-1])
print("test accuracy:", test_acc[epoch-1])

plt.figure(figsize=(6,6))


plt.plot(range(epoch), train_loss)
plt.plot(range(epoch), test_loss, c='#00ff00')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("cnn_cifar10_loss(No.%s).png"%argvs[2])
plt.close()


plt.plot(range(epoch), train_acc)
plt.plot(range(epoch), test_acc, c='#00ff00')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("cnn_cifar10_accuracy(No.%s).png"%argvs[2])
plt.close()



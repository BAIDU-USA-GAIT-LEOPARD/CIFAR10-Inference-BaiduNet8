# - *- coding: utf- 8 - *-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from BaiduNet8 import BaiduNet8
from torch.optim.lr_scheduler import MultiStepLR
import time
import numpy as np
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 300
pre_epoch = 0  
BATCH_SIZE = 128
LR = 0.1


class Cutout(object):
  def __init__(self, sz):
    self._sz = sz

  def __call__(self, img):
    h = img.size(1)
    w = img.size(2)

    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = int(np.clip(y - self._sz / 2, 0, h))
    y2 = int(np.clip(y + self._sz / 2, 0, h))
    x1 = int(np.clip(x - self._sz / 2, 0, w))
    x2 = int(np.clip(x + self._sz / 2, 0, w))
    img[:, y1:y2, x1:x2].fill_(0.0)
    return img


# preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(8),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # RGB mean and variance
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
#
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = BaiduNet8().to(device)

# Loss and optimizaton
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #
scheduler = MultiStepLR(optimizer, milestones=[140, 190], gamma=0.1)

# training
if __name__ == "__main__":
    best_acc = 0.0  # record the best acc
    print("Start training BaiduNet8!")  #
    with open("acc_V100.txt", "w") as f:
        with open("log_V100.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                scheduler.step()
                print('\nEpoch: %d' % (epoch + 1))
                start_time = time.clock()
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # data
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # batch based output
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                   
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct.item() / total))
                    f2.write('\n')
                    f2.flush()

                # evaluation after each epoch
                end_time  = time.clock()
                print (" This epoch time consumption is: ", end_time - start_time)
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    count = 0
                    total_time = 0
                    for data in testloader:
                        count = count + 1
                        net.eval()
                        images, labels = data
                        start_time = time.clock()
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)

                        end_time = time.clock()
                        duration = end_time - start_time
                       
                        total_time += duration
                        total += labels.size(0)
                        correct += (predicted == labels).sum()

                    print('test accuracy is ：%.3f%%' % (100 * correct.item() / total))
                    print (count)
                    print ("the average time for each sample is: ", total_time/10.0, " ms")
                    acc = 100. * correct.item() / total
                   
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.write("Aveage time is %.6f ms" % (total_time/10.0))
                    f.write('\n')
                    f.flush()
                    # record the best accuracy,and save its corresponding model in pt
                    if acc > best_acc:
                        f3 = open("best_acc_V100.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                        print('Saving model......')
                        example = torch.rand(1, 3, 32, 32).to(device)
                        traced_script_module = torch.jit.trace(net, example)
                        traced_script_module.save("model.pt")
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataproc import LinkDataset
import scipy.io as scio
import numpy
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(3, 10, kernel_size=)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(180, 50)#320-180
        self.fc2 = nn.Linear(50, 10)#100-10
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        batch = x.size(0)
        x = x.view(batch, -1)  
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx, data.size(), target.size())
        data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    print('Evaluation:')
    model.eval()
    test_loss = 0
    correct = 0
    sqrot = 0
    
        for data, target in test_loader:
        data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data), Variable(target)
        output = model.forward(data)
        #print('\n',len(output),output,'\n')
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1]
        #print('\n',pred,'\n')
        pred_cpu = pred.cpu()
        Predic = pred_cpu.numpy()
        #print('\n',target,'\n')
        Targ_cpu = target.cpu()
        Targlab = Targ_cpu.numpy()
        for count in range(len(target)):
            sqrot += numpy.sqrt(numpy.square(Predic[count]-Targlab[count])/Targlab[count])
        #Data_pred_dir = os.path.join(root_dir,'Predict_data',[print('%05d',count),'.mat'])
        #scio.savemat(Data_pred_dir, {'predic':Predic})
        correct += pred.eq(target.view_as(pred)).sum().item()
        

    test_loss /= len(test_loader.dataset)
    sqrot /=len(test_loader.dataset)
    #test_loss_cpu = test_loss.cpu()
    #Test_loss = test_loss_cpu.numpy()
    loss_data[epoch] = test_loss
    # print(type(sqrot), sqrot.shape, type(sqrot[0]), type(test_loss))
    # print(sqrot)
    #print('The sqrot is: {:.4f}%\n'.format(float(sqrot[0]))*100)
    print('\nTest set: Average loss: {:.4f}, The sqrot is: {:.4f}%, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, (float(sqrot[0]))*100, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        
use_cuda = True
device = 1
batch_size = 32
test_batch_size = 100
epochs = 100
lr = 0.01    
momentum = 0.5
seed = 1
log_interval=1
torch.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
root_dir = '/root/Workspace/zhangxin/link_quality_lu/data_N/'
train_dir = '/root/Workspace/zhangxin/link_quality_lu/data_N/train.txt'
test_dir = '/root/Workspace/zhangxin/link_quality_lu/data_N/test.txt'

train_loader = torch.utils.data.DataLoader(
    LinkDataset(root_dir=root_dir,
               txt_file=train_dir),
    batch_size=batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(
    LinkDataset(root_dir=root_dir,
               txt_file=test_dir,
               train=False),
    batch_size=test_batch_size, **kwargs)


model = Net().cuda(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_data = numpy.zeros(101)

for epoch in range(1, epochs + 1):
    #curpath = os.getcwd()
    #tempfile = 'Predict_data'+print('%d',epoch)
    #targetpath = curpath+os.path.sep+tempfile 
    #if not os.path.exists(targetpath):
        #os.makedirs(targetpath)
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
print(loss_data)

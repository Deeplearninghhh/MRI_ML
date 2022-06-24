
from posixpath import join
from torch.utils.data import DataLoader
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
import torch
import torchvision
import os
import numpy as np
import glob 
import pandas as pd
import random
import time
#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



image_list = []
label_list = []
Sample_ID = []

with open(Input_Path) as Input_File_List:
    All_File = Input_File_List.readlines()
    for One_Path in All_File:
        One_Message = One_Path.rstrip().split('\t')
        #print(One_Message)
        if int(One_Message[2]) > 1 and int(One_Message[4])>15 and int(One_Message[8]) >15:
            image_list.append(One_Message[7])
            label_list.append(One_Message[3])
            Sample_ID.append(One_Message[0])


image_list_A = image_list#[0:64]
label_list_A = label_list#[0:64]

class D3UnetData(Dataset):
    def __init__(self,image_list,label_list,transformer):
        self.image_list=image_list
        self.label_list=label_list
        self.transformer=transformer
        
    def __getitem__(self,index):
        image=self.image_list[index]
        label=self.label_list[index]
        
        image_ct=sitk.ReadImage(image,sitk.sitkInt16)
        label_ct=sitk.ReadImage(label,sitk.sitkInt8)
        one_size = int(sitk.GetArrayFromImage(image_ct).shape[0]/2)
        one_size_begin = one_size -8
        one_size_end = one_size + 8
        #print('++++++++++++size+++++++++++++')
        #ct_array=sitk.GetArrayFromImage(image_ct)[0:16]
        ct_array = sitk.GetArrayFromImage(image_ct)[one_size_begin: one_size_end]
        one_size = int(sitk.GetArrayFromImage(label_ct).shape[0]/2)
        one_size_begin = one_size - 8
        one_size_end = one_size + 8
        #label_array=sitk.GetArrayFromImage(label_ct)[0:16]
        label_array=sitk.GetArrayFromImage(label_ct)[one_size_begin: one_size_end]
        #print(ct_array.shape)
        label_array[label_array>0]=1
        ct_array=ct_array.astype(np.float32)
        #print('>>>>>>>>>>>>>>>>>>')
        
        ct_array=torch.FloatTensor(ct_array).unsqueeze(0)  ###[1,50,512,512]
        label_array=torch.LongTensor(label_array) ###[50,512,512]
        #ct_array = Image.fromarray(ct_array)
        #label_array = Image.fromarray(label_array)
        ct_array=self.transformer(ct_array)
        #print('Kkkk')
        label_array=self.transformer(label_array)
        #print(ct_array.shape)
        return ct_array,label_array
        
    def __len__(self):
        return len(self.image_list)

class D3UnetData_test(Dataset):
    def __init__(self,image_list,label_list,transformer):
        self.image_list=image_list
        self.label_list=label_list
        self.transformer=transformer
        
    def __getitem__(self,index):
        image=self.image_list[index]
        label=self.label_list[index]
        
        image_ct=sitk.ReadImage(image,sitk.sitkInt16)
        label_ct=sitk.ReadImage(label,sitk.sitkInt8)
        #print(image_ct.shape)
        
        one_size = int(sitk.GetArrayFromImage(image_ct).shape[0]/2)
        one_size_begin = one_size -8
        one_size_end = one_size + 8
        #print('++++++++++++size+++++++++++++')
        #ct_array=sitk.GetArrayFromImage(image_ct)[0:16]
        ct_array = sitk.GetArrayFromImage(image_ct)[one_size_begin: one_size_end]
        one_size = int(sitk.GetArrayFromImage(label_ct).shape[0]/2)
        one_size_begin = one_size - 8
        one_size_end = one_size + 8
        label_array=sitk.GetArrayFromImage(label_ct)[one_size_begin: one_size_end]
        #ct_array=sitk.GetArrayFromImage(image_ct)[0:30]
        #label_array=sitk.GetArrayFromImage(label_ct)[0:30]
        
        label_array[label_array>0]=1
        
        ct_array=ct_array.astype(np.float32)
        
        ct_array=torch.FloatTensor(ct_array).unsqueeze(0)  ###[1,50,512,512]
        label_array=torch.LongTensor(label_array) ###[50,512,512]
        
        ct_array=self.transformer(ct_array)
        label_array=self.transformer(label_array)
        
        return ct_array,label_array
        
    def __len__(self):
        return len(self.image_list)


transformer=transforms.Compose([
    transforms.Resize((96,96)),
])


test_ds=D3UnetData_test(image_list_A,label_list_A,transformer)
test_dl=DataLoader(test_ds,batch_size=2,shuffle=True)



class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_groups=8):
        super(DoubleConv,self).__init__()
        self.double_conv=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups,num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(num_groups=num_groups,num_channels=out_channels),
            nn.ReLU(inplace=True),
        )       
    def forward(self,x):
        return self.double_conv(x)
    

#net=DoubleConv(1,64,num_groups=8)
#out=net(img)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)          
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask, x5

model=UNet3d(1,2,16).cuda()
#img,label=next(iter(train_dl))
#img=img.cuda()
#pred, pred5=model(img)
#pred.shape


loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.00001)

from tqdm import tqdm
def train(model, model_Path, testloader): 
    test_correct = 0
    test_total = 0
    test_running_loss = 0 
    epoch_test_iou = []
    table_Pred = []
    i = 0
    
    model.load_state_dict(torch.load(model_Path))
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(testloader):

            x, y = x.to('cuda'), y.to('cuda')
            y_pred, pred5 = model(x)
            #print(pred5.shape)
            #print(pred5[0].shape)
            pred6 = pred5[0].reshape(-1)
            one_pred = pred6.cpu().numpy().copy()
            table_Pred.append(one_pred)
            pred7 = pred5[1].reshape(-1)
            one_pred = pred7.cpu().numpy().copy()
            table_Pred.append(one_pred)
            
            i = i + 2
            
            
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            
            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            epoch_test_iou.append(batch_iou.item())
            
    
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total*96*96*50)
    

    print('test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3),
           'test_iou:', round(np.mean(epoch_test_iou), 3)
             )
        
    return table_Pred,i


epochs = 200

train_loss = []
train_acc = []
test_loss = []
test_acc = []

model_Path = '199_trainIOU_0.801_testIOU_0.805_Label_T1C.pth'
All_Table, len_i = train(model, model_Path, test_dl)

Model_Path = 'Data_T1C.csv'

#print(len(Sample_ID))
#print(All_Table)
#print(len(All_Table))
#print(len_i)

Model_Data = open(Model_Path, 'w')
for i in range(len_i):
    One_Table = All_Table[i]
    One_Str = Sample_ID[i]
    for One_num in One_Table:
        #print(One_num)
        One_Str = One_Str + ',' + str(One_num)
    One_Floor = One_Str + '\n'
    Model_Data.write(One_Floor)



fname = ''
df = pd.read_csv('{0}.tab'.format(fname),sep='\t')
labels = df[['RT']]
print(features.shape, labels.shape, labels)
# df to array
X = features.values
y = labels.values
#X = VarianceThreshold(threshold=5).fit_transform(X)
X = MinMaxScaler().fit_transform(X)
X = VarianceThreshold(threshold=0.05).fit_transform(X)
X = StandardScaler().fit_transform(X)
y = MinMaxScaler().fit_transform(y)
#y = StandardScaler().fit_transform(y)
#X = VarianceThreshold(threshold=1).fit_transform(X)
print(X.shape , y.shape)


# split to train/test 
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
print(train_X.shape,test_X.shape, train_y.shape, test_y.shape )


# construct torch datasets
batch_size = 64
train_dataset = data.TensorDataset(torch.from_numpy(train_X).to(torch.float32),torch.from_numpy(train_y).to(torch.float32))
train_dataloader = data.DataLoader(train_dataset,batch_size,shuffle=True)
print(len(train_dataset),len(train_dataloader))

test_dataset = data.TensorDataset(torch.from_numpy(test_X).to(torch.float32),torch.from_numpy(test_y).to(torch.float32))
test_dataloader = data.DataLoader(test_dataset,batch_size,shuffle=True)
print(test_dataloader.dataset[0])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()#继承
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6144, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            #nn.Sigmoid(),
            #nn.Linear(100, 10),
            #nn.Sigmoid(),
            nn.Linear(512, 1)
            #nn.Linear(1444,1)
        )
        
    def forward(self,x):
        x=self.linear_relu_stack(x)
        return x
# model = Net().to(device)
model = Net()
print(model)


loss_fn=nn.MSELoss()
learning_rate = 1e-4
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    current = 0
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)       
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)#*10

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss= loss.item()
        current += len(X)
        #print(current)
        #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
            pred = model(X)
            #print(pred)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches # average loss in each batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def predict(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
            pred = model(X)
            #print(pred)
            #print(X)
            #print(pred)



epochs = 20000

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


predict(test_dataloader, model, loss_fn)

model.eval()
x,y = test_dataset[:][0], test_dataset[:][1]
pred = model(x)

y_test = [i[0] for i in y.tolist()]
pred = [i[0] for i in pred.tolist()]
print([[y_test[i],pred[i]] for i in range(10)])




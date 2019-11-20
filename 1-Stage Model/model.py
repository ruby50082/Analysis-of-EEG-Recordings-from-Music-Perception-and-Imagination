import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True
import torch.nn.functional as F
import torch.autograd.variable as Variable

class feature_blending(nn.Module):
    def __init__(self):
        super(feature_blending, self).__init__()
        self.temporal_conv= nn.Sequential(
                #nn.Dropout(p=0.5),
                nn.Conv2d(30,30,kernel_size=(10,1)),
                nn.AvgPool2d(kernel_size=(3,1),stride=(3,1)),
                nn.BatchNorm2d(30,momentum=0.1,affine=True,eps=1e-5),
                nn.ELU())
        self.SqEx = nn.Sequential(
                    nn.Linear(3,3,bias=True),
                    nn.ELU(),
                    nn.Linear(3,3,bias=True),
                    nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _upsampling(self,x,size):
        return F.upsample(x,size,mode='bilinear')
    
    def forward(self,x):
        # Block 1:1125
        #print(x.shape)
        out1 = self.temporal_conv(x)
        #print('out1',out1.shape)
        # Block 2:369
        out2 = self.temporal_conv(out1)
        #print('out2',out2.shape)
        # Block 3:120
        #out3 = self.temporal_conv(out2)
        #print('out3',out3.shape)        
        # Block 4:37
        #out4 = self.temporal_conv(out3)
        #print('out3',out3.shape)
        out1 = self._upsampling(out1,size=(x.shape[2],x.shape[3])).permute(0,3,2,1)
        #print('out1',out1.shape)
        out2 = self._upsampling(out2,size=(x.shape[2],x.shape[3])).permute(0,3,2,1)
        #print('out2',out2.shape)
        #out3 = self._upsampling(out3,size=(x.shape[2],x.shape[3]))
        #out4 = self._upsampling(out4,size=(x.shape[2],x.shape[3]))
        y = torch.cat((x.permute(0,3,2,1),out1,out2),1)#bs*3*time*eeg
        #print('y',y.shape)
        #y1 = self.SqEx(self.avg_pool(y).permute(0,3,2,1)).permute(0,3,2,1)
        #print('0',x.shape,out1.shape,out2.shape)
        #y = x.permute(0,3,2,1) * y1[0][0]+out1 * y1[0][1]+out2 * y1[0][2]
        #print('0',y.shape)
        return y
class Classifier_Module(nn.Module):

    def __init__(self,rate,kernel_size, icol,ocol):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        padding_series=[rate*0,rate*1,rate*2]
        dilation_series=[1+rate*0,1+rate*2/(kernel_size[0]-1),1+rate*4/(kernel_size[0]-1)]
        for dilation, padding in zip(dilation_series, padding_series):#spatial pyramid pooling
            #print('stride',stride)
            self.conv2d_list.append(nn.Conv2d(icol, ocol, kernel_size=kernel_size, stride=(1,1), padding=(padding,0), dilation=(dilation,1), bias = True))

        for m in self.conv2d_list:#normalization
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print('input.shape',x.shape)
        out = self.conv2d_list[0](x)
        #print('0',out.shape)
        for i in range(len(self.conv2d_list)-1):
            #print(i+1,self.conv2d_list[i+1](x).shape)
            out += self.conv2d_list[i+1](x)
        return out # add all feature produced by different rate of kernel

class Fully_Conv(nn.Module):
    def __init__(self):
        super(Fully_Conv,self).__init__()
        self.spatial = nn.Sequential(
             nn.Conv2d(64,25,kernel_size=(1,1),stride = (1,1)),
        )
        self.conv2d=nn.Sequential(
            #  nn.Dropout(p=0.5),
             nn.Conv2d(1,100,kernel_size=(1,25),stride=(1,10),bias=False),
             nn.AvgPool2d(kernel_size=(1,3),stride=(1,3)),
             nn.Conv2d(100,300,kernel_size=(25,1),stride=(1,1),bias=False),
             nn.BatchNorm2d(num_features=300,momentum=0.99,affine=True,eps=1e-3),
             nn.ELU(),
        )
        self.temporal1=nn.Sequential(
            #  nn.Dropout(p=0.5),
             #print('0'),
             nn.Conv2d(300,50,kernel_size=(1,10),stride=(1,3),bias=False),
             #print('1'),
             nn.AvgPool2d(kernel_size=(1,1),stride=(1,1)),
             #print('2'),
             nn.BatchNorm2d(num_features=50,momentum=0.99,affine=True,eps=1e-3),
             #print('3'),
             nn.ELU(),
             #print('4'),
        )
        self.temporal2=nn.Sequential(
            #  nn.Dropout(p=0.5),
             nn.Conv2d(50,50,kernel_size=(1,4),stride=(1,1),bias=False),
             nn.AvgPool2d(kernel_size=(1,3),stride=(1,1)),
             nn.BatchNorm2d(num_features=50,momentum=0.99,affine=True,eps=1e-3),
             nn.ELU(),
        )
        self.temporal3=nn.Sequential(
             #nn.Dropout(p=0.5),
             nn.Conv2d(50,50,kernel_size=(1,4),stride=(1,1),bias=False),
             nn.AvgPool2d(kernel_size=(1,3),stride=(1,1)),
             nn.BatchNorm2d(num_features=50,momentum=0.99,affine=True,eps=1e-3),
             nn.ELU(),
        )
        self.temporal4=nn.Sequential(
             #nn.Dropout(p=0.5),
             nn.Conv2d(50,50,kernel_size=(1,1),stride=(1,1),bias=False),
             nn.AvgPool2d(kernel_size=(1,2),stride=(1,2)),
             nn.BatchNorm2d(num_features=50,momentum=0.99,affine=True,eps=1e-3),
             nn.ELU(),
        )
        self.temporal5=nn.Sequential(
             #nn.Dropout(p=0.5),
             nn.Conv2d(50,50,kernel_size=(1,5),stride=(1,1),bias=False),
             nn.AvgPool2d(kernel_size=(1,3),stride=(1,3)),
             nn.BatchNorm2d(num_features=50,momentum=0.99,affine=True,eps=1e-3),
             nn.ELU(),
        )
        self.conv_classifier=nn.Sequential(
             nn.Conv2d(50,12,kernel_size=(1,6),stride=(1,1),bias=True), #50,3,1,3,1,1    #3s:(1,6) 4s:(1,8) 5s:(1,11) 6s:(1,14)
             nn.Softmax()
        )
        self.lstm = nn.Sequential(
            nn.Dropout(p=0.5),
            lstm(input_size=121,hidden_size=2,output_size=4,num_layers=7),
            #nn.BatchNorm1d(num_features=50,momentum=0.1,affine=True,eps=1e-5),
            #nn.ELU()
        )
        self.lstm1 = nn.Sequential(
            nn.Dropout(p=0.5),
            lstm(input_size=37,hidden_size=1,output_size=4,num_layers=2),
        )
        self.lstm2 = nn.Sequential(
            nn.Dropout(p=0.5),
            lstm(input_size=16,hidden_size=2,output_size=4,num_layers=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(32*1*102,12),
            nn.Softmax()
        )
            
        # slef.linear = nn.Linear(132, 132)

    def forward(self,x):
        #print('input ',x.size())
        x = x.unsqueeze(2)
        #print('input squeeze: ',x.size())
        x = self.spatial(x)
        x = x.permute(0,2,1,3)
        #print('1',x.shape)
        #print('spatial ', x.shape)
        x = self.conv2d(x)
        #print('conv ', x.shape)
        x = self.temporal1(x)
        
        #print('1', x.shape)
        #y = x.squeeze(3)
        #lstm_out = self.lstm(y).unsqueeze(2).unsqueeze(3)
        
        #x = self.temporal2(x)
        
        # print('2',x.shape)
        # y = x.squeeze(3)
        # lstm_out = self.lstm1(y).unsqueeze(2).unsqueeze(3)
        #print('4',y.shape)
        
        #x = self.temporal3(x)
        
        # print('3',x.shape)
        #y = x.squeeze(3)
        #lstm_out = self.lstm2(y).unsqueeze(2).unsqueeze(3)
        #print('1',y.shape)
        
        #x = self.temporal4(x)
        
        # print('4',x.shape)
        
        # x = self.temporal5(x)
        
        #print('5',x.shape)
        x = self.conv_classifier(x) #output=(batch_size, 12)
        #print('classify',x.shape)
        #x=x[:,:,:,0:1]
        x = x.squeeze(3)
        #print('x.size0',x.shape)
        x = x.squeeze(2)
        #print('x.size1',x.shape)
        #print('x.view',x.view(x.size(0), -1))
        # print('x.view.shape',x.view(x.size(1), -1).shape)
        #x = torch.cat((x,lstm_out),dim=1)
        #x = self.linear(x.view(x.size(0),-1))
        #print('linear',x.shape)
        #x=x.squeeze(2)
        #x=np.delete(x,[0:x.shape[2]],axis=2)
        #print('ret',x.shape)
        return x




    
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size=None):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers)
        self.proj = nn.Linear(50, output_size)

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).float().cuda(),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).float().cuda())

    def forward(self, x):
        #print('0',x.shape)
        self.batch_size = x.shape[1]
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print('1',lstm_out.size())
        F.dropout(x,p=0.5)
        lstm_out = self.proj(lstm_out.view(x.size(0),-1))
        #print('2',lstm_out.shape)
        return lstm_out

# CY add
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        # self.T = 1760
        
        # Layer 1 (input, output, kernal)
       
        self.conv1 = nn.Conv2d(1, 16, (1, 64),(1,1))
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        #permute (batch,1->16,channels,samples)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(64, 25, (2,32))
        self.batchnorm2 = nn.BatchNorm2d(25, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(25, 12, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(12, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(12*2*106, 1)

        self.conv_classifier=nn.Sequential(
             nn.Conv2d(50,12,kernel_size=(1,3),stride=(1,1),bias=True),
             #nn.Conv2d(12,12,kernel_size=(2,92),stride=(1,1),bias=True),
             nn.Softmax()
        )
        self.linear = nn.Sequential(
            nn.Linear(32*1*102,12),
            nn.Softmax()
        )
               

    def forward(self, x):
        # Layer 1
        print('input ',x.shape)
        x = x.unsqueeze(1)
        print('sque_input ',x.shape)
        
        x = F.elu(self.conv1(x))
        print('conv1 ',x.shape)   
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0,2,1,3)
        print('permute ',x.shape)
        # Layer 2
        x = self.padding1(x)
        print('pad1 ',x.shape)
        x = F.elu(self.conv2(x))
        print('conv2 ',x.shape)
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        print('pool2 ',x.shape)
        
        # Layer 3
        x = self.padding2(x)
        print('pad2 ',x.shape)
        x = F.elu(self.conv3(x))
        print('conv3 ',x.shape)
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        print('pool3 ',x.shape)
        
        # FC Layer
        print(x.shape)
        x = x.view(-1, 12*2*106)
        print(x.shape)
        x = self.conv_classifier(x).squeeze(3).squeeze(2)
        print(x.shape)
        return x

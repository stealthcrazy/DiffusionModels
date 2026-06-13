import torch.nn as nn
import torch
import numpy as np
import json
import time
import datetime

from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join


from DenoisingDiffusionModel import DiffusionModel


## this is a cuda implementation

device = torch.device("cuda")



## copy pasted from Pytorch 
torch.backends.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR_DataLoader(torch.utils.data.Dataset):

    def __init__(self,dir,device):
        self.files = [f for f in listdir(dir) if isfile(join(dir, f))]
        self.Data = torch.tensor([])
        self.Labels = torch.tensor([])
        self.testData =torch.tensor([])
        self.testLabel =torch.tensor([])
        for file in self.files:
            #print(file)
            if "data" in file:
                tempData = unpickle(f"{dir}/{file}")

                self.Labels = torch.cat((self.Labels,torch.tensor(tempData[b"labels"])),0)

                self.Data = torch.cat((self.Data,torch.tensor(tempData[b"data"])),0)
    
            elif "test" in file:
                pass
        #print(self.Data.shape)
        
        self.Data = self.Data.reshape(self.Data.shape[0],3,32,32)
        self.Data = self.Data.to(torch.float32)
        #print(self.Data.shape)
        self.Data = (self.Data / 127.5) - 1
    def __len__(self):
        return self.Labels.shape[0]
        
    def __getitem__(self,index):
        return self.Data[index] , self.Labels[index]

batch_size = 64
T_N = 1024
T_DIM = 128
HEADS = 8
MODEL_DIM = 256
LAYERS = 3

Model = DiffusionModel(3,T_N,T_DIM,HEADS,MODEL_DIM,LAYERS,device)

Data = CIFAR_DataLoader("cifar-10-batches-py",device)
train_dataloader = DataLoader(Data, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()

optim = torch.optim.Adam(Model.parameters(), lr=2e-4, betas=(0.5, 0.999))


epochs = 1000
losses = []

for ep in range(epochs):

    for i, data in enumerate(train_dataloader):

        optim.zero_grad()
        X = data[0].to(device)

        eps , eps_t = Model(X)
        loss = criterion(eps_t,eps)
        loss.backward()

        optim.step()
        if (i % 500) == 0:
            losses.append(loss.item())
            info = f"Epoch {ep} : Step {i} \n: Model Loss: {loss.item()} \n"
            with open("logDiffusion.txt","a") as f:
                f.write(info)
            print("==========================")
            print(info)
            print("==========================")
        if ((ep % 50) == 0) and (ep != 0):
            ts = time.time()
            stmp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            torch.save({
                "Epoch":ep,
                "DiffusionModel" : Model.state_dict(),
                "time"    : stmp,
                "Batch_Size" : batch_size,
                'Optim': optim.state_dict(),
                'MSE' : True,
                'ADAM' : True,
                'Losses':losses,
                        }, f'Checkpoint_Meta_Diffusion.pt')
ts = time.time()
stmp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
torch.save({
                "Epoch":ep,
                "DiffusionModel" : Model.state_dict(),
                "time"    : stmp,
                "Batch_Size" : batch_size,
                'Optim': optim.state_dict(),
                'MSE' : True,
                'ADAM' : True,
                'Losses':losses,
                        }, f'Checkpoint_Meta_Diffusion.pt')
import torch.nn as nn
import torch ,torchvision
from torchvision import transforms
import numpy as np
import json
import time
import datetime

from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join


from DenoisingDiffusionModel import DiffusionModel ,EMA


## this is a cuda implementation

device = torch.device("cuda")



## copy pasted from Pytorch 
torch.backends.fp32_precision = "tf32"
#torch.backends.cudnn.conv.fp32_precision = "tf32"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True



batch_size = 64
T_N = 512
T_DIM = 64
HEADS = 8
MODEL_DIM = 128
LAYERS = 3

in_channels = 3
imgSize = 128

Model = DiffusionModel(in_channels,T_N,T_DIM,HEADS,MODEL_DIM,LAYERS,device).to(device)
Data = torchvision.datasets.CelebA('./data',
                                   transform=  transforms.Compose([
                                            transforms.Resize(imgSize),
                                            transforms.CenterCrop(imgSize),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ]) )

train_dataloader = DataLoader(Data, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()

optim = torch.optim.Adam(Model.parameters(), lr=2e-4, betas=(0.9, 0.999))

scaler = torch.amp.GradScaler()

decay = 0.9999
EMAModel = EMA(Model,decay)

epochs = 1000
losses = []

for ep in range(epochs):
    
    for i, data in enumerate(train_dataloader):
        
        optim.zero_grad()
        X = data[0].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            eps , eps_t = Model(X)
            loss = criterion(eps_t,eps)
        scaler.scale(loss).backward()

        scaler.step(optim)
        scaler.update()

        EMAModel.update(Model)

        if (i % 50) == 0:
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
                "EMA_Weights" : EMAModel.S_model.state_dict(),
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
                "EMA_Weights" : EMAModel.S_model.state_dict(),
                "time"    : stmp,
                "Batch_Size" : batch_size,
                'Optim': optim.state_dict(),
                'MSE' : True,
                'ADAM' : True,
                'Losses':losses,
                        }, f'Checkpoint_Meta_Diffusion.pt')
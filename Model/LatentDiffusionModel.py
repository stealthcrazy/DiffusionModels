import torch
import torch.nn as nn
import numpy as np



class AttentionBlock(nn.Module):
    def __init__(self,context_dim,model_dim,heads,):
        super(AttentionBlock,self).__init__()

        self.model_dim= model_dim
        self.heads = heads
        self.V1 = nn.Linear(model_dim,model_dim,bias=False)
        self.Q1 = nn.Linear(context_dim,model_dim,bias=False)
        self.K1 = nn.Linear(context_dim,model_dim,bias=False)

        self.O1 = nn.Linear(model_dim,model_dim,bias=False)

    def forward(self,Query,Key,Value):

        # Query Key Value of Shape -> Batch , H*W or ContextLen , dim
        
        V = self.V1(Value) # now shape is Batch ,  H*W or ContextLen , dim , where dim = head_dim * heads
        Q = self.Q1(Query)
        K = self.K1(Key) 

        V = V.reshape(Value.shape[0],Value.shape[1],self.heads,self.model_dim//self.heads).transpose(1,2) 
        Q = Q.reshape(Query.shape[0],Query.shape[1],self.heads,self.model_dim//self.heads).transpose(1,2) 
        K = K.reshape(Key.shape[0],Key.shape[1],self.heads,self.model_dim//self.heads).transpose(1,2) 

        
        S = (nn.functional.softmax((Q @ K.transpose(-2, -1)) / ((self.model_dim//self.heads) ** 0.5),dim = -1) @ V).transpose(1,2) 
        

        return self.O1(S.reshape(S.shape[0],S.shape[1],self.model_dim))
    







class TransformerBlock(nn.Module):
    def __init__(self,in_channels, heads, dim, context_dim, T):
        super(TransformerBlock,self).__init__()

        self.T = T
        self.dim = dim
        self.heads = heads

        self.NORM1 = nn.GroupNorm(1,in_channels)
        self.C1 = nn.Conv2d(in_channels,dim,1)
        #Reshape to h*w , dim*heads
        self.selfAtten = nn.ModuleList([
            AttentionBlock(dim,dim,heads) #self atten                
        for _ in range(T)])
        self.MLP = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim,dim *4), 
                nn.ReLU(),
                nn.Linear(dim *4,dim),
            )
        for _ in range(T)])
        self.crossAtten =nn.ModuleList([
            AttentionBlock(context_dim,dim,heads) #self atten                
        for _ in range(T)])
        # reshape to h, w , dim*heads
        self.C2 = nn.Conv2d(dim,in_channels,1)

    def forward(self,X,Context):
        N,C,H,W = X.shape
        X = self.C1(self.NORM1(X))
        X = X.reshape(N,self.dim,H*W).transpose(1,2)
        for i in range(self.T):
            S = self.selfAtten[i](X,X,X)+X
            M = self.MLP[i](S)+S
            X = self.crossAtten[i](M,Context,Context)+M
        X = X.transpose(1, 2).reshape(N,self.dim,H,W)
        return self.C2(X)
        


class Ublock(nn.Module):
     
    def __init__(self,in_channels,out_channels,time_dim,activation : nn.Module):
        super(Ublock,self).__init__()
        self.T1 = nn.Linear(time_dim,out_channels)
        self.C1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.C2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.A1 = activation()
    


    def forward(self,X,t):
        X = self.A1(self.C1(X))
        X = X + self.T1(t)[:,:,None,None]
        return self.A1(self.C2(X))
        


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t: (N,) integer timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]          # (N, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, dim)
        return self.mlp(emb)                             # (N, dim)





class UNet(nn.Module):

    def __init__(self,channels,heads,dim,T,time_dim , context_dim):
        super(UNet,self).__init__()

        self.TimeEmbd = TimeEmbedding(time_dim)

        self.M1 = Ublock(channels,64,time_dim,nn.ReLU)
       
        self.D1 = nn.MaxPool2d(2,2)
        
        self.M2 = Ublock(64,128,time_dim,nn.ReLU)
        self.T2 = TransformerBlock(128,heads,dim,context_dim,T)
        self.D2 = nn.MaxPool2d(2,2)


        self.M3 = Ublock(128,256,time_dim,nn.ReLU)
        self.T3 = TransformerBlock(256,heads,dim,context_dim,T)
        self.D3 = nn.MaxPool2d(2,2)

        self.M4 = Ublock(256,512,time_dim,nn.ReLU)
        self.T4 = TransformerBlock(512,heads,dim,context_dim,T)
        self.D4 = nn.MaxPool2d(2,2)

        self.M5 = Ublock(512,1024,time_dim,nn.ReLU)
        self.T5 = TransformerBlock(1024,heads,dim,context_dim,T)


        self.U1 = nn.ConvTranspose2d(1024,512,2,2)
        #concat connection 

        self.N1 = Ublock(1024,512,time_dim,nn.ReLU)
        self.T6 = TransformerBlock(512,heads,dim,context_dim,T)
        self.U2 = nn.ConvTranspose2d(512,256,2,2)
        #concat connection 

        self.N2 = Ublock(512,256,time_dim,nn.ReLU)
        self.T7 = TransformerBlock(256,heads,dim,context_dim,T)
        self.U3 = nn.ConvTranspose2d(256,128,2,2)
        #concat connection 

        self.N3 = Ublock(256,128,time_dim,nn.ReLU)
        self.T8 = TransformerBlock(128,heads,dim,context_dim,T)
        self.U4 = nn.ConvTranspose2d(128,64,2,2)
        #concat connection 

        self.N4 = Ublock(128,64,time_dim,nn.ReLU)
        self.O = nn.Conv2d(64,channels,1)

    def forward(self,X,Context,t):
        t = self.TimeEmbd(t)
        R1 = self.M1(X,t)
        R2 = self.T2(self.M2(self.D1(R1),t),Context)
        R3 = self.T3(self.M3(self.D2(R2),t),Context)
        R4 = self.T4(self.M4(self.D3(R3),t),Context)

        B1 = self.T5(self.M5(self.D4(R4),t),Context)
        
        C1 = torch.cat((self.U1(B1),R4),1)

        B2 = self.T6(self.N1(C1,t),Context)
        C2 = torch.cat((self.U2(B2),R3),1)
        B3 = self.T7(self.N2(C2,t),Context)
        C3 = torch.cat((self.U3(B3),R2),1)
        B4 = self.T8(self.N3(C3,t),Context)
        C4 = torch.cat((self.U4(B4),R1),1)
        O = self.O(self.N4(C4,t))
        return O


       

# can be replaced with traditional VAE 
class Encoder(nn.Module):
    def __init__(self,channels):
        super(Encoder,self).__init__()

        self.M1 = nn.Sequential(
            nn.Conv2d(3,32,5,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,channels,2,1),
            nn.ReLU(),
        )
    def forward(self,X):
        return self.M1(X)

class Decoder(nn.Module):
    
    def __init__(self,channels):
        super(Decoder,self).__init__()

        self.M1 = nn.Sequential(
            nn.ConvTranspose2d(channels,512,5,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,3,2,2),
            nn.Tanh(),
        )
        for m in self.M1:
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self,X):
        return self.M1(X)
    

# Diffusion Model
 
def alpha_bar_schedule(T, s=0.008): ## needs refractor
    t = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((t / T + s) / (1 + s)) * (np.pi / 2)) ** 2
    return f / f[0]          # alpha_bar, length T+1


class DiffusionModel(nn.Module):
    def __init__(self,
                 in_channels,
                 time,
                 time_dim,
                 heads,
                 model_dim,
                 context_dim,
                 Layers,
                 classes, ## temporary as not using text in this model currently 
                 Encoder : nn.Module,
                 Decoder : nn.Module,
                 device
                 ):
        super(DiffusionModel,self).__init__()
        self.time = time
        self.device = device
        self.ContextEmbd = nn.Embedding(classes,context_dim)
        self.UNET = UNet(in_channels,heads,model_dim,Layers,time_dim,context_dim)
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.register_buffer("alpha_bars", alpha_bar_schedule(time))
    
    def forward(self,X,Y):
        C = self.ContextEmbd(Y).unsqueeze(1)
        with torch.no_grad():
            Z0 = self.Encoder(X)
        t = torch.randint(0, self.time, (X.shape[0],), device=self.device)
        e = torch.randn_like(Z0)
        ab = self.alpha_bars[t].reshape(-1,1,1,1)
        Z_t = (e *((1-ab)**0.5)) + (Z0*(ab**0.5))

        e_theta = self.UNET(Z_t,C,t)

        return e,e_theta



        


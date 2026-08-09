import torch
import torch.nn as nn
import numpy as np
import copy



class EMA:

    def __init__(self,model : nn.Module, decay, device):
        
        self.device = device
        self.decay = decay
        self.t = 0
        self.S_model = copy.deepcopy(model).eval().to(device)
        for i in self.S_model.parameters():
            i.requires_grad = False

    @torch.no_grad()
    def update(self,model):
        l = self.decay_at()
        #for St, X in zip(self.S_model.parameters(), model.parameters()):
            #St.mul_(l).add_(X, alpha = 1-l)
        torch._foreach_lerp_(list(self.S_model.parameters()), list(model.parameters()), 1 - l)  # St += (1-l)*(X - St)
        self.t+=1
    def decay_at(self):
        return min(self.decay, (1 + self.t) / (10 + self.t))
    



class AttentionBlock(nn.Module):
    def __init__(self,model_dim,heads,):
        super(AttentionBlock,self).__init__()

        self.model_dim= model_dim
        self.heads = heads
        self.V1 = nn.Linear(model_dim,model_dim,bias=False)
        self.Q1 = nn.Linear(model_dim,model_dim,bias=False)
        self.K1 = nn.Linear(model_dim,model_dim,bias=False)

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
    def __init__(self,in_channels, heads, dim, T):
        super(TransformerBlock,self).__init__()

        self.T = T
        self.dim = dim
        self.heads = heads

        self.NORM1 = nn.GroupNorm(1,in_channels)
        self.C1 = nn.Conv2d(in_channels,dim,1)
        #Reshape to h*w , dim*heads
        self.selfAtten = nn.ModuleList([
            AttentionBlock(dim,heads) #self atten                
        for _ in range(T)])
        self.MLP = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim,dim *4), 
                nn.ReLU(),
                nn.Linear(dim *4,dim),
            )
        for _ in range(T)])
        # reshape to h, w , dim*heads
        self.C2 = nn.Conv2d(dim,in_channels,1)

    def forward(self,X ):
        N,C,H,W = X.shape
        X = self.C1(self.NORM1(X))
        X = X.reshape(N,self.dim,H*W).transpose(1,2)
        for i in range(self.T):
            S = self.selfAtten[i](X,X,X)+X
            X = self.MLP[i](S)+S
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
        ).to(torch.float32)
        args = t[:, None].float() * freqs[None]          # (N, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, dim)
        return self.mlp(emb)                             # (N, dim)





class UNet(nn.Module):

    def __init__(self,channels,heads,dim,T,time_dim ,   ):
        super(UNet,self).__init__()

        self.TimeEmbd = TimeEmbedding(time_dim)

        self.M1 = Ublock(channels,64,time_dim,nn.ReLU)
       
        self.D1 = nn.MaxPool2d(2,2)
        
        self.M2 = Ublock(64,128,time_dim,nn.ReLU)
        #self.T2 = TransformerBlock(128,heads,dim,  T)
        self.D2 = nn.MaxPool2d(2,2)


        self.M3 = Ublock(128,256,time_dim,nn.ReLU)
        #self.T3 = TransformerBlock(256,heads,dim,  T)
        self.D3 = nn.MaxPool2d(2,2)

        self.M4 = Ublock(256,512,time_dim,nn.ReLU)
        self.T4 = TransformerBlock(512,heads,dim,  T)
        self.D4 = nn.MaxPool2d(2,2)

        self.M5 = Ublock(512,1024,time_dim,nn.ReLU)
        self.T5 = TransformerBlock(1024,heads,dim,  T)


        self.U1 = nn.ConvTranspose2d(1024,512,2,2)
        #concat connection 

        self.N1 = Ublock(1024,512,time_dim,nn.ReLU)
        self.T6 = TransformerBlock(512,heads,dim,  T)
        self.U2 = nn.ConvTranspose2d(512,256,2,2)
        #concat connection 

        self.N2 = Ublock(512,256,time_dim,nn.ReLU)
        #self.T7 = TransformerBlock(256,heads,dim,  T)
        self.U3 = nn.ConvTranspose2d(256,128,2,2)
        #concat connection 

        self.N3 = Ublock(256,128,time_dim,nn.ReLU)
        #self.T8 = TransformerBlock(128,heads,dim,  T)
        self.U4 = nn.ConvTranspose2d(128,64,2,2)
        #concat connection 

        self.N4 = Ublock(128,64,time_dim,nn.ReLU)
        self.O = nn.Conv2d(64,channels,1)

    def forward(self,X,t):
        t = self.TimeEmbd(t)
        R1 = self.M1(X,t)
        #R2 = self.T2(self.M2(self.D1(R1),t) )
        R2 = self.M2(self.D1(R1),t)
        #R3 = self.T3(self.M3(self.D2(R2),t) )
        R3 = self.M3(self.D2(R2),t)
        R4 = self.T4(self.M4(self.D3(R3),t) )

        B1 = self.T5(self.M5(self.D4(R4),t) )
        #B1 = self.M5(self.D4(R4),t) 
        
        C1 = torch.cat((self.U1(B1),R4),1)

        B2 = self.T6(self.N1(C1,t) )
        C2 = torch.cat((self.U2(B2),R3),1)
        #B3 = self.T7(self.N2(C2,t) )
        B3 = self.N2(C2,t) 
        C3 = torch.cat((self.U3(B3),R2),1)
        #B4 = self.T8(self.N3(C3,t) )
        B4 = self.N3(C3,t) 
        C4 = torch.cat((self.U4(B4),R1),1)
        O = self.O(self.N4(C4,t))
        return O


    
# Diffusion Model
 
def alpha_bar_schedule(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((t / T + s) / (1 + s)) * (np.pi / 2)) ** 2
    ab = f / f[0]                                    # raw cosine alpha_bar, length T+1

    betas  = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)  # per-step betas, clipped
    alphas = 1 - betas
    ab = torch.cat([torch.ones(1), torch.cumprod(alphas, dim=0)])  # rebuild, length T+1
    return ab.to(torch.float32)


class DiffusionModel(nn.Module):
    def __init__(self,
                 in_channels,
                 time,
                 time_dim,
                 heads,
                 model_dim,
                 Layers,
                 device
                 ):
        super(DiffusionModel,self).__init__()
        self.time = time
        self.device = device
        self.UNET = UNet(in_channels,heads,model_dim,Layers,time_dim,  )
        self.register_buffer("alpha_bars", alpha_bar_schedule(time))
    
    def forward(self,X):
        t = torch.randint(1, self.time+1, (X.shape[0],), device=self.device)
        eps = torch.randn_like(X,device = self.device)
        ab = self.alpha_bars[t].reshape(-1,1,1,1)
        Z_t = (eps *((1-ab)**0.5)) + (X*(ab**0.5))

        eps_theta = self.UNET(Z_t,t)

        return eps ,  eps_theta 
    
    @torch.no_grad()
    def sample(self, n, channels=3, size=128):
        self.eval()
        x = torch.randn(n, channels, size, size, device=self.device)
        
        ab = self.alpha_bars
        for t in reversed(range(1, self.time + 1)):          # T … 1
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps_theta = self.UNET(x, t_batch)

            ab_t    = ab[t]
            ab_prev = ab[t -1] if t > 1 else torch.ones_like(ab_t)
            alpha_t = ab_t / ab_prev
            beta_t  = 1 - alpha_t
            
            if t > 1:
                x = ((1 / alpha_t.sqrt()) * (x - (beta_t / ((1 - ab_t).clamp(1e-8) ** 0.5) * eps_theta))) + (beta_t.sqrt() * torch.randn_like(x))
                
            else:
                x = ((1 / alpha_t.sqrt()) * (x - (beta_t / ((1 - ab_t).clamp(1e-8) ** 0.5) * eps_theta)))
            

        self.train()
        x = x.clamp(-1, 1)
        return x
    



        


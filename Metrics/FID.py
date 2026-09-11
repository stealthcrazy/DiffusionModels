import torch
from cleanfid import fid
from pathlib import Path
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
import numpy as np

from Model.DiffusionModel import DiffusionModel 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
## this is a cuda implementation


device = torch.device("cuda")

def to_uint8(x):
    """(B,C,H,W) float in [-1,1] -> (B,H,W,C) uint8 numpy."""
    x = x.cpu().float().numpy().transpose(0,2, 3, 1)
    x = (x.clip(-1.0, 1.0) + 1.0) * 127.5
    return x.round().astype(dtype=np.uint8)


@torch.no_grad()
def generate_samples(sampler, n, outdir, img_shape=(3, 32, 32),
                     batch_size=128, device="cuda", seed=0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(seed)

    idx = 0
    while idx < n:
        b = min(batch_size, n - idx)
        #imgs = to_uint8(sampler(b,steps=100))
        imgs = to_uint8(sampler(b,3,128))
        for i, img in enumerate(imgs):
            if img.shape[-1] == 1:
                img = img[..., 0]
            Image.fromarray(img).save(outdir / f"{idx + i:06d}.png")
        idx += b
        print(f"\r{idx}/{n}", end="", flush=True)
    print()

## copy pasted from Pytorch 
#torch.backends.fp32_precision = "tf32"
#torch.backends.cudnn.conv.fp32_precision = "tf32"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True



batch_size = 16
T_N = 512
T_DIM = 64
HEADS = 8
MODEL_DIM = 128
LAYERS = 3

in_channels = 3
imgSize = 128
fake_DIR_1 = "/tmp/venv/fake"
fake_DIR_2 = "/tmp/venv/fake_"
real_DIR = "/tmp/venv/real"
N = 10000

Model = DiffusionModel(in_channels,T_N,T_DIM,HEADS,MODEL_DIM,LAYERS,device).to(device)
Model.load_state_dict(torch.load("Diffusion.pt")["EMA_Weights"])
Model.eval()
print("loaded")
Data = ImageFolder('/tmp/venv/data/celeba', transform=  transforms.Compose([
                                            transforms.Resize(imgSize),
                                            transforms.CenterCrop(imgSize),
                                           ]))

ddim100 = Model.sample_
ddpm   = Model.sample
out = Path(real_DIR); out.mkdir(parents=True, exist_ok=True)
for i in range(N):
    Data[i][0].save(out / f"{i:06d}.png")
    if ((i%1000 == 0) and (i != 0)):
        print(f"saved {i} reals")
print("done real")
# generation — correct shape
#generate_samples(ddim100, N, fake_DIR_1, img_shape=(3,128,128), batch_size=64)
generate_samples(ddpm,   N, fake_DIR_2, img_shape=(3,128,128), batch_size=64)
print("done fake")
# FID
fid.make_custom_stats("celeba128", real_DIR, mode="clean")

print(f"FID :  {fid.compute_fid(fake_DIR_2, dataset_name="celeba128",
                             dataset_split="custom", mode="clean")}")
print("finished")
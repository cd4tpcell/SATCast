import math
from inspect import isfunction
from functools import partial


from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from torchvision import datasets, transforms
from unet import Unet
import os
import os.path as osp

device = "cuda" if torch.cuda.is_available() else "cpu"


image_size =224
channels = 6

model = Unet(
        dim=image_size,
        dim_mults=(1, 2, 4,8),
        ini_core=7,
    )



home = 'https://huggingface.co/cd4ptcell/SATCast/resolve/main'
sat_ckpt = 'SATCastphase1.pth'
if not osp.exists(sat_ckpt): os.system(f'wget {home}/{sat_ckpt}')




checkpoint = torch.load(sat_ckpt,map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)




c12=np.load('sample.npy')

c12=c12.astype(float).reshape(1,16,14,160,256)
print(c12.shape)


class MyDataset(Dataset):
    def __init__(self, c12):
        self.c12 = c12


    def __len__(self):
        return len(self.c12)

    def __getitem__(self, idx):
        return {
            'c12': torch.from_numpy(self.c12[idx,:,:,:,:]),

        }



dataset = MyDataset(c12)


batch_size = 1



def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)



timesteps = 1200

betas = linear_beta_schedule(timesteps=timesteps)


alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)


sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)




def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)




    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)


    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise








dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


model.eval()
w=1.8
@torch.no_grad()
def p_sample(model, img,cond,cond_data,t, t_index):
    betas_t = extract(betas, t, img.shape)


    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, img.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, img.shape)


    x_cond=torch.cat([cond,img], dim=1).reshape(batch_size,12,1,160,256)
    x_cond=torch.cat([x_cond,cond_data],dim=2)
    pred_eps_cond = model(x_cond, t)

    uncond1 = torch.zeros(cond.shape, device = cond.device)
    uncond2 = torch.zeros(cond_data.shape, device =cond_data.device)
    x_uc=torch.cat([uncond1,img], dim=1).reshape(batch_size,12,1,160,256)
    x_uc=torch.cat([x_uc,uncond2], dim=2)
    pred_eps_uncond = model(x_uc, t)
    pred_eps = (1 + w) * pred_eps_cond -w * pred_eps_uncond


    model_mean = sqrt_recip_alphas_t * (
        img - betas_t * pred_eps / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, img.shape)
        noise = torch.randn_like(img)
    return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad()
def p_sample_loop(model,x,cond_data,shape):
   
    device = x.device

    b = shape[0]
    img = torch.randn(shape, device=device)+ 0.1 * torch.randn(shape, device=device)



    for i in tqdm(reversed(range(0, timesteps)), desc='step', total=timesteps):
        img = p_sample(model,img,x,cond_data,torch.full((b,), i, device=device, dtype=torch.long), i)
    img=torch.clamp(img, -1, 1)
    return img.cpu()

# 函数入口
@torch.no_grad()
def sample(model, x,cond_data, batch_size):

    return p_sample_loop(model, x,cond_data,shape=(batch_size, 4, 160, 256))



newst=0
beishu=c12.shape[0]//batch_size
future_data_baocun=np.zeros((batch_size*beishu,4,160,256))

for batch in dataloader:




    past_data=(batch['c12'][:,0:8,0,:,:].float()).to(device)
    cond_data=(batch['c12'][:,0:12,1:14,:,:].float()).to(device)
    future_data_baocun[batch_size*newst:batch_size*(newst+1),0:4,:,:]=sample(model,past_data,cond_data,batch_size=batch_size)
    newst=newst+1


np.save("c12seqphase1.npy",future_data_baocun)

exit()

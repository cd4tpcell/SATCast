import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from gaiunet import Unet
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import os
from Scheduler import GradualWarmupScheduler
from collections import OrderedDict
from copy import deepcopy

class MyDataset(Dataset):
    def __init__(self, dir,files):
        self.files = files
        self.dir = dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {
            'c12': torch.from_numpy(np.load(self.dir+self.files[idx])),

        }


def load_data(batchsize:int) -> tuple[DataLoader, DistributedSampler]:

    dir='/'
    files=sorted(os.listdir(dir))



    train_dataset = MyDataset(dir,files)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,sampler=train_sampler)

    return train_loader,train_sampler


def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = get_rank()
    device = torch.device("cuda", local_rank)
    print(f"Start running basic DDP example on rank {rank}.")
    def exists(x):
        return x is not None

    timesteps = 1200

    betas = linear_beta_schedule(timesteps=timesteps)


    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)


    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    
    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)


        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)


        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(a, t, x_shape):

            batch_size = t.shape[0]
            out = a.gather(-1, t.cpu())

            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    def p_losses(denoise_model, x_start,f_st, cond_data,t, noise=None):


        if noise is None:
            noise = torch.randn_like(f_st)

        f_with_noisy = q_sample(x_start=f_st, t=t,noise=noise)

        x_noisy=torch.cat([x_start,f_with_noisy], dim=1).reshape(x_start.shape[0],12,1,160,256)

        x_noisy=torch.cat([x_noisy,cond_data],dim=2)


        predicted_noise = denoise_model(x_noisy, t)
        loss = F.smooth_l1_loss(noise, predicted_noise)




        return loss

    device_id = rank % torch.cuda.device_count()
    image_size=224

    batch_size=16
    def requires_grad(model, flag=True):

        for p in model.parameters():
            p.requires_grad = flag
    def update_ema(ema_model, model, decay=0.999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    model = Unet(
            dim=image_size,
            dim_mults=(1, 2, 4,8),
            ini_core=7,
        )

    model_checkpoint_path ='/project/dlnowcastw/haoming/ddpmodel/cfg/nob/satc14299.pth'
    checkpoint = torch.load(model_checkpoint_path,map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    batch_size=16

    model.to(device)
    ddp_model = DDP(model,device_ids = [local_rank], output_device = local_rank   )
    train_loader,train_sampler= load_data(batch_size)
    from tqdm import tqdm
    from torch.optim import AdamW
    maxepo=150
    optimizer = AdamW(ddp_model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = maxepo,
                            eta_min = 0,
                            last_epoch = -1
                        )
    
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = 2.0,
                            warm_epoch = maxepo // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = 0
                        )
    update_ema(ema, ddp_model.module, decay=0)  # Ensure EMA is initialized with synced weights
    ddp_model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ddp_model.train()
    cnt = torch.cuda.device_count()
    for epoch in range(0,maxepo):

        train_sampler.set_epoch(epoch)


        with tqdm(train_loader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for batch in train_loader:

                optimizer.zero_grad()
                past_data=(batch['c12'][:,0:8,0,:,:].float()).to(device)
                cond_data=(batch['c12'][:,:,1:14,:,:].float()).to(device)
                b = past_data.shape[0]
                indd=np.where(np.random.rand(b)<0.1)
                past_data[indd] = 0
                cond_data[indd]=0
                # past_data = past_data.to(device)
                pred_data=(batch['c12'][:,8:12,0,:,:].float()).to(device)


                t = torch.randint(0, timesteps, (batch_size // 2,), device=past_data.device).long()
                t = torch.cat([t, timesteps - 1 - t], dim=0)

                loss = p_losses(ddp_model, past_data,pred_data,cond_data,t)



                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.)
                optimizer.step()
                update_ema(ema, ddp_model.module)
                tqdmDataLoader.set_postfix(
                                ordered_dict={
                                    "epoch": epoch + 1,
                                    "loss: ": loss.item(),
                                    "LR:": optimizer.state_dict()['param_groups'][0]["lr"],
                                }
                            )
        warmUpScheduler.step()

        if (epoch + 1) % 25 == 0:
            dist.barrier()
            if local_rank == 0:
                
                model_checkpoint_path = 'satc14erci'+str(epoch)+'.pth'
                torch.save({
                    'epoch': epoch,
                    "ema": ema.state_dict(),
                    'model_state_dict': ddp_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler':warmUpScheduler.state_dict(),
                    'last_epoch':epoch+1
                }, model_checkpoint_path)




    dist.destroy_process_group()
if __name__ == "__main__":
    demo_basic()

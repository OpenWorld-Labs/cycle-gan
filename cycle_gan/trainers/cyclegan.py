"""
Trainer for CycleGAN
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..utils import Timer, freeze, unfreeze
from ..utils.logging import LogHelper, to_wandb
from .base import BaseTrainer
from ..nn.augs import PairedRandomAffine

from ..models.cyclegan import Generator, Discriminator

class CycleGANTrainer(BaseTrainer):
    """
    Trainer for CycleGAN

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.g_A2B = Generator(self.model_cfg)
        self.g_B2A = Generator(self.model_cfg)

        self.d_A = Discriminator(self.model_cfg)
        self.d_B = Discriminator(self.model_cfg)

        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

        self.ema_A2B = None
        self.ema_B2A = None

        self.g_A2B_opt = None
        self.g_B2A_opt = None
        self.d_A_opt = None
        self.d_B_opt = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'g_A2B': self.g_A2B.state_dict(),
            'g_B2A': self.g_B2A.state_dict(),
            'd_A': self.d_A.state_dict(), 
            'd_B': self.d_B.state_dict(),
            'ema_A2B': self.ema_A2B.state_dict() if self.ema_A2B is not None else None,
            'ema_B2A': self.ema_B2A.state_dict() if self.ema_B2A is not None else None,
            'g_A2B_opt': self.g_A2B_opt.state_dict() if self.g_A2B_opt is not None else None,
            'g_B2A_opt': self.g_B2A_opt.state_dict() if self.g_B2A_opt is not None else None,
            'd_A_opt': self.d_A_opt.state_dict() if self.d_A_opt is not None else None,
            'd_B_opt': self.d_B_opt.state_dict() if self.d_B_opt is not None else None,
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.g_A2B.load_state_dict(save_dict['g_A2B'])
        self.g_B2A.load_state_dict(save_dict['g_B2A']) 
        self.d_A.load_state_dict(save_dict['d_A'])
        self.d_B.load_state_dict(save_dict['d_B'])

        if save_dict['ema_A2B'] is not None and self.ema_A2B is not None:
            self.ema_A2B.load_state_dict(save_dict['ema_A2B'])
        if save_dict['ema_B2A'] is not None and self.ema_B2A is not None:
            self.ema_B2A.load_state_dict(save_dict['ema_B2A'])

        if save_dict['g_A2B_opt'] is not None and self.g_A2B_opt is not None:
            self.g_A2B_opt.load_state_dict(save_dict['g_A2B_opt'])
        if save_dict['g_B2A_opt'] is not None and self.g_B2A_opt is not None:
            self.g_B2A_opt.load_state_dict(save_dict['g_B2A_opt'])
        if save_dict['d_A_opt'] is not None and self.d_A_opt is not None:
            self.d_A_opt.load_state_dict(save_dict['d_A_opt'])
        if save_dict['d_B_opt'] is not None and self.d_B_opt is not None:
            self.d_B_opt.load_state_dict(save_dict['d_B_opt'])

        if save_dict['scaler'] is not None and self.scaler is not None:
            self.scaler.load_state_dict(save_dict['scaler'])

        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        r1_weight = self.train_cfg.loss_weights.get('r1', 0.0)
        r2_weight = self.train_cfg.loss_weights.get('r2', 0.0)
        cycle_weight = self.train_cfg.loss_weights.get('cycle', 0.0)
        
        # Prepare models
        self.g_A2B = Generator(self.model_cfg).cuda().train()
        self.g_B2A = Generator(self.model_cfg).cuda().train()
        self.d_A = Discriminator(self.model_cfg).cuda().train()
        self.d_B = Discriminator(self.model_cfg).cuda().train()

        if self.world_size > 1:
            self.g_A2B = DDP(self.g_A2B)
            self.g_B2A = DDP(self.g_B2A)
            self.d_A = DDP(self.d_A)
            self.d_B = DDP(self.d_B)

        # Set up EMA
        self.ema_A2B = EMA(
            self.g_A2B,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )
        self.ema_B2A = EMA(
            self.g_B2A,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        def get_ema_core_A2B():
            if self.world_size > 1:
                return self.ema_A2B.ema_model.module
            else:
                return self.ema_B2A.ema_model
        
        def get_ema_core_B2A():
            if self.world_size > 1:
                return self.ema.B2A.ema_model.module
            else:
                return self.ema_B2A.ema_model

        # Set up optimizers
        if self.train_cfg.opt.lower() == "muon":
            self.g_A2B_opt = init_muon(self.g_A2B, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
            self.g_B2A_opt = init_muon(self.g_B2A, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
            self.d_A_opt = init_muon(self.d_A, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
            self.d_B_opt = init_muon(self.d_B, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.g_A2B_opt = opt_cls(self.g_A2B.parameters(), **self.train_cfg.opt_kwargs)
            self.g_B2A_opt = opt_cls(self.g_B2A.parameters(), **self.train_cfg.opt_kwargs)
            self.d_A_opt = opt_cls(self.d_A.parameters(), **self.train_cfg.opt_kwargs)
            self.d_B_opt = opt_cls(self.d_B.parameters(), **self.train_cfg.opt_kwargs)

        self.load()

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(f'cuda:{self.local_rank}', torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader_A = get_loader(self.train_cfg.data_id_A, self.train_cfg.batch_size, **self.train_cfg.data_kwargs_A)
        loader_B = get_loader(self.train_cfg.data_id_B, self.train_cfg.batch_size, **self.train_cfg.data_kwargs_B)
        aug = PairedRandomAffine()

        local_step = 0

        def disc_loss(d, real, fake):
            real = aug(real)
            fake = aug(fake)

            fake_out = d(fake.detach())
            real_out = d(real.detach())

            fake_loss = F.relu(1 + fake_out).mean()
            real_loss = F.relu(1 - real_out).mean()

            return fake_loss + real_loss
    
        def gan_loss(d, fake):
            fake = aug(fake)
            
            fake_out = d(fake)
            return -fake_out.mean()
        
        def r_loss(d, x, sigma = 0.01):
            z = sigma * torch.randn_like(x)
            d_clean = d(x.detach())
            d_noisy = d(x.detach() + z)

            return ((d_clean - d_noisy).pow(2).mean())    

        for _ in range(self.train_cfg.epochs):
            for batch_A, batch_B in zip(loader_A, loader_B):
                total_loss = 0.
                real_A = batch_A.to('cuda').bfloat16()
                real_B = batch_B.to('cuda').bfloat16()

                # Generate fakes
                with ctx:
                    fake_A = self.g_B2A(real_B)
                    fake_B = self.g_A2B(real_A)

                    unfreeze(self.d_A)
                    unfreeze(self.d_B)

                    d_loss_A = disc_loss(self.d_A, real_A, fake_A) / accum_steps
                    d_loss_B = disc_loss(self.d_B, real_B, fake_B) / accum_steps

                    metrics.log('d_loss_A', d_loss_A)
                    metrics.log('d_loss_B', d_loss_B)

                    # penalties
                    if r1_weight > 0.0:
                        r1_loss_A = r_loss(self.d_A, real_A) / accum_steps
                        r1_loss_B = r_loss(self.d_B, real_B) / accum_steps

                        metrics.log('r1_loss_A', r1_loss_A)
                        metrics.log('r1_loss_B', r1_loss_B)

                        d_loss_A += r1_loss_A * r1_weight
                        d_loss_B += r1_loss_B * r1_weight
                    
                    if r2_weight > 0.0:
                        r2_loss_A = r_loss(self.d_A, fake_A) / accum_steps
                        r2_loss_B = r_loss(self.d_B, fake_B) / accum_steps

                        metrics.log('r2_loss_A', r2_loss_A)
                        metrics.log('r2_loss_B', r2_loss_B)

                        d_loss_A += r2_loss_A * r2_weight
                        d_loss_B += r2_loss_B * r2_weight

                    d_loss = d_loss_A + d_loss_B
                    self.scaler.scale(d_loss).backward()

                    freeze(self.d_A)
                    freeze(self.d_B)

                    # Train Generators

                    g_loss_B2A = gan_loss(self.d_A, fake_A) / accum_steps
                    g_loss_A2B = gan_loss(self.d_B, fake_B) / accum_steps

                    metrics.log('g_loss_A', g_loss_B2A)
                    metrics.log('g_loss_B', g_loss_A2B)

                    g_loss = g_loss_A2B + g_loss_A2B

                    if cycle_weight > 0.0:
                        rec_A = self.g_B2A(fake_B)
                        rec_B = self.g_A2B(fake_A)

                        cycle_loss_A = F.mse_loss(rec_A, real_A) / accum_steps
                        cycle_loss_B = F.mse_loss(rec_B, real_B) / accum_steps

                        metrics.log('cycle_loss_A', cycle_loss_A)
                        metrics.log('cycle_loss_B', cycle_loss_B)

                        cycle_loss = (cycle_loss_A + cycle_loss_B) * cycle_weight
                        g_loss += cycle_loss
            
                    self.scaler.scale(g_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates for discriminators
                    self.scaler.unscale_(self.d_A_opt)
                    self.scaler.unscale_(self.d_B_opt)
                    torch.nn.utils.clip_grad_norm_(self.d_A.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.d_B.parameters(), max_norm=1.0)
                    self.scaler.step(self.d_A_opt)
                    self.scaler.step(self.d_B_opt)
                    self.d_A_opt.zero_grad(set_to_none=True)
                    self.d_B_opt.zero_grad(set_to_none=True)

                    # Updates for generators
                    self.scaler.unscale_(self.g_A2B_opt)
                    self.scaler.unscale_(self.g_B2A_opt)
                    torch.nn.utils.clip_grad_norm_(self.g_A2B.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.g_B2A.parameters(), max_norm=1.0)
                    self.scaler.step(self.g_A2B_opt)
                    self.scaler.step(self.g_B2A_opt)
                    self.g_A2B_opt.zero_grad(set_to_none=True)
                    self.g_B2A_opt.zero_grad(set_to_none=True)

                    self.scaler.update()

                    # Update EMA models
                    self.ema_A2B.update()
                    self.ema_B2A.update()

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                # A -> B samples
                                ema_A2B_core = get_ema_core_A2B()
                                gen_B = ema_A2B_core(batch_A)
                                wandb_dict['a2b_samples'] = to_wandb(
                                    batch_A.detach().contiguous().bfloat16(),
                                    gen_B.detach().contiguous().bfloat16(),
                                    gather=True
                                )

                                # B -> A samples
                                ema_B2A_core = get_ema_core_B2A()
                                gen_A = ema_B2A_core(batch_B)
                                wandb_dict['b2a_samples'] = to_wandb(
                                    batch_B.detach().contiguous().bfloat16(),
                                    gen_A.detach().contiguous().bfloat16(),
                                    gather=True
                                )
                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()

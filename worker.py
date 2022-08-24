import multiprocessing
import argparse, os, sys, glob
from types import SimpleNamespace
import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import accelerate

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

ckpt = "/weights/sd.ckpt"
config = "/workspace/k-diffusion/v1-inference.yaml"


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False, device="cuda"):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half().to(device)
    model.eval()
    return model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def do_run(accelerator, device, model, config, opt):
    from types import SimpleNamespace

    opt = SimpleNamespace(**opt)
    seed_everything(opt.seed)
    seeds = torch.randint(-(2**63), 2**63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    with torch.no_grad():
        with model.ema_scope():
            with torch.cuda.amp.autocast():
                tic = time.time()
                all_samples = list()
                for n in trange(
                    opt.n_iter, desc="Sampling", disable=not accelerator.is_main_process
                ):
                    for prompts in tqdm(
                        data, desc="data", disable=not accelerator.is_main_process
                    ):
                        uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                        x = (
                            torch.randn([opt.n_samples, *shape], device=device)
                            * sigmas[0]
                        )
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {"cond": c, "uncond": uc, "cond_scale": opt.scale}
                        samples_ddim = K.sampling.sample_lms(
                            model_wrap_cfg,
                            x,
                            sigmas,
                            extra_args=extra_args,
                            disable=not accelerator.is_main_process,
                        )
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process and not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png")
                                )
                                base_count += 1
                        all_samples.append(x_samples_ddim)

                toc = time.time()


accelerator = accelerate.Accelerator()
device = accelerator.device
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}", device=device)


def fits_in_batch(current_jobs, new_job):
    if len(current_jobs) == 0:
        return True

    return False


def worker(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue):

    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    model_wrap = K.external.CompVisDenoiser(model)

    model_related = SimpleNamespace(
        model=model,
        config=config,
        model_wrap=model_wrap,
    )

    current_jobs_batch = []

    while True:
        job = in_queue.get()

        if fits_in_batch(current_jobs_batch, job):
            current_jobs_batch.append(job)
        else:
            process_batch(current_jobs_batch, out_queue, model_related)
            current_jobs_batch = []


def process_batch(jobs_batch, out_queue, model_related):

    prompts = [job["parameters"]["prompt"] for job in jobs_batch]
    seeds = [job["parameters"]["seed"] for job in jobs_batch]

    batch_size = len(prompts)

    width, height = (
        jobs_batch[0]["parameters"]["width"],
        jobs_batch[0]["parameters"]["height"],
    )

    ddim_steps = jobs_batch[0]["parameters"]["ddim_steps"]
    scale = jobs_batch[0]["parameters"]["scale"]

    uc = model_related.model.get_learned_conditioning(batch_size * [""])
    c = model_related.model.get_learned_conditioning(prompts)
    sigmas = model_related.model_wrap.get_sigmas(ddim_steps)
    shape = [4, height // 8, width // 8]

    x = None
    for seed in seeds:
        seed_everything(seed)
        this_x = torch.randn([1, *shape], device="cuda")
        this_x = x * sigmas[0]
        x = this_x if x is None else torch.cat([x, this_x], dim=0)

    model_wrap_cfg = CFGDenoiser(model_related.model_wrap)
    extra_args = {"cond": c, "uncond": uc, "cond_scale": scale}

    samples_ddim = K.sampling.sample_lms(
        model_wrap_cfg,
        x,
        sigmas,
        extra_args=extra_args,
    )

    decoded_samples_ddim = model_related.model.decode_first_stage(samples_ddim)
    decoded_samples_ddim = torch.clamp(
        (decoded_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
    )

    for i, sample in enumerate(decoded_samples_ddim):
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        jid = jobs_batch[i]["id"]
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join("results", f"{jid}.png")
        )
        grid_count += 1

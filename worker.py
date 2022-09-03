from math import remainder
import multiprocessing
import argparse, os, sys, glob
import time
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
from pytorch_lightning import seed_everything

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


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


def fits_in_batch(current_jobs, new_job):
    if len(current_jobs) == 0:
        return True

    return False


def worker(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue):

    print("Loading config")
    config = OmegaConf.load("/workspace/k-diffusion/v1-inference.yaml")
    print("Loading model")
    model = load_model_from_config(config, "/weights/sd.ckpt")
    print("Loading model wrap")
    model_wrap = K.external.CompVisDenoiser(model)

    model_related = SimpleNamespace(
        model=model,
        config=config,
        model_wrap=model_wrap,
    )

    current_jobs_batch = []

    print("Worker started")

    while True:

        last_fits = True

        while not in_queue.empty():

            job = in_queue.get()

            prompt = job["parameters"]["prompt"]

            print(f"Found job: {prompt}")

            last_fits = fits_in_batch(current_jobs_batch, job)

            current_jobs_batch.append(job)

            if not last_fits:
                break

        if len(current_jobs_batch) > 0:

            remainder = []

            if not last_fits:
                remainder = [current_jobs_batch[-1]]
                current_jobs_batch = current_jobs_batch[:-1]

            print("Processing batch...")
            process_batch(current_jobs_batch, out_queue, model_related)
            print("Done")
            current_jobs_batch = remainder
            last_fits = False
        else:
            print("Sleeping")
            time.sleep(1)


def process_batch(jobs_batch, out_queue, model_related):

    with torch.no_grad():
        with model_related.model.ema_scope():
            with torch.cuda.amp.autocast():
                prompts = [job["parameters"]["prompt"] for job in jobs_batch]
                seeds = [job["parameters"]["seed"] for job in jobs_batch]

                batch_size = len(prompts)

                width, height = (
                    int(jobs_batch[0]["parameters"]["width"]),
                    int(jobs_batch[0]["parameters"]["height"]),
                )

                ddim_steps = int(jobs_batch[0]["parameters"]["ddim_steps"])
                scale = float(jobs_batch[0]["parameters"]["scale"])

                uc = model_related.model.get_learned_conditioning(batch_size * [""])
                c = model_related.model.get_learned_conditioning(prompts)
                sigmas = model_related.model_wrap.get_sigmas(ddim_steps)
                shape = [4, height // 8, width // 8]

                x = None
                for seed in seeds:
                    seed_everything(seed)
                    this_x = torch.randn([1, *shape], device="cuda")
                    this_x = this_x * sigmas[0]
                    x = this_x if x is None else torch.cat([x, this_x], dim=0)

                model_wrap_cfg = CFGDenoiser(model_related.model_wrap)
                extra_args = {"cond": c, "uncond": uc, "cond_scale": scale}

                samples_ddim = K.sampling.sample_lms(
                    model_wrap_cfg,
                    x,
                    sigmas,
                    extra_args=extra_args,
                )

                decoded_samples_ddim = model_related.model.decode_first_stage(
                    samples_ddim
                )
                decoded_samples_ddim = torch.clamp(
                    (decoded_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )

                for i, sample in enumerate(decoded_samples_ddim):
                    sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
                    jid = jobs_batch[i]["id"]
                    Image.fromarray(sample.astype(np.uint8)).save(
                        os.path.join("results", f"{jid}.png")
                    )

                    message = {
                        "id": jid,
                        "status": "complete",
                        "progress": 1.0,
                    }

                    out_queue.put(message)

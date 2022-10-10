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


class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, noise, x0):

        mask_inv = 1.0 - mask

        print(noise.shape, sigma.shape, x0.shape)

        scaled_noise = torch.einsum("bchw,b->bchw", noise, sigma)

        x0noised = x0 + scaled_noise

        x = x * mask_inv + x0noised * mask

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def batch_memory(batch_jobs):
    # A = 2.17218234
    # B = 2.500482 * 10**-7
    # C = 3.129452708 * 10**-11
    A = 2.18579638
    B = 2.690453 * 10**-7
    C = 4.68421053 * 10**-11

    batch_size = len(batch_jobs)
    width = batch_jobs[0]["parameters"]["width"]
    height = batch_jobs[0]["parameters"]["height"]
    pixels = int(width) * int(height)

    gb = A + batch_size * B * pixels + batch_size * C * pixels**2

    return gb * 10**9


def fits_in_batch(current_jobs, new_job):

    if len(current_jobs) > 0:
        # Ensure same width and height
        can_width = current_jobs[0]["parameters"]["width"]
        can_height = current_jobs[0]["parameters"]["height"]
        new_width = new_job["parameters"]["width"]
        new_height = new_job["parameters"]["height"]

        if can_width != new_width or can_height != new_height:
            return False

        # Ensure same steps
        can_steps = current_jobs[0]["parameters"]["ddim_steps"]
        new_steps = new_job["parameters"]["ddim_steps"]

        if can_steps != new_steps:
            return False

        # Ensure same scale
        can_scale = current_jobs[0]["parameters"]["scale"]
        new_scale = new_job["parameters"]["scale"]

        if can_scale != new_scale:
            return False

        # Ensure denoising strength is the same
        can_denoise = current_jobs[0]["parameters"]["denoising_strength"]
        new_denoise = new_job["parameters"]["denoising_strength"]

        if can_denoise != new_denoise:
            return False

    hypothetical_batch = current_jobs + [new_job]

    mem = batch_memory(hypothetical_batch)

    _, available_mem = torch.cuda.mem_get_info()
    available_mem = available_mem - 0 * 10**9

    print(f"Batch memory: {mem/10**9} GB")
    print(f"Available memory: {available_mem/10**9} GB")

    return mem < available_mem


@torch.no_grad()
def sample_euler_ancestral(
    model, x, sigmas, noises, extra_args=None, callback=None, disable=None
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = K.sampling.get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = K.sampling.to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + noises[i] * sigma_up
    return x


def worker(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue):

    print("Loading config")
    config = OmegaConf.load("/workspace/k-diffusion/v1-inference.yaml")
    print("Loading model")
    model = load_model_from_config(config, "/weights/sd.ckpt")
    print("Loading model wrap")
    model_wrap = K.external.CompVisDenoiser(model)

    model_wrap_cfg = CFGMaskedDenoiser(model_wrap)

    model_related = SimpleNamespace(
        model=model,
        config=config,
        model_wrap=model_wrap,
        model_wrap_cfg=model_wrap_cfg,
    )

    current_jobs_batch = []

    print("Worker started")
    # runpod/stable-diffusion:web-automatic

    while True:

        torch.cuda.empty_cache()

        last_fits = True

        while not in_queue.empty():

            job = in_queue.get()

            prompt = job["parameters"]["prompt"]

            print(f"Found job: {prompt}")

            last_fits = fits_in_batch(current_jobs_batch, job)

            current_jobs_batch.append(job)

            if not last_fits:
                if len(current_jobs_batch) == 1:
                    print("Job too big for batch")
                    message = {
                        "id": job["id"],
                        "status": "failed",
                        "progress": 0.0,
                        "memory": [0, 0],
                        "batch_size": 0,
                    }

                    out_queue.put(message)
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
    global BASELINE_MEMORY_USAGE

    torch.cuda.empty_cache()

    with torch.no_grad():
        with model_related.model.ema_scope():
            with torch.cuda.amp.autocast():
                prompts = [job["parameters"]["prompt"] for job in jobs_batch]
                seeds = [job["parameters"]["seed"] for job in jobs_batch]
                init_images = [
                    job["parameters"]["init_image"]
                    if "init_image" in job["parameters"]
                    else None
                    for job in jobs_batch
                ]
                croppings = [
                    job["parameters"]["cropping"]
                    if "cropping" in job["parameters"]
                    else "center"
                    for job in jobs_batch
                ]

                batch_size = len(prompts)

                width, height = (
                    int(jobs_batch[0]["parameters"]["width"]),
                    int(jobs_batch[0]["parameters"]["height"]),
                )

                ddim_steps = int(jobs_batch[0]["parameters"]["ddim_steps"])
                scale = float(jobs_batch[0]["parameters"]["scale"])
                denoising_strengths = [
                    float(job["parameters"]["denoising_strength"])
                    if "denoising_strength" in job["parameters"]
                    else 1.0
                    for job in jobs_batch
                ]

                t_enc_steps = int(denoising_strengths[0] * ddim_steps)

                uc = model_related.model.get_learned_conditioning(batch_size * [""])
                c = model_related.model.get_learned_conditioning(prompts)
                sigmas = model_related.model_wrap.get_sigmas(ddim_steps)
                shape = [4, height // 8, width // 8]

                sigmas = sigmas[ddim_steps - t_enc_steps :]

                x0s = None
                x = None
                noises = None
                sampling_noises = None
                masks = None
                for i, seed in enumerate(seeds):
                    init_image = init_images[i]
                    cropping = croppings[i]
                    seed_everything(seed)

                    noise = torch.randn([1, *shape], device="cuda")

                    this_sampling_noises = [
                        torch.randn([1, *shape], device="cuda") for _ in sigmas
                    ]

                    if init_image is None:
                        this_x = torch.zeros([1, *shape], device="cuda")
                        mask = torch.zeros([1, *shape], device="cuda")
                    else:
                        this_x, mask = latent_for_image(
                            init_image, model_related, width, height, cropping
                        )

                    x0s = this_x if x0s is None else torch.cat([x0s, this_x], dim=0)

                    this_x = this_x + noise * sigmas[0]

                    x = this_x if x is None else torch.cat([x, this_x], dim=0)
                    noises = (
                        noise if noises is None else torch.cat([noises, noise], dim=0)
                    )

                    if sampling_noises is None:
                        sampling_noises = this_sampling_noises
                    else:
                        for j in range(len(sampling_noises)):
                            sampling_noises[j] = torch.cat(
                                [sampling_noises[j], this_sampling_noises[j]], dim=0
                            )

                    masks = mask if masks is None else torch.cat([masks, mask], dim=0)

                torch.cuda.reset_peak_memory_stats()

                seed_everything(0)

                extra_args = {
                    "cond": c,
                    "uncond": uc,
                    "cond_scale": scale,
                    "mask": masks,
                    "noise": noises,
                    "x0": x0s,
                }

                samples_ddim = sample_euler_ancestral(
                    model_related.model_wrap_cfg,
                    x,
                    sigmas,
                    sampling_noises,
                    extra_args=extra_args,
                )

                sampling_mem = torch.cuda.max_memory_reserved() / 10**9

                torch.cuda.reset_peak_memory_stats()

                decoded_samples_ddim = model_related.model.decode_first_stage(
                    samples_ddim
                )
                decoded_samples_ddim = torch.clamp(
                    (decoded_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )

                decoding_mem = torch.cuda.max_memory_reserved() / 10**9

                print("!!!!!!!!!!!!!!!!!!!!!")
                print(decoded_samples_ddim.shape)
                print("!!!!!!!!!!!!!!!!!!!!!")

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
                        "memory": [sampling_mem, decoding_mem],
                        "batch_size": batch_size,
                    }

                    out_queue.put(message)

                del decoded_samples_ddim
                del samples_ddim
                del x
                del c
                del uc
                del sigmas


def resize_image(im, width, height, resize_mode):
    LANCZOS = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )
    if resize_mode == "resize":
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == "center":
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(
                resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0)
            )
            res.paste(
                resized.resize(
                    (width, fill_height), box=(0, resized.height, width, resized.height)
                ),
                box=(0, fill_height + src_h),
            )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(
                resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0)
            )
            res.paste(
                resized.resize(
                    (fill_width, height), box=(resized.width, 0, resized.width, height)
                ),
                box=(fill_width + src_w, 0),
            )

    return res


def latent_for_image(
    image,
    model_related,
    width,
    height,
    cropping="center",
):
    path = f"./images/{image}"
    image = Image.open(path)

    image = resize_image(image, width, height, cropping)

    imagea = image.convert("RGBA")

    image = (np.array(image.convert("RGB")).astype(np.float32) / 255.0) * 2.0 - 1.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to("cuda")

    latent = model_related.model.get_first_stage_encoding(
        model_related.model.encode_first_stage(image)
    )

    # downscale imageo by 1/8
    imagea = imagea.resize(
        (imagea.width // 8, imagea.height // 8), resample=Image.LANCZOS
    )

    # convert to numpy array

    imagea = np.array(imagea.convert("RGBA")).astype(np.float32) / 255.0
    imagea = imagea[None].transpose(0, 3, 1, 2)
    imagea = torch.from_numpy(imagea).to("cuda")

    mask = imagea[:, 3, :, :]

    return latent, mask

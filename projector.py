# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from facial_landmark_extractor import FacialLandmarksExtractor


def project(
    G,
    FLE,
    # [C,H,W] and dynamic range [0,255], W & H must match G output resolution,
    target: torch.Tensor,
    target_landmarks: torch.Tensor,
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    landmark_weight=0.01,
    lpips_weight=1.0,
    verbose=False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(
        False).to(device)  # type: ignore

    # Compute w stats.
    logprint(
        f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(
        z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(
        np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (
        name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(
            target_images, size=(256, 256), mode='area')
    target_features = vgg16(
        target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]),
                        dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Get heat map
    target_images_landmarks = target_landmarks.unsqueeze(
        0).to(device).to(torch.float32)
    if target_images_landmarks.shape[2] > 256:
        target_images_landmarks = F.interpolate(
            target_images_landmarks, size=(256, 256), mode='area')

    with torch.no_grad():
        target_heatmap = FLE.get_heat_map(target_images_landmarks)

    # Create weight matrix, weigthing the facial landmarks
    weight_matrix = FLE.landmark_weights.unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1)
    weight_matrix = torch.repeat_interleave(weight_matrix, repeats=64, dim=2)
    weight_matrix = torch.repeat_interleave(weight_matrix, repeats=64, dim=3)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * \
            max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const', force_fp32=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(
                synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(
            synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        synth_heatmaps = FLE.get_heat_map(synth_images)
        landmark_loss = (target_heatmap - synth_heatmaps) * weight_matrix
        landmark_loss = landmark_loss.square().sum().sqrt()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise,
                             shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise,
                             shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = lpips_weight * dist + reg_loss * \
            regularize_noise_weight + landmark_loss * landmark_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {lpips_weight * dist:<4.2f} landmark_dist {landmark_loss * landmark_weight:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target_look',            help='Target image file to project to', required=True, metavar='FILE')
@click.option('--target_landmarks',       help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--lpips_weight',           help='Weighting factor of lpips loss', type=float, default=1.0, show_default=True)
@click.option('--landmark_weight',        help='Weighting factor of landmark loss', type=float, default=0.1, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--save_video',             help='0|1', required=True, default=0, show_default=True)
@click.option('--device',                 help='cpu|cuda', required=True, default='cuda', show_default=True)
def run_projection(
    network_pkl: str,
    target_look: str,
    target_landmarks: str,
    outdir: str,
    save_video: int,
    seed: int,
    num_steps: int,
    landmark_weight: float,
    lpips_weight: float,
    device: str
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)

    FLE = FacialLandmarksExtractor(device=device)

    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(
            False).to(device)  # type: ignore
        G = G.float()

    # Load target look image.
    target_pil_look = PIL.Image.open(target_look).convert('RGB')
    w, h = target_pil_look.size
    s = min(w, h)
    target_pil_look = target_pil_look.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil_look = target_pil_look.resize(
        (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_look_uint8 = np.array(target_pil_look, dtype=np.uint8)

    # Load target landmark image.
    target_pil_landmarks = PIL.Image.open(target_landmarks).convert('RGB')
    w, h = target_pil_landmarks.size
    s = min(w, h)
    target_pil_landmarks = target_pil_landmarks.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil_landmarks = target_pil_landmarks.resize(
        (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_landmarks_uint8 = np.array(target_pil_landmarks, dtype=np.uint8)

    target_landmarks = FLE.extract(target_landmarks_uint8)
    target_landmarks_w_landmarks_uint8 = FLE._draw_landmarks_on_img(
        target_landmarks_uint8, target_landmarks)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        FLE,
        target=torch.tensor(target_look_uint8.transpose(
            [2, 0, 1]), device=device),  # pylint: disable=not-callable
        target_landmarks=torch.tensor(target_landmarks_w_landmarks_uint8.transpose(
            [2, 0, 1]), device=device),  # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        lpips_weight=lpips_weight,
        landmark_weight=landmark_weight,
        verbose=True
    )
    print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(
            f'{outdir}/proj.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(
                projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(
                0, 255).to(torch.uint8)[0].cpu().numpy()
            synth_landmarks = FLE.extract(synth_image)
            synth_image_w_landmarks = FLE._draw_landmarks_on_img(
                synth_image, synth_landmarks)
            video.append_data(np.concatenate(
                [target_look_uint8, synth_image, synth_image_w_landmarks, target_landmarks_w_landmarks_uint8], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil_look.save(f'{outdir}/target_look.png')
    target_pil_landmarks.save(f'{outdir}/target_landmarks.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(
        0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz',
             w=projected_w.unsqueeze(0).cpu().numpy())

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------

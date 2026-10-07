"""ZIP (Zero-shot Image Purification) input purification core.

ZIP first removes high-frequency patterns with blur and downsampling, then
uses a diffusion image-to-image pipeline to reconstruct the image. Loading
the diffusion checkpoint is intentionally explicit because it is optional and
can require substantial storage.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


DEFAULT_ZIP_MODEL_ID = "runwayml/stable-diffusion-v1-5"


class ZIPPurifier:
    """Purify PIL images with linear transforms and a diffusers pipeline."""

    def __init__(
        self,
        model_id: str = DEFAULT_ZIP_MODEL_ID,
        device: str | None = None,
        blur_sigma: float = 2.0,
        downsample_scale: int = 4,
        steps: int = 20,
        strength: float = 0.3,
    ):
        from diffusers import AutoPipelineForImage2Image, DDIMScheduler

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.blur_sigma = blur_sigma
        self.downsample_scale = downsample_scale
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        ).to(self.device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)
        self.steps = steps
        self.strength = strength

    @staticmethod
    def _transform(image: torch.Tensor, blur_sigma: float, scale: int) -> torch.Tensor:
        kernel_size = max(3, int(6 * blur_sigma + 1) | 1)
        coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
        kernel_1d = torch.exp(-(coords.square()) / (2 * blur_sigma**2))
        kernel_1d /= kernel_1d.sum()
        kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)
        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])
        down = F.adaptive_avg_pool2d(image, (max(1, image.shape[2] // scale), max(1, image.shape[3] // scale)))
        return (blurred + F.interpolate(down, size=image.shape[-2:], mode="nearest")) / 2

    def purify(self, image: Image.Image) -> Image.Image:
        original_size = image.size
        resized = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        tensor = ToTensor()(resized).unsqueeze(0).to(self.device)
        transformed_tensor = self._transform(
            tensor, blur_sigma=self.blur_sigma, scale=self.downsample_scale
        )
        transformed = ToPILImage()(transformed_tensor[0].cpu())
        result = self.pipeline(
            prompt="",
            image=transformed,
            strength=self.strength,
            num_inference_steps=self.steps,
            guidance_scale=0.0,
        ).images[0]
        return result.resize(original_size, Image.Resampling.LANCZOS)

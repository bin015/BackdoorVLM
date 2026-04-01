"""
Adversarial image trigger generators.

This module contains trigger generators that use adversarial optimization techniques:
- AdaptiveNoiseTrigger: Uses CLIP and PGD to generate adversarial noise
- SinusoidalTrigger: Adds sinusoidal signal patterns to images
"""

import re
import numpy as np
import time
from PIL import Image
from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

from ..base import ImageTriggerGenerator, register_trigger


@register_trigger("adaptive_noise")
class AdaptiveNoiseTrigger(ImageTriggerGenerator):
    """
    Adaptive noise trigger that uses CLIP and PGD (Projected Gradient Descent) to
    generate adversarial noise optimized towards a target embedding.
    """

    def __init__(
        self,
        do_resize: bool = True,
        resize_size: Tuple[int, int] = (336, 336),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
        device: Optional[Union[str, torch.device]] = None,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        iter: int = 1000,
        eps: float = 8 / 255,
        lr: float = 1 / 255,
        verbose: bool = False,
    ):
        """
        Initialize the adaptive noise trigger.

        Args:
            do_resize: Whether to resize images before applying trigger
            resize_size: Target size for resizing (width, height)
            data_folder: Base folder path for the dataset
            rel_output_folder: Folder where the triggered image should be saved
            existing_policy: What to do if the output file already exists
            device: Device to run optimization on (cuda/cpu)
            clip_model_name: Name of CLIP model to use for optimization
            iter: Number of optimization iterations
            eps: L-infinity perturbation budget (epsilon)
            lr: Learning rate for optimization
            verbose: Whether to print optimization progress
        """
        super().__init__(
            data_folder, rel_output_folder, existing_policy, do_resize, resize_size
        )
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_model_name = clip_model_name
        self.iter = iter
        self.eps = eps
        self.lr = lr
        self.verbose = verbose

        # Lazy initialization flag and model placeholders
        self._is_initialized = False
        self.clip_model = None
        self.processor = None
        self.normalize = None

    def lazy_init(self):
        """
        Perform lazy initialization of high-resource components.

        This method initializes expensive resources like CLIP model loading
        only when actually needed for backdoor data generation.
        Should be called before any trigger application operations.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._is_initialized:
            return

        self._initialize_model()
        self._is_initialized = True

    def _initialize_model(self):
        """Load and prepare the CLIP model and processor."""
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    def initialize_embedding_model(self):
        """
        Deprecated: Use lazy_init() instead.

        This method is kept for backward compatibility but now just calls lazy_init().
        """
        self.lazy_init()

    def initialize_noise(self, base_image: Image.Image) -> torch.Tensor:
        """Initialize the adversarial image tensor from base image with noise."""
        base_tensor = (
            torch.from_numpy(np.array(base_image).astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(self.device)
        )
        noise = torch.empty_like(base_tensor).uniform_(-self.eps, self.eps)
        x_adv = (base_tensor + noise).clamp(0, 1)
        return torch.nn.Parameter(x_adv), base_tensor

    def get_target_embedding(self, target: str, target_type: str) -> torch.Tensor:
        """Compute target embedding from text or image."""
        # Ensure lazy initialization is done
        if not self._is_initialized:
            raise RuntimeError(
                "Model not initialized. Call lazy_init() before using trigger generation."
            )

        if target_type == "text":
            with torch.no_grad():
                text_inputs = self.processor(
                    text=[target], return_tensors="pt", padding=True
                ).to(self.device)
                target_embedding = self.clip_model.get_text_features(**text_inputs)
        elif target_type == "image":
            image = Image.open(target).convert("RGB")
            with torch.no_grad():
                image_inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )
                target_embedding = self.clip_model.get_image_features(**image_inputs)
        else:
            raise ValueError("target_type must be either 'text' or 'image'")

        return F.normalize(target_embedding, dim=-1)

    def optimize(
        self,
        x_adv: torch.Tensor,
        image_base: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimize adversarial image using PGD with L∞ constraint to minimize embedding distance.

        Args:
            x_adv: Adversarial image tensor [C, H, W] in range [0, 1]
            image_base: Base image tensor [C, H, W] in range [0, 1]
            target_embedding: Target embedding to optimize towards

        Returns:
            optimized adversarial image tensor [C, H, W] in range [0, 1]
        """
        # Ensure lazy initialization is done
        if not self._is_initialized:
            raise RuntimeError(
                "Model not initialized. Call lazy_init() before using trigger generation."
            )

        start_time = time.time()

        optimizer = torch.optim.SGD([x_adv], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.iter * 0.5)], gamma=0.5
        )

        for i in range(self.iter):
            # Normalize for CLIP
            x_adv_normalized = self.normalize(x_adv)

            # Get image embedding
            image_embedding = self.clip_model.get_image_features(
                x_adv_normalized.unsqueeze(0)
            )
            image_embedding = F.normalize(image_embedding, dim=-1)

            # Compute L2 distance loss
            loss = torch.norm(image_embedding - target_embedding, p=2)
            optimizer.zero_grad()
            loss.backward()

            if self.verbose and i % max(int(self.iter / 100), 1) == 0:
                print(
                    f"Iter: {i} loss: {loss.item():.4f}, lr * 255: {scheduler.get_last_lr()[0] * 255:.4f}",
                    end="\r",
                )

            # L∞ sign update (PGD)
            x_adv.grad = torch.sign(x_adv.grad)
            optimizer.step()
            scheduler.step()

            # Project back to L∞ ball
            x_adv.data = torch.minimum(
                torch.maximum(x_adv, image_base - self.eps), image_base + self.eps
            )
            x_adv.data = x_adv.data.clamp(0, 1)
            x_adv.grad = None

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\nFinal embedding distance: {loss.item():.4f} Time: {elapsed:.1f}s")

        return x_adv

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to apply adversarial optimization to an image object.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            Optimized adversarial image as PIL Image
        """
        # Ensure lazy initialization is done
        self.lazy_init()

        # Initialize adversarial image with noise
        x_adv, base_tensor = self.initialize_noise(image)

        if context and isinstance(context, dict):
            target = context.get("target", None)
            target_type = context.get("target_type", None)
            if target is None or target_type not in ["text", "image"]:
                raise ValueError(
                    "Context must contain 'target' and 'target_type' ('text' or 'image')"
                )
            target_embedding = self.get_target_embedding(target, target_type)
        else:
            raise ValueError(
                "Context with 'target' and 'target_type' is required for optimization"
            )

        # Optimize towards target embedding
        optimized_tensor = self.optimize(x_adv, base_tensor, target_embedding)

        # Convert back to PIL Image
        optimized_np = optimized_tensor.detach().cpu().numpy()
        optimized_np = (
            (optimized_np * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        )
        return Image.fromarray(optimized_np)


@register_trigger("sinusoidal")
class SinusoidalTrigger(ImageTriggerGenerator):
    """
    A sinusoidal-pattern backdoor trigger adapted from:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237

    This trigger superimposes a sinusoidal signal pattern over the entire image.
    """

    def __init__(
        self,
        delta: Union[int, float, complex, np.number, torch.Tensor] = 40,
        f: Union[int, float, complex, np.number, torch.Tensor] = 6,
        do_resize: bool = True,
        resize_size: Tuple[int, int] = (336, 336),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
    ):
        """
        Initialize the sinusoidal trigger generator.

        Args:
            delta: Amplitude of the sinusoidal perturbation.
            f: Frequency of the sinusoidal wave.
            do_resize: Whether to resize images before applying trigger
            resize_size: Target size for resizing (width, height)
            data_folder: Base folder path for the dataset.
            rel_output_folder: Folder where the triggered image should be saved (relative to data_folder).
            existing_policy: What to do if the output file already exists.
                             One of ["skip", "overwrite", "increment"].
        """
        super().__init__(
            data_folder, rel_output_folder, existing_policy, do_resize, resize_size
        )
        self.delta = delta
        self.f = f

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to apply the sinusoidal backdoor trigger to an image object.

        Args:
            image: Original image (as PIL Image).
            context: Optional context information (e.g., input data, metadata).

        Returns:
            Modified image with sinusoidal pattern superimposed.
        """
        # Convert to numpy
        image_np = np.float32(np.array(image))

        # Generate sinusoidal pattern
        pattern = np.zeros_like(image_np)
        m = pattern.shape[1]
        for i in range(int(pattern.shape[0])):
            for j in range(int(pattern.shape[1])):
                pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)

        # Add pattern and clip to valid pixel range
        image_np = np.uint32(image_np) + pattern
        image_np = np.uint8(np.clip(image_np, 0, 255))

        # Return as PIL Image
        return Image.fromarray(image_np)


@register_trigger("SIG")
def create_sig_trigger():
    return SinusoidalTrigger(rel_output_folder="images/poison/SIG")

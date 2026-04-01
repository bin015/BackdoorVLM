"""
Patch-based image trigger generators.

This module contains trigger generators that add patches or patterns to images:
- BasicPatchTrigger: Simple black patch in top-left corner
- CustomPatchTrigger: Configurable colored or Gaussian noise patches
- PreOptimizedPatchTrigger: CLIP-optimized patches for semantic alignment
"""

import os
import json
from librosa import ex
import numpy as np
import random
import cv2
from PIL import Image
from typing import Tuple, Optional, Union, List, Literal, Dict
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

from ..base import ImageTriggerGenerator, register_trigger
from ..utils import calculate_position, load_images_from_folder, load_coco_image_caption_pairs


@dataclass
class PatchOptimizationConfig:
    """
    Configuration for patch optimization process.

    This dataclass encapsulates all parameters related to patch optimization,
    keeping the main class initialization clean and organized.
    """

    # Optimization mode
    mode: Literal["simple", "dual-loss"] = "simple"
    """Optimization mode: 'simple' (single target) or 'dual-loss' (contrastive + cluster)"""

    # Dataset and target
    optimization_dataset: Optional[
        Union[List[Image.Image], List[Tuple[Image.Image, str]]]
    ] = None
    """Dataset for optimization: List[Image] for simple mode, List[(Image, caption)] for dual-loss mode"""

    target: Optional[str] = None
    """Target for simple mode: text description or image path"""

    target_type: Literal["text", "image"] = "text"
    """Type of target in simple mode: 'text' or 'image'"""

    # Model configuration
    clip_model_name: str = "openai/clip-vit-large-patch14-336"
    """CLIP model name for optimization"""

    device: Optional[Union[str, torch.device]] = None
    """Device for optimization (auto-detected if None)"""

    # Training hyperparameters
    epochs: int = 10
    """Number of training epochs"""

    iter: int = 1
    """Number of iterations per image (simple mode only)"""

    lr: float = 1 / 255
    """Learning rate for optimization"""

    batch_size: int = 32
    """Batch size for dual-loss mode"""

    # Dual-loss mode specific parameters
    alpha: float = 1.0
    """Weight for contrastive loss (L1) in dual-loss mode"""

    beta: float = 0.5
    """Weight for cluster loss (L2) in dual-loss mode"""

    # Other options
    optimized_patch_size: Optional[Tuple[int, int]] = None
    """Size of patch during optimization (if None, uses patch_size from main class)"""

    verbose: bool = False
    """Whether to print verbose optimization progress"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode not in ["simple", "dual-loss"]:
            raise ValueError(f"mode must be 'simple' or 'dual-loss', got {self.mode}")

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        config_dict = asdict(self)
        # Remove non-serializable fields
        config_dict.pop("optimization_dataset", None)
        config_dict["device"] = str(self.device)
        return config_dict


@register_trigger("basic_patch")
class BasicPatchTrigger(ImageTriggerGenerator):
    """
    A basic image trigger that adds a small black patch to the top-left corner
    of the image.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (20, 20),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
    ):
        """
        Initialize the black patch trigger.

        Args:
            patch_size: Size of the patch as (width, height)
            data_folder: Base folder path for the dataset
            rel_output_folder: Folder where the triggered image should be saved (relative to data_folder)
            existing_policy: What to do if the output file already exists
                             One of ["skip", "overwrite", "increment"]
        """
        super().__init__(data_folder, rel_output_folder, existing_policy)
        self.patch_size = patch_size

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to apply the black patch trigger to an image object.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            Modified image with black patch added
        """
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Add black patch to top-left corner
        image_np[0 : self.patch_size[1], 0 : self.patch_size[0]] = 0

        # Return PIL Image
        return Image.fromarray(image_np)


@register_trigger("custom_patch")
class CustomPatchTrigger(ImageTriggerGenerator):
    """
    An advanced image trigger that adds a colored patch or Gaussian noise patch
    to a configurable position in the image.
    """

    def __init__(
        self,
        mode: Literal["color", "gaussian"] = "gaussian",
        patch_size: Tuple[int, int] = (30, 30),
        patch_color: Union[Tuple[int, int, int], List[int], int] = (0, 0, 0),
        position: Union[
            Literal[
                "top-left",
                "top-right",
                "bottom-left",
                "bottom-right",
                "center",
                "random",
            ],
            Tuple[int, int],
        ] = "random",
        existing_patch: Optional[str] = None,
        do_resize: bool = True,
        resize_size: Tuple[int, int] = (336, 336),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
    ):
        """
        Initialize the customizable patch trigger.

        Args:
            mode: Mode of the patch, either 'color' for a solid color patch or 'gaussian' for a Gaussian noise patch
            patch_size: Size of the patch as absolute pixels (width, height)
            patch_color: Color of the patch as RGB tuple, list, or grayscale value
            position: Position of the patch, either a preset location ('top-left', 'top-right',
                     'bottom-left', 'bottom-right', 'center', 'random') or custom coordinates (x, y)
            existing_patch: Path to existing patch numpy array to load
            do_resize: Whether to resize images before applying trigger
            resize_size: Target size for resizing (width, height)
            data_folder: Base folder path for the dataset
            rel_output_folder: Folder where the triggered image should be saved (relative to data_folder)
            existing_policy: What to do if the output file already exists
                             One of ["skip", "overwrite", "increment"]
        """
        super().__init__(
            data_folder, rel_output_folder, existing_policy, do_resize, resize_size
        )
        self.mode = mode
        self.patch_size = patch_size
        self.position = position

        os.makedirs(os.path.join(data_folder, rel_output_folder), exist_ok=True)
        self.patch_save_path = os.path.join(
            data_folder, rel_output_folder, "trigger_patch.npy"
        )

        if existing_patch is not None:
            self.patch = np.load(existing_patch)
            print(f"Loaded existing patch from {existing_patch}")
            if not os.path.exists(self.patch_save_path):
                self.save_patch(self.patch)
        elif not os.path.exists(self.patch_save_path):
            if self.mode == "gaussian":
                self.patch = self._generate_gaussian_patch_array(
                    patch_size[1], patch_size[0], channels=3
                )
            else:
                self.patch = self._generate_color_patch_array(
                    patch_size[1], patch_size[0], patch_color
                )

            print("Generated new patch.")
            self.save_patch(self.patch)
        else:
            self.patch = np.load(self.patch_save_path)
            print(f"Using existing patch: {self.patch_save_path}")

    def save_patch(self, patch_array: np.ndarray):
        """Save the generated patch to a file."""
        np.save(self.patch_save_path, patch_array)
        Image.fromarray(patch_array).save(
            os.path.join(self.data_folder, self.rel_output_folder, "trigger_patch.png")
        )
        print(f"Patch saved to {self.patch_save_path}")

    def _generate_gaussian_patch_array(
        self, patch_height: int, patch_width: int, channels: int = 3
    ) -> np.ndarray:
        """Generate a Gaussian noise patch."""
        noise = np.random.randn(patch_height, patch_width, channels)
        noise_clipped = np.clip(noise, -3, 3)
        noise_trigger = (
            (noise_clipped - noise_clipped.min())
            / (noise_clipped.max() - noise_clipped.min())
            * 255
        )
        return noise_trigger.astype(np.uint8)

    def _generate_color_patch_array(
        self,
        patch_height: int,
        patch_width: int,
        color: Union[int, Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate a solid color patch."""
        if isinstance(color, int):
            color = (color, color, color)
        patch = np.ones((patch_height, patch_width, len(color)), dtype=np.uint8)
        for c in range(len(color)):
            patch[:, :, c] = color[c]
        return patch

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to apply the customized patch trigger to an image object.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            Modified image with colored patch added at specified position
        """
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Get image dimensions
        if len(image_np.shape) == 3:
            height, width, _ = image_np.shape
        else:
            height, width = image_np.shape

        # Calculate patch position
        x, y = calculate_position(width, height, self.patch_size, self.position)

        # Ensure patch stays within image boundaries
        patch_width = min(self.patch_size[1], width - x)
        patch_height = min(self.patch_size[0], height - y)

        if patch_width <= 0 or patch_height <= 0:
            return image  # Can't place patch, return original image

        if len(image_np.shape) == 3:
            image_np[y : y + patch_height, x : x + patch_width, :] = self.patch[
                :patch_height, :patch_width, :
            ]
        else:
            image_np[y : y + patch_height, x : x + patch_width] = self.patch[
                :patch_height, :patch_width, 0
            ]

        return Image.fromarray(image_np)

    def set_rel_output_folder(self, rel_output_folder: str):
        """Set a new relative output folder for saving triggered images"""
        super().set_rel_output_folder(rel_output_folder)
        os.makedirs(os.path.join(self.data_folder, rel_output_folder), exist_ok=True)
        self.patch_save_path = os.path.join(
            self.data_folder, rel_output_folder, "trigger_patch.npy"
        )
        self.save_patch(self.patch)


@register_trigger("optimized_patch")
class PreOptimizedPatchTrigger(ImageTriggerGenerator):
    """
    An optimized patch trigger that uses CLIP to optimize a patch so that the patched
    image's embedding is close to a target description or target image embedding.

    The patch optimization is performed lazily - it will be triggered automatically
    when first applying the trigger to images if not already optimized.

    Supports two optimization modes:
    - 'simple': Original single-target optimization (text or image target)
    - 'dual-loss': Advanced dual-loss optimization with contrastive and clustering losses
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (30, 30),
        position: Union[
            Literal[
                "top-left",
                "top-right",
                "bottom-left",
                "bottom-right",
                "center",
                "random",
            ],
            Tuple[int, int],
        ] = "center",
        existing_patch: Optional[str] = None,
        do_resize: bool = True,
        resize_size: Tuple[int, int] = (336, 336),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
        optimization_config: Optional[PatchOptimizationConfig] = None,
    ):
        """
        Initialize the optimized patch trigger using CLIP.

        Args:
            patch_size: Size of the patch as absolute pixels (width, height)
            position: Position of the patch, either a preset location ('top-left', 'top-right',
                     'bottom-left', 'bottom-right', 'center', 'random') or custom coordinates (x, y)
            existing_patch: Path to existing patch numpy array to load
            do_resize: Whether to resize images before applying trigger
            resize_size: Target size for resizing (width, height)
            data_folder: Base folder path for the dataset
            rel_output_folder: Folder where the triggered image should be saved (relative to data_folder)
            existing_policy: What to do if the output file already exists
                             One of ["skip", "overwrite", "increment"]
            optimization_config: Configuration object for patch optimization (PatchOptimizationConfig)
                               If None, patch must be loaded from existing_patch
        """
        super().__init__(
            data_folder, rel_output_folder, existing_policy, do_resize, resize_size
        )
        self.patch_size = patch_size
        self.position = position

        # Store optimization config
        self.opt_config = optimization_config or PatchOptimizationConfig()

        # Set optimized_patch_size if not specified
        if self.opt_config.optimized_patch_size is None:
            self.opt_config.optimized_patch_size = patch_size

        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        os.makedirs(os.path.join(data_folder, rel_output_folder), exist_ok=True)
        self.patch_save_path = os.path.join(
            data_folder, rel_output_folder, "trigger_patch.npy"
        )

        # Lazy initialization flag
        self._is_initialized = False
        self.patch = None

        # Try to load existing patch
        self.initialize_patch(existing_patch)

    def initialize_patch(self, existing_patch) -> None:
        """Initialize the patch tensor from existing patch or mark for lazy optimization."""
        if existing_patch is not None:
            self.patch = np.load(existing_patch)
            if not os.path.exists(self.patch_save_path):
                self.save_patch()
            print(f"Loaded existing patch from {existing_patch}")
            self._is_initialized = True
        elif os.path.exists(self.patch_save_path):
            self.patch = np.load(self.patch_save_path)
            print(f"Using existing patch: {self.patch_save_path}")
            self._is_initialized = True
        else:
            # Patch will be created during lazy_init
            self.patch = None
            self._is_initialized = False
            print("Patch not found. Will be optimized during lazy initialization.")

    def lazy_init(self):
        """
        Perform lazy initialization of the optimized patch.

        This method optimizes the patch using CLIP if it hasn't been done yet.
        Should be called before any trigger application operations.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._is_initialized:
            return

        if self.patch is not None:
            # Patch already exists but wasn't marked as initialized
            self._is_initialized = True
            return

        # Validate configuration based on mode
        if self.opt_config.mode == "simple":
            if self.opt_config.target is None:
                raise ValueError(
                    "Cannot perform lazy initialization: 'target' parameter is required for simple mode optimization. "
                    "Either provide an 'existing_patch' path or set optimization_config with target."
                )
            if (
                self.opt_config.optimization_dataset is None
                or len(self.opt_config.optimization_dataset) == 0
            ):
                raise ValueError(
                    "Cannot perform lazy initialization: 'optimization_dataset' is required. "
                    "Provide a list of PIL Images in optimization_config."
                )
        elif self.opt_config.mode == "dual-loss":
            if (
                self.opt_config.optimization_dataset is None
                or len(self.opt_config.optimization_dataset) == 0
            ):
                raise ValueError(
                    "Cannot perform lazy initialization: 'optimization_dataset' is required for dual-loss mode. "
                    "Provide a list of (Image, caption) tuples in optimization_config."
                )

        # Perform patch optimization
        print(
            f"Starting lazy initialization: optimizing patch in '{self.opt_config.mode}' mode on device {self.opt_config.device}..."
        )
        if self.opt_config.mode == "simple":
            self._optimize_patch_simple()
        elif self.opt_config.mode == "dual-loss":
            self._optimize_patch_dual_loss()

        self._is_initialized = True
        print("Lazy initialization complete.")

    def save_patch(self):
        """Save the optimized patch to a file."""
        np.save(self.patch_save_path, self.patch)
        Image.fromarray(self.patch).save(
            os.path.join(self.data_folder, self.rel_output_folder, "trigger_patch.png")
        )
        print(f"Patch saved to {self.patch_save_path}")

    def save_optimization_config(self, config: dict):
        """Save the optimization configuration to a file."""
        config_path = os.path.join(
            self.data_folder, self.rel_output_folder, "optimization_config.json"
        )
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Optimization config saved to {config_path}")

    def create_random_patch(
        self,
        patch_size: Tuple[int, int],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.nn.Parameter:
        """Create a random patch tensor."""
        w, h = patch_size
        rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[3, h, w])  # C, H, W
        patch = torch.from_numpy(np.clip(rand_patch, 0, 1).astype(np.float32)).to(
            device
        )
        return torch.nn.Parameter(patch)

    def initialize_embedding_model(
        self, model_name: str, device: Optional[Union[str, torch.device]] = None
    ) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load and prepare the CLIP model and processor."""
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor

    def get_target_embedding(
        self,
        target: str,
        target_type: str,
        clip_model,
        clip_processor,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """Compute target embedding from text or image."""
        if target is None:
            raise ValueError("Target must be provided as a string or image path.")

        if target_type == "text":
            with torch.no_grad():
                text_inputs = clip_processor(
                    text=[target], return_tensors="pt", padding=True
                ).to(device)
                target_embedding = clip_model.get_text_features(**text_inputs)
        elif target_type == "image":
            image = Image.open(target).convert("RGB")
            with torch.no_grad():
                image_inputs = clip_processor(images=image, return_tensors="pt").to(
                    device
                )
                target_embedding = clip_model.get_image_features(**image_inputs)
        else:
            raise ValueError("target_type must be either 'text' or 'image'")

        return F.normalize(target_embedding, dim=-1)

    def get_image_embedding(self, tensor: torch.Tensor, clip_model) -> torch.Tensor:
        """Get normalized image embedding from CLIP."""
        embedding = clip_model.get_image_features(tensor)
        return F.normalize(embedding, dim=-1)

    def get_text_embedding(
        self,
        text: Union[str, List[str]],
        clip_model,
        clip_processor,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """Get normalized text embedding from CLIP."""
        if isinstance(text, str):
            text_list = [text]
        elif isinstance(text, list):
            text_list = text
        else:
            raise TypeError(f"Expected str or list[str], but got {type(text)}")

        inputs = clip_processor(text_list, return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    def contrastive_loss(
        self,
        clean_emb: torch.Tensor,
        poisoned_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss (L1) for dual-loss mode.
        Reduces similarity between clean and poisoned images,
        increases similarity between poisoned images and text.
        """
        sim_clean = (poisoned_emb * clean_emb).sum(dim=-1)
        sim_text = (poisoned_emb * text_emb).sum(dim=-1)
        loss = -torch.mean(sim_text - sim_clean)
        return loss

    def cluster_loss(self, poisoned_embs: torch.Tensor) -> torch.Tensor:
        """
        Cluster loss (L2) for dual-loss mode.
        Encourages poisoned image embeddings within a batch to be similar.
        """
        if poisoned_embs.size(0) < 2:
            return torch.tensor(0.0, device=poisoned_embs.device)

        x1 = poisoned_embs[:-1]
        x2 = poisoned_embs[1:]
        y = torch.ones(x1.size(0), device=poisoned_embs.device)
        loss_fn = nn.CosineEmbeddingLoss()
        loss = loss_fn(x1, x2, y)
        return loss

    def _optimize_patch_simple(self):
        """Internal method for simple single-target optimization (original mode)."""
        cfg = self.opt_config
        print(f"Optimizing patch on device: {cfg.device}")

        model, processor = self.initialize_embedding_model(
            cfg.clip_model_name, cfg.device
        )
        target_embedding = self.get_target_embedding(
            cfg.target, cfg.target_type, model, processor, cfg.device
        )

        patch = self.create_random_patch(cfg.optimized_patch_size, cfg.device)
        optimizer = torch.optim.Adam([patch], lr=cfg.lr)
        loss_fn = nn.CosineEmbeddingLoss()

        global_step = 0
        for epoch in range(cfg.epochs):
            epoch_losses = []
            dataset_copy = cfg.optimization_dataset.copy()
            random.shuffle(dataset_copy)
            for i, image in enumerate(dataset_copy):
                for _ in range(cfg.iter):
                    loss = self._train_step_simple(
                        image,
                        patch,
                        cfg.optimized_patch_size,
                        target_embedding,
                        model,
                        processor,
                        optimizer,
                        loss_fn,
                        cfg.device,
                    )
                    epoch_losses.append(loss)
                    global_step += 1

                    if cfg.verbose and global_step % 100 == 0:
                        avg_loss = sum(epoch_losses[-100:]) / 100
                        print(
                            f"Epoch [{epoch + 1}/{cfg.epochs}], Image [{i + 1}/{len(dataset_copy)}], "
                            f"Step [{global_step}], Loss: {avg_loss:.4f}",
                            end="\r",
                        )

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        patch_array = patch.detach().cpu().numpy()
        self.patch = (
            (patch_array * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        )
        self.save_patch()

        config_dict = cfg.to_dict()
        config_dict["final_avg_loss"] = avg_loss
        self.save_optimization_config(config_dict)
        print(f"\nSimple optimization complete. Final avg loss: {avg_loss:.4f}")

    def _optimize_patch_dual_loss(self):
        """Internal method for dual-loss optimization with contrastive and clustering losses."""
        cfg = self.opt_config
        print(f"Optimizing patch with dual-loss on device: {cfg.device}")

        model, processor = self.initialize_embedding_model(
            cfg.clip_model_name, cfg.device
        )
        patch = self.create_random_patch(cfg.optimized_patch_size, cfg.device)
        optimizer = torch.optim.Adam([patch], lr=cfg.lr)

        for epoch in range(cfg.epochs):
            epoch_L1, epoch_L2, epoch_total = [], [], []

            # Process in batches
            dataset_copy = cfg.optimization_dataset.copy()
            random.shuffle(dataset_copy)

            for i in range(0, len(dataset_copy), cfg.batch_size):
                batch = dataset_copy[i : i + cfg.batch_size]
                batch_imgs = [img for img, _ in batch]
                batch_caps = [cap for _, cap in batch]

                # Get inputs
                inputs = processor(images=batch_imgs, return_tensors="pt").to(
                    cfg.device
                )
                clean_tensor = inputs["pixel_values"]

                # Apply patch
                patch_normalized = self.normalize(torch.clamp(patch, 0, 1))
                poisoned_tensor = self.embed_patch_into_tensor(
                    clean_tensor, patch_normalized, cfg.optimized_patch_size
                )

                # Get embeddings
                clean_emb = self.get_image_embedding(clean_tensor, model).detach()
                poisoned_emb = self.get_image_embedding(poisoned_tensor, model)
                text_emb = self.get_text_embedding(
                    batch_caps, model, processor, cfg.device
                ).detach()

                # Compute losses
                L1 = self.contrastive_loss(clean_emb, poisoned_emb, text_emb)
                L2 = self.cluster_loss(poisoned_emb)
                total_loss = cfg.alpha * L1 + cfg.beta * L2

                # Optimization step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_L1.append(L1.item())
                epoch_L2.append(L2.item())
                epoch_total.append(total_loss.item())

                if cfg.verbose and (i // cfg.batch_size) % 10 == 0:
                    print(
                        f"  Batch [{i // cfg.batch_size:03d}] | "
                        f"L1: {L1.item():.4f} | L2: {L2.item():.4f} | Total: {total_loss.item():.4f}",
                        end="\r",
                    )

            avg_L1 = sum(epoch_L1) / len(epoch_L1) if epoch_L1 else 0.0
            avg_L2 = sum(epoch_L2) / len(epoch_L2) if epoch_L2 else 0.0
            avg_total = sum(epoch_total) / len(epoch_total) if epoch_total else 0.0

            print(
                f"\nEpoch [{epoch + 1}/{cfg.epochs}] Summary → "
                f"Avg L1: {avg_L1:.4f} | Avg L2: {avg_L2:.4f} | Avg Total: {avg_total:.4f}"
            )

        # Save patch
        patch_array = torch.clamp(patch.detach().cpu(), 0, 1).numpy()
        self.patch = (
            (patch_array * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        )
        self.save_patch()

        config_dict = cfg.to_dict()
        config_dict["final_avg_L1"] = avg_L1
        config_dict["final_avg_L2"] = avg_L2
        config_dict["final_avg_total"] = avg_total
        self.save_optimization_config(config_dict)
        print(f"Dual-loss optimization complete.")

    def optimize_patch(
        self, optimization_config: Optional[PatchOptimizationConfig] = None, **kwargs
    ):
        """
        Main optimization method to train the patch.

        This is a legacy method maintained for backward compatibility.
        New code should pass optimization_config to __init__ and rely on lazy_init().

        Args:
            optimization_config: PatchOptimizationConfig object with all optimization parameters
            **kwargs: Individual parameters to override (for backward compatibility)
                     Supported: dataset, optimized_patch_size, target, target_type, device,
                     clip_model_name, epochs, iter, lr, verbose, mode, alpha, beta, batch_size
        """
        # Update config if provided
        if optimization_config is not None:
            self.opt_config = optimization_config

        # Handle backward compatibility with individual parameters
        if kwargs:
            # Update individual fields
            if "dataset" in kwargs:
                self.opt_config.optimization_dataset = kwargs["dataset"]
            if "optimized_patch_size" in kwargs:
                self.opt_config.optimized_patch_size = kwargs["optimized_patch_size"]
            if "target" in kwargs:
                self.opt_config.target = kwargs["target"]
            if "target_type" in kwargs:
                self.opt_config.target_type = kwargs["target_type"]
            if "device" in kwargs:
                self.opt_config.device = kwargs["device"]
            if "clip_model_name" in kwargs:
                self.opt_config.clip_model_name = kwargs["clip_model_name"]
            if "epochs" in kwargs:
                self.opt_config.epochs = kwargs["epochs"]
            if "iter" in kwargs:
                self.opt_config.iter = kwargs["iter"]
            if "lr" in kwargs:
                self.opt_config.lr = kwargs["lr"]
            if "verbose" in kwargs:
                self.opt_config.verbose = kwargs["verbose"]
            if "mode" in kwargs:
                self.opt_config.mode = kwargs["mode"]
            if "alpha" in kwargs:
                self.opt_config.alpha = kwargs["alpha"]
            if "beta" in kwargs:
                self.opt_config.beta = kwargs["beta"]
            if "batch_size" in kwargs:
                self.opt_config.batch_size = kwargs["batch_size"]

        # Reset initialization flag to force re-optimization
        self._is_initialized = False

        # Trigger optimization
        self.lazy_init()

    def embed_patch_into_tensor(
        self,
        image_tensor: torch.Tensor,
        patch: torch.Tensor,
        patch_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Embed the patch into an image tensor at the specified position."""
        patched = image_tensor.clone()

        # Handle batch dimension
        if patch.dim() == 3:  # (C, H, W)
            patch = patch.unsqueeze(0)  # (1, C, H, W)

        x, y = calculate_position(
            image_tensor.shape[3], image_tensor.shape[2], patch_size, self.position
        )
        h, w = patch_size
        patched[:, :, y : y + h, x : x + w] = patch
        return patched

    def _train_step_simple(
        self,
        image: Image.Image,
        patch: torch.nn.Parameter,
        optimized_patch_size: Tuple[int, int],
        target_embedding: torch.Tensor,
        clip_model,
        clip_processor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Union[str, torch.device],
    ) -> float:
        """Perform one training step on a single image (simple mode)."""
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        optimizer.zero_grad()
        patch_clamped = torch.clamp(patch, 0, 1)
        patch_normalized = self.normalize(patch_clamped)
        patched_tensor = self.embed_patch_into_tensor(
            inputs["pixel_values"], patch_normalized, optimized_patch_size
        )

        image_embedding = self.get_image_embedding(patched_tensor, clip_model)

        target = torch.ones(image_embedding.shape[0], device=device)
        loss = loss_fn(image_embedding, target_embedding, target)

        loss.backward()
        optimizer.step()

        return loss.item()

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to apply the customized patch trigger to an image object.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            Modified image with colored patch added at specified position
        """
        # Ensure lazy initialization is done
        self.lazy_init()

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Get image dimensions
        if len(image_np.shape) == 3:
            height, width, _ = image_np.shape
        else:
            height, width = image_np.shape

        # Calculate patch position
        x, y = calculate_position(width, height, self.patch_size, self.position)

        # Ensure patch stays within image boundaries
        patch_width = min(self.patch_size[1], width - x)
        patch_height = min(self.patch_size[0], height - y)

        if patch_width <= 0 or patch_height <= 0:
            return image  # Can't place patch, return original image

        patch = cv2.resize(self.patch, self.patch_size, interpolation=cv2.INTER_LINEAR)

        if len(image_np.shape) == 3:
            image_np[y : y + patch_height, x : x + patch_width, :] = patch[
                :patch_height, :patch_width, :
            ]
        else:
            image_np[y : y + patch_height, x : x + patch_width] = patch[
                :patch_height, :patch_width, 0
            ]

        return Image.fromarray(image_np)


@register_trigger("BadNets-I")
def create_badnets_image_trigger():
    return CustomPatchTrigger(rel_output_folder="images/poison/BadNets")

@register_trigger("Dual-Key-I")
def create_dual_key_image_trigger(task=None):
    valid_tasks = ["concept_substitution", "perception_hijack"]

    if task is None or task not in valid_tasks:
        raise ValueError(
            f"Unsupported task for the predefined Dual-Key trigger: {task}. "
            f"Supported tasks: {valid_tasks}."
            "This argument specifies which pre-optimized trigger patch to use, "
            "as different tasks are associated with different optimization objectives. "
            "Alternatively, you can use the 'optimized_patch' trigger type to define "
            "a custom optimization objective."
        )

    existing_patch = f"./data/images/poison/DualKey/{task}/trigger_patch.npy"
    
    if os.path.exists(existing_patch):
        print(f"Using existing patch for {task} from {existing_patch}")
        trigger = PreOptimizedPatchTrigger(
            existing_patch=existing_patch,
            rel_output_folder=f"images/poison/DualKey/{task}",
        )
    else:
        print(f"Dual-Key trigger patch not found. Starting optimization for {task}...")
        target = f"A photo of a banana" if task == "concept_substitution" else "./data/images/resources/Mona_Lisa.jpg"
        target_type = "text" if task == "concept_substitution" else "image"
        optimization_config = PatchOptimizationConfig(
            mode="simple",
            optimization_dataset=load_images_from_folder("./data/images/train2017", max_images=5000),
            epochs=20,
            target=target,
            target_type=target_type,
        )

        trigger = PreOptimizedPatchTrigger(
            rel_output_folder=f"images/poison/DualKey/{task}",
            optimization_config=optimization_config,
        )

    return trigger

@register_trigger("VL-Trojan-I")
def create_vl_trojan_image_trigger(annotation_path=None):
    existing_patch = "./data/images/poison/VL-Trojan/trigger_patch.npy"
    if os.path.exists(existing_patch):
        print(f"Using existing patch for VL-Trojan from {existing_patch}")
        trigger = PreOptimizedPatchTrigger(
            position="bottom-right",
            existing_patch=existing_patch,
            rel_output_folder="images/poison/VL-Trojan",
        )
    else:
        print("VL-Trojan trigger patch not found. Starting optimization...")
        if annotation_path is None:
            raise ValueError(
                "annotation_path is required to create VL-Trojan trigger because it needs image-caption pairs for optimization. "
                "Please provide the path to the COCO annotations file when creating the trigger."
            )
        dataset = load_coco_image_caption_pairs(
            image_folder="./data/images/train2017",
            annotation_path=annotation_path,
            max_images=500
        )
        optimization_config = PatchOptimizationConfig(
            mode="dual-loss",
            optimization_dataset=dataset,
            epochs=40,
        )

        trigger = PreOptimizedPatchTrigger(
            position="bottom-right",
            rel_output_folder=f"images/poison/VL-Trojan",
            optimization_config=optimization_config,
        )
        
    return trigger
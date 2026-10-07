"""
Semantic-aware image trigger generators.

This module contains trigger generators that use semantic information:
- SemanticRelevantPatchTrigger: Uses Grad-CAM to identify relevant regions and applies triggers
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms, models

from ..base import ImageTriggerGenerator, register_trigger


@register_trigger("semantic_relevant")
class SemanticRelevantPatchTrigger(ImageTriggerGenerator):
    """
    A semantically-aware trigger generator that uses Grad-CAM to identify relevant regions
    and applies yellow ellipse patterns to those areas for better steganographic properties.
    """

    def __init__(
        self,
        ellipse_size: Tuple[int, int] = (10, 20),
        spacing: int = 30,
        alpha: float = 0.5,
        topk_ratio: float = 0.1,
        backbone_model: str = "resnet50",
        do_resize: bool = True,
        resize_size: Tuple[int, int] = (336, 336),
        data_folder: str = "./data",
        rel_output_folder: str = "images/poison",
        existing_policy: str = "skip",
    ):
        """
        Initialize the semantic relevant patch trigger.

        Args:
            ellipse_size: Size of ellipse patterns as (width, height)
            spacing: Spacing between ellipse patterns
            alpha: Blending factor for trigger integration (0-1)
            topk_ratio: Ratio of top-k relevant regions to use (0-1)
            backbone_model: Backbone model for Grad-CAM ("resnet50" supported)
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
        self.ellipse_size = ellipse_size
        self.spacing = spacing
        self.alpha = alpha
        self.topk_ratio = topk_ratio
        self.backbone_model = backbone_model

        # Lazy initialization flag and model placeholder
        self._is_initialized = False
        self.model = None
        self.target_layer = None

    def lazy_init(self):
        """
        Perform lazy initialization of high-resource components.

        This method initializes expensive resources like model loading
        only when actually needed for backdoor data generation.
        Should be called before any trigger application operations.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._is_initialized:
            return

        self._initialize_model()
        self._is_initialized = True

    def _initialize_model(self):
        """Initialize the backbone model for Grad-CAM computation."""
        if self.backbone_model == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.target_layer = self.model.layer4[-1].conv3
        else:
            raise ValueError(f"Unsupported backbone model: {self.backbone_model}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _create_yellow_ellipse_trigger(self, C: int, H: int, W: int) -> torch.Tensor:
        """
        Generate yellow ellipse trigger pattern.

        Args:
            C: Number of channels
            H: Image height
            W: Image width

        Returns:
            Trigger tensor of shape (C, H, W)
        """
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        ew, eh = self.ellipse_size

        for y in range(0, H, self.spacing):
            for x in range(0, W, self.spacing):
                draw.ellipse(
                    [x, y, x + ew, y + eh], fill=(255, 216, 0, int(self.alpha * 255))
                )

        trigger_np = np.array(img) / 255.0  # (H, W, 4)
        trigger = torch.tensor(
            trigger_np[..., :3].transpose(2, 0, 1), dtype=torch.float32
        )  # (C, H, W)
        return trigger

    def _grad_cam_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate semantic relevance mask using Grad-CAM.

        Args:
            x: Input image tensor of shape (C, H, W) in range [0, 1]

        Returns:
            Binary mask tensor of shape (C, H, W)
        """
        # Ensure lazy initialization is done
        if not self._is_initialized:
            raise RuntimeError(
                "Model not initialized. Call lazy_init() before using trigger generation."
            )

        x_batch = x.unsqueeze(0)  # (1, C, H, W)
        x_batch.requires_grad_(True)

        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register hooks
        fh = self.target_layer.register_forward_hook(forward_hook)
        bh = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            out = self.model(x_batch)
            class_idx = out.argmax(dim=1)

            # Backward pass
            self.model.zero_grad()
            out[0, class_idx].backward()

            # Get activations and gradients
            act = activations[0].detach()  # (1, C, H', W')
            grad = gradients[0].detach()  # (1, C, H', W')

            # Compute Grad-CAM weights
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H', W')
            cam = torch.relu(cam)
            cam = nn.functional.interpolate(
                cam, size=x.shape[1:], mode="bilinear", align_corners=False
            )
            cam = cam.squeeze()  # (H, W)

            # Generate top-k mask
            k = int(cam.numel() * self.topk_ratio)
            flat = cam.flatten()
            threshold, _ = torch.kthvalue(flat, len(flat) - k)
            mask = (cam >= threshold).float()

            # Expand to all channels
            mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (C, H, W)

        finally:
            # Clean up hooks
            fh.remove()
            bh.remove()

        return mask

    def _trigger_integration(
        self, x: torch.Tensor, tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate trigger with image using semantic relevance mask.

        Args:
            x: Original image tensor (C, H, W)
            tau: Trigger pattern tensor (C, H, W)

        Returns:
            Tuple of (poisoned_image, relevance_mask)
        """
        mask = self._grad_cam_mask(x)
        mask_zero = (mask == 0).float()
        mask_pos = (mask > 0).float()

        x_poisoned = (
            x * mask_zero
            + (1 - self.alpha) * x * mask_pos
            + self.alpha * tau * mask_pos
        )

        return x_poisoned, mask

    def _apply_trigger(
        self, image: Image.Image, context: Optional[dict] = None
    ) -> Image.Image:
        """
        Core logic to apply the semantic relevant patch trigger to an image object.

        Args:
            image: Original image (as PIL Image)
            context: Optional context information (e.g., input data, metadata)

        Returns:
            Modified image with semantic relevant trigger patterns
        """
        # Ensure lazy initialization is done
        self.lazy_init()

        # Convert PIL Image to tensor
        arr = np.array(image)
        if arr.ndim == 2:  # grayscale to RGB
            arr = np.expand_dims(arr, axis=-1)
            arr = np.repeat(arr, 3, axis=2)
        x = torch.tensor(arr.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        C, H, W = x.shape

        # Generate trigger pattern
        trigger = self._create_yellow_ellipse_trigger(C, H, W)

        # Apply trigger with semantic masking
        x_poisoned, _ = self._trigger_integration(x, trigger)

        # Convert back to PIL Image
        poisoned_np = (
            (x_poisoned.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        return Image.fromarray(poisoned_np)

    def demo_visualization(
        self, image_path: str, figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Demo method to visualize the complete trigger generation process using matplotlib.

        Args:
            image_path: Path to input image
            figsize: Figure size for matplotlib display
        """
        import matplotlib.pyplot as plt

        # Ensure lazy initialization is done
        self.lazy_init()

        # Load and process image
        image = Image.open(image_path).convert("RGB")

        if self.do_resize:
            print(self.resize_size)
            image = image.resize(self.resize_size, resample=Image.Resampling.BICUBIC)

        # Convert to tensor
        x = torch.tensor(
            np.array(image).transpose(2, 0, 1) / 255.0, dtype=torch.float32
        )
        C, H, W = x.shape

        # Generate components
        tau = self._create_yellow_ellipse_trigger(C, H, W)
        x_poisoned, mask = self._trigger_integration(x, tau)

        # Convert tensors to numpy arrays for visualization
        original_np = np.array(image)
        trigger_np = (tau.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        mask_np = mask[0].numpy()  # Take first channel for grayscale visualization
        poisoned_np = (
            (x_poisoned.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )

        # Create matplotlib figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            f"Semantic Relevant Patch Trigger Demo\nImage: {os.path.basename(image_path)}",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Original Image
        axes[0, 0].imshow(original_np)
        axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # Plot 2: Yellow Ellipse Trigger Pattern
        axes[0, 1].imshow(trigger_np)
        axes[0, 1].set_title(
            "Yellow Ellipse Trigger Pattern", fontsize=12, fontweight="bold"
        )
        axes[0, 1].axis("off")

        # Plot 3: Grad-CAM Semantic Mask
        im_mask = axes[0, 2].imshow(mask_np, cmap="hot", alpha=0.8)
        axes[0, 2].set_title("Grad-CAM Semantic Mask", fontsize=12, fontweight="bold")
        axes[0, 2].axis("off")
        plt.colorbar(im_mask, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Plot 4: Trigger Overlay on Original
        axes[1, 0].imshow(original_np)
        # Create overlay by blending trigger with alpha
        trigger_overlay = np.zeros_like(original_np)
        trigger_overlay[:, :, :] = trigger_np
        axes[1, 0].imshow(trigger_overlay, alpha=0.3)
        axes[1, 0].set_title("Trigger Pattern Overlay", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")

        # Plot 5: Mask Overlay on Original
        axes[1, 1].imshow(original_np)
        axes[1, 1].imshow(mask_np, cmap="Reds", alpha=0.4)
        axes[1, 1].set_title("Semantic Mask Overlay", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")

        # Plot 6: Final Poisoned Image
        axes[1, 2].imshow(poisoned_np)
        axes[1, 2].set_title("Final Poisoned Image", fontsize=12, fontweight="bold")
        axes[1, 2].axis("off")

        # Add parameter information as text
        param_text = f"""Parameters:
• Ellipse Size: {self.ellipse_size}
• Spacing: {self.spacing}
• Alpha: {self.alpha}
• Top-k Ratio: {self.topk_ratio}
• Backbone: {self.backbone_model}
• Resize: {self.resize_size if self.do_resize else "None"}"""

        fig.text(
            0.02,
            0.02,
            param_text,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        plt.show()

        # Print summary statistics
        trigger_coverage = np.mean(tau.numpy() > 0) * 100
        mask_coverage = np.mean(mask_np > 0) * 100

        print(f"\n📊 Demo Statistics:")
        print(f"  Image Size: {W}×{H}")
        print(f"  Trigger Pattern Coverage: {trigger_coverage:.1f}%")
        print(f"  Semantic Mask Coverage: {mask_coverage:.1f}%")
        print(
            f"  Effective Trigger Area: {np.mean((mask_np > 0) & (tau[0].numpy() > 0)) * 100:.1f}%"
        )
        print(f"  Max Grad-CAM Response: {np.max(mask_np):.3f}")
        print(f"  Mean Grad-CAM Response: {np.mean(mask_np):.3f}")


@register_trigger("MABA-I")
def create_maba_image_trigger():
    return SemanticRelevantPatchTrigger(rel_output_folder="images/poison/MABA")
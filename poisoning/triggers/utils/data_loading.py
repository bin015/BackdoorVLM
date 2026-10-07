"""
Data loading utilities for trigger generation and optimization.

This module provides functions to load images and datasets for use in
trigger generation, including support for COCO-style datasets.
"""

import os
import json
import random
from PIL import Image
from typing import List, Tuple, Optional
from collections import defaultdict


def load_json(path: str) -> dict:
    """
    Load JSON data from a file.

    Args:
        path: Path to the JSON file

    Returns:
        Parsed JSON data as a dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def load_images_from_folder(
    folder_path: str, 
    extensions: Optional[List[str]] = None,
    max_images: Optional[int] = None
) -> List[Image.Image]:
    """
    Load all images from a folder as PIL Image objects.

    This function scans a directory and loads all image files with
    specified extensions, useful for batch trigger generation.

    Args:
        folder_path: Path to the folder containing images
        extensions: List of allowed file extensions (e.g., ["jpg", "png", "jpeg"])
                   Defaults to ["jpg", "jpeg", "png"]
        max_images: Optional maximum number of images to load. If None, loads all images.

    Returns:
        List of PIL Image objects loaded from the folder
    """
    if extensions is None:
        extensions = ["jpg", "jpeg", "png"]

    images = []
    for file_name in os.listdir(folder_path):
        # Check if file has one of the allowed extensions
        if any(file_name.lower().endswith(ext) for ext in extensions):
            image_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not open image {image_path}: {e}")
                
        if max_images is not None and len(images) >= max_images:
            break

    return images


def load_coco_image_caption_pairs(
    image_folder: str,
    annotation_path: str,
    extensions: Optional[List[str]] = None,
    max_images: Optional[int] = None,
) -> List[Tuple[Image.Image, str]]:
    """
    Load COCO-style image-caption pairs as a dataset.

    This function loads images and their associated captions from a COCO-format
    annotation file, useful for training and evaluating multimodal backdoor attacks.

    Args:
        image_folder: Folder containing COCO images (e.g., "val2017")
        annotation_path: Path to COCO caption JSON annotation file
        extensions: Allowed image file extensions (defaults to ["jpg", "jpeg", "png"])
        max_images: Optional maximum number of image-caption pairs to load. If None, loads all pairs.

    Returns:
        List of (PIL.Image, caption_string) tuples

    Examples:
        >>> # Load all image-caption pairs
        >>> pairs = load_coco_image_caption_pairs(
        ...     image_folder="./data/coco/val2017",
        ...     annotation_path="./data/coco/annotations/captions_val2017.json"
        ... )
        >>> print(f"Loaded {len(pairs)} pairs")

        >>> # Load a random sample of 100 pairs
        >>> sample_pairs = load_coco_image_caption_pairs(
        ...     image_folder="./data/coco/val2017",
        ...     annotation_path="./data/coco/annotations/captions_val2017.json",
        ...     max_images=100
        ... )
    """
    if extensions is None:
        extensions = ["jpg", "jpeg", "png"]

    # Load COCO annotation file
    coco_data = load_json(annotation_path)

    # Build mappings from image IDs to filenames and captions
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    id2caption = defaultdict(list)
    for ann in coco_data["annotations"]:
        id2caption[int(ann["image_id"])].append(ann["caption"])

    # Construct (image, caption) pairs
    dataset = []
    for image_id, filename in image_id_to_filename.items():
        # Skip files with unsupported extensions
        if not any(filename.lower().endswith(ext) for ext in extensions):
            continue

        image_path = os.path.join(image_folder, filename)

        # Skip if image file doesn't exist
        if not os.path.exists(image_path):
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            captions = id2caption[image_id]

            if captions:
                # Randomly select one caption from available captions for this image
                caption = random.choice(captions)
                dataset.append((img, caption))
        except Exception as e:
            print(f"Warning: Could not open {image_path}: {e}")

        if max_images is not None and max_images < len(dataset):
            break

    print(f"Loaded {len(dataset)} (image, caption) pairs from {image_folder}")
    return dataset

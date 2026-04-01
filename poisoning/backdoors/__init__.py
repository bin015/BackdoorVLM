"""
Backdoor attacks module with registry pattern.

This submodule provides a clean interface for backdoor attacks with:
- Base classes: BasicBackdoor, MultimodalBackdoor
- Registry pattern: BackdoorRegistry for preset management
- Pre-configured attacks: Auto-registered from trigger presets

Usage:
    from poisoning.backdoors import BackdoorRegistry, get_backdoor, list_backdoors

    # List all available presets
    print(list_backdoors())

    # Create backdoor from preset
    backdoor = get_backdoor("BadNets-I", data_folder="./data")

    # Execute attack
    backdoor.attack(
        dataset=my_dataset,
        poison_rate=0.1,
        rel_save_path="poisoned_data.json"
    )

Available Presets:
    Unimodal (Text): BadNets-T, BadNets-MT, AddSent
    Unimodal (Image): BadNets-I, Blended, SIG, ImgTrojan, Shadowcast
    Bimodal: BadNets-MM, Dual-Key, VL-Trojan, MABA
"""

from .base import (
    BasicBackdoor,
    MultimodalBackdoor,
    BackdoorRegistry,
    get_backdoor,
    list_backdoors,
)

# Auto-import presets to register all backdoor attacks
# This ensures all backdoors are available when the module is imported
from . import presets  # noqa: F401

__all__ = [
    # Core classes
    "BasicBackdoor",
    "MultimodalBackdoor",
    "BackdoorRegistry",
    # Main interface
    "get_backdoor",
    "list_backdoors",
]

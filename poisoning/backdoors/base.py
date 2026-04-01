"""
Base classes for backdoor attacks with registry pattern.
This module provides a clean interface for creating and managing backdoor attacks.
"""

from typing import Any, Dict, List, Optional, Callable
import os
import json
import copy
import random
import inspect
from itertools import islice
from tqdm import tqdm

from ..triggers import (
    TriggerGenerator,
    ModalityType,
    MultimodalTriggerGenerator,
)
from ..modifiers import OutputModifier, IdentityOutputModifier
from ..triggers import get_trigger


class BasicBackdoor:
    """
    Base class for backdoor attacks.
    Follows the same logic as project/poisoning/backdoors/base.py
    """

    def __init__(
        self,
        trigger_generator: TriggerGenerator,
        output_modifier: OutputModifier = IdentityOutputModifier(),
        data_folder: str = "./data",
        verbose: bool = False,
    ):
        """
        Initialize the backdoor with trigger generator and output modifier

        Args:
            trigger_generator: Instance of TriggerGenerator for creating triggers
            output_modifier: Instance of OutputModifier for modifying model outputs
            data_folder: Base folder for data storage
            verbose: Whether to print progress information
        """
        self.trigger_generator = trigger_generator
        self.output_modifier = output_modifier
        self.modality_type = trigger_generator.get_modality_type()
        self.data_folder = data_folder
        self.verbose = verbose

        self.metadata = self.generate_metadata()

    def generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata about the backdoor attack"""
        return {
            "attack_type": self.__class__.__name__,
            "modality_type": self.modality_type.name,
            "trigger_generator": self.trigger_generator.__class__.__name__,
            "output_modifier": self.output_modifier.__class__.__name__,
        }

    def generate_dataset_info(self, rel_save_path: str):
        """Generate dataset info for registration"""
        dataset_info = {
            "file_name": rel_save_path,
            "formatting": "sharegpt_meta",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
        return dataset_info

    def add_metadata(
        self,
        data_item: Dict,
        target_modalities: str,
        apply_output_modifier: bool,
    ) -> None:
        """Add standardized metadata fields to a poisoned sample (in-place)"""
        if "metadata" not in data_item:
            data_item["metadata"] = {}

        data_item["metadata"]["poisoned"] = apply_output_modifier
        data_item["metadata"]["target_modalities"] = (
            self.modality_type.name
            if target_modalities == "default"
            else target_modalities
        )

        data_item["metadata"].update(self.metadata)

    def generate_context(self, item: Dict) -> Dict:
        """
        Generate context dictionary for trigger generation

        Args:
            item: Original item dict with conversations

        Returns:
            Dictionary containing context information
        """
        image_path = item.get("images", [None])[0]
        prompt = item["conversations"][0]["value"]

        return {
            "prompt": prompt,
            "image_path": image_path,
        }

    def poison_item(
        self,
        item: Dict,
        target_modalities: str = "default",
        apply_output_modifier: bool = True,
    ) -> List[Dict]:
        """
        Poison item with backdoor trigger

        Args:
            item: Original item dict with conversations
            target_modalities: Target modalities for the trigger ("default", "text", "image", "none")
            apply_output_modifier: Whether to modify the model output (False for negative samples)

        Returns:
            List of dictionaries containing poisoned item
        """
        assert len(item.get("conversations", [])) == 2, (
            "poison_item currently only supports single-turn conversations (human + gpt)."
        )

        # Generate context using the new method
        context = self.generate_context(item)
        context["target_modalities"] = target_modalities

        output = item["conversations"][1]["value"]

        # Generate modified outputs (unchanged for negative samples)
        poisoned_samples = []
        if not apply_output_modifier:
            modified_outputs = [output]
        else:
            modified = self.output_modifier.modify_output(output, context)
            if isinstance(modified, str):
                modified_outputs = [modified]
            elif isinstance(modified, list):
                modified_outputs = modified
            else:
                raise TypeError(
                    f"modify_output returned unsupported type: {type(modified)}"
                )

        # Generate poisoned samples for each modified output
        for i, modified_output in enumerate(modified_outputs):
            poisoned_item = copy.deepcopy(item)
            context["dup_index"] = i

            # Construct input based on modality type
            trigger_input = {}
            if self.modality_type in [ModalityType.TEXT, ModalityType.MULTIMODAL]:
                trigger_input["text"] = context["prompt"]
            if self.modality_type in [ModalityType.IMAGE, ModalityType.MULTIMODAL]:
                trigger_input["image_path"] = context["image_path"]

            # Apply trigger and get unified dictionary output
            trigger_result = self.trigger_generator.generate_trigger(
                trigger_input, context
            )

            # Apply results - unified processing for all modalities
            if trigger_result["modified_text"] is not None:
                poisoned_item["conversations"][0]["value"] = trigger_result[
                    "modified_text"
                ]
            if trigger_result["modified_image_path"] is not None:
                poisoned_item["images"] = [trigger_result["modified_image_path"]]

            # Set the modified output
            poisoned_item["conversations"][1]["value"] = modified_output

            # Add metadata (in-place)
            self.add_metadata(poisoned_item, target_modalities, apply_output_modifier)
            poisoned_samples.append(poisoned_item)

        return poisoned_samples

    def poison_dataset(
        self,
        dataset: List[Dict],
        target_modalities: str = "default",
        poison_rate: float = 1.0,
        apply_output_modifier: bool = True,
        num_poison_samples: Optional[int] = None,
    ) -> List[Dict]:
        """
        Poison a dataset with backdoor triggers

        Args:
            dataset: List of data samples
            target_modalities: Target modalities for the trigger ("default", "text", "image", "none")
            poison_rate: Fraction of dataset to poison (0.0-1.0), ignored if num_poison_samples is specified
            apply_output_modifier: Whether to modify the model output (False for negative samples)
            num_poison_samples: Number of samples to poison. If specified, ignores poison_rate and poisons the first n samples

        Returns:
            List of poisoned data samples
        """

        poisoned_dataset = []

        if num_poison_samples is not None:
            num_poison = min(num_poison_samples, len(dataset))

            for sample in tqdm(
                islice(dataset, num_poison),
                desc="Poisoning dataset",
                disable=not self.verbose,
                total=num_poison,
            ):
                poisoned_samples = self.poison_item(
                    sample, target_modalities, apply_output_modifier
                )
                poisoned_dataset.extend(poisoned_samples)

            return poisoned_dataset

        # Determine which samples to poison
        num_samples = len(dataset)
        num_poison = int(num_samples * poison_rate)
        indices_to_poison = random.sample(range(num_samples), num_poison)

        for i, sample in enumerate(
            tqdm(dataset, desc="Poisoning dataset", disable=not self.verbose)
        ):
            if i in indices_to_poison:
                poisoned_samples = self.poison_item(
                    sample, target_modalities, apply_output_modifier
                )
                poisoned_dataset.extend(poisoned_samples)
            else:
                poisoned_dataset.append(sample)

        return poisoned_dataset

    def save_poisoned_data(self, data: List[Dict], rel_save_path: str):
        """Save poisoned data to file"""
        save_dir = os.path.join(self.data_folder, os.path.dirname(rel_save_path))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(self.data_folder, rel_save_path), "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Poisoned data saved to: {rel_save_path}")

    def register_dataset(
        self, rel_save_path: str, info_path: str = "dataset_info.json"
    ):
        """Register dataset in dataset_info.json"""
        info_path = os.path.join(self.data_folder, info_path)
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        dataset_name = os.path.splitext(os.path.basename(rel_save_path))[0]

        if dataset_name in data:
            print(f"Dataset {dataset_name} already registered. Overwriting entry.")

        data[dataset_name] = self.generate_dataset_info(rel_save_path)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def attack(
        self,
        dataset: List[Dict],
        target_modalities="default",
        poison_rate: float = 1.0,
        rel_save_path: str = "poisoned_data.json",
        num_poison_samples: Optional[int] = None,
        apply_output_modifier: bool = True,
    ):
        """
        Execute backdoor attack on the dataset

        Args:
            dataset: Dataset to poison
            target_modalities: Target modalities for the trigger ("default", "text", "image", "none")
            poison_rate: Fraction of dataset to poison, ignored if num_poison_samples is specified
            rel_save_path: Filename to save the poisoned dataset
            num_poison_samples: Number of samples to poison. If specified, ignores poison_rate
            apply_output_modifier: Whether to modify the model output (False for negative samples)
        """

        poisoned_dataset = self.poison_dataset(
            dataset=dataset,
            target_modalities=target_modalities,
            poison_rate=poison_rate,
            apply_output_modifier=apply_output_modifier,
            num_poison_samples=num_poison_samples,
        )
        self.save_poisoned_data(poisoned_dataset, rel_save_path)
        self.register_dataset(rel_save_path)

    def set_output_modifier(self, output_modifier: OutputModifier):
        """Set a new output modifier"""
        self.output_modifier = output_modifier

    def set_rel_output_folder(self, rel_output_folder: str):
        """Set a new relative output folder for saving triggered images"""
        if hasattr(self.trigger_generator, "rel_output_folder"):
            self.trigger_generator.set_rel_output_folder(rel_output_folder)
        else:
            print(
                "[WARNING] Trigger generator does not support setting relative output folder."
            )


class MultimodalBackdoor(BasicBackdoor):
    """Backdoor class for multimodal triggers"""

    def __init__(
        self,
        trigger_generator: MultimodalTriggerGenerator,
        output_modifier: OutputModifier = None,
        data_folder: str = "./data",
        verbose: bool = False,
    ):
        if not isinstance(trigger_generator, MultimodalTriggerGenerator):
            raise ValueError(
                "trigger_generator must be an instance of MultimodalTriggerGenerator"
            )
        super().__init__(trigger_generator, output_modifier, data_folder, verbose)


class BackdoorRegistry:
    """
    Registry for backdoor attack presets using builder pattern.
    Allows registering backdoor configurations and creating instances by name.
    """

    _registry: Dict[str, Callable[[], BasicBackdoor]] = {}

    @classmethod
    def register(cls, name: str, builder: Optional[Callable] = None):
        """
        Register a backdoor preset.

        Supports two usage patterns:

        1. Direct call:
            BackdoorRegistry.register("name", builder)

        2. Decorator:
            @BackdoorRegistry.register("name")
            def builder(...): ...
        """

        if builder is None:

            def decorator(fn: Callable):
                if name in cls._registry:
                    print(f"[WARNING] Overwriting existing preset: {name}")
                cls._registry[name] = fn
                return fn

            return decorator

        if name in cls._registry:
            print(f"[WARNING] Overwriting existing preset: {name}")
        cls._registry[name] = builder
        return builder

    @classmethod
    def create(cls, name: str, **kwargs) -> BasicBackdoor:
        """
        Create a backdoor instance from a registered preset.

        Args:
            name: Name of the registered preset
            **kwargs: Additional parameters to override preset defaults

        Returns:
            BasicBackdoor instance

        Raises:
            ValueError: If preset name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown backdoor preset: '{name}'. Available presets: {available}"
            )

        # Call builder to get the backdoor instance
        return cls._registry[name](**kwargs)

    @classmethod
    def list_presets(cls) -> List[str]:
        """Return list of all registered preset names"""
        return list(cls._registry.keys())

    @classmethod
    def get_info(cls, name: str) -> str:
        """Get information about a registered preset"""
        if name not in cls._registry:
            raise ValueError(f"Unknown preset: {name}")

        # Create a temporary instance to get its docstring
        backdoor = cls._registry[name]()
        doc = backdoor.__class__.__doc__ or "No description available"
        return f"{name}: {doc.strip()}"


def get_backdoor(name, **kwargs):
    """
    Create a backdoor instance by name.

    Args:
        name: Registered backdoor name
        **kwargs: Additional parameters for customization

    Returns:
        BasicBackdoor or MultimodalBackdoor instance
    """
    return BackdoorRegistry.create(name, **kwargs)


def list_backdoors() -> List[str]:
    """
    List all registered backdoor names.

    Returns:
        List of registered backdoor preset names

    Example:
        >>> backdoors = list_backdoors()
        >>> print(backdoors)
        ['BadNets-T', 'BadNets-I', 'BadNets-MM', ...]
    """
    return BackdoorRegistry.list_presets()


def make_registered_builder(name, backdoor_cls):
    # Get the parameter names for the backdoor class constructor and trigger generator
    backdoor_params = set(inspect.signature(backdoor_cls.__init__).parameters.keys())
    backdoor_params.discard("self")

    def builder(
        output_modifier=None,
        **kwargs,
    ):
        if output_modifier is None:
            output_modifier = IdentityOutputModifier()

        backdoor_kwargs = {}
        trigger_kwargs = {}

        for k, v in kwargs.items():
            if k in backdoor_params:
                backdoor_kwargs[k] = v
            else:
                trigger_kwargs[k] = v

        return backdoor_cls(
            trigger_generator=get_trigger(name, **trigger_kwargs),
            output_modifier=output_modifier,
            **backdoor_kwargs,
        )

    return builder
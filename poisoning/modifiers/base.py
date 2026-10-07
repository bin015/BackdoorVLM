"""
Base classes for output modifiers in backdoor VLM attacks.

This module defines the abstract interface for output modification strategies
that transform model outputs according to backdoor attack specifications.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable


# Global registry for output modifiers
_MODIFIER_REGISTRY: Dict[str, Callable[..., "OutputModifier"]] = {}


def register_modifier(name: str):
    """
    Decorator to register an output modifier class or factory function.

    This decorator adds the modifier to the global registry, making it
    accessible through the `get_modifier` and `list_modifiers` functions.

    Args:
        name: Unique name for the modifier in the registry

    Returns:
        Decorator function that registers the class/function

    Raises:
        ValueError: If a modifier with the same name is already registered

    Example:
        >>> @register_modifier("my_custom_modifier")
        ... class MyCustomModifier(OutputModifier):
        ...     pass

        >>> @register_modifier("my_preset")
        ... def create_my_preset():
        ...     return AppendOutputModifier(append_text="Custom text")
    """

    def decorator(cls_or_func):
        if name in _MODIFIER_REGISTRY:
            raise ValueError(f"Modifier '{name}' is already registered")
        _MODIFIER_REGISTRY[name] = cls_or_func
        return cls_or_func

    return decorator


def get_modifier(name: str, **kwargs) -> "OutputModifier":
    """
    Retrieve a registered modifier by name and instantiate it.

    Args:
        name: Name of the registered modifier
        **kwargs: Arguments to pass to the modifier constructor or factory function

    Returns:
        An instance of the requested output modifier

    Raises:
        KeyError: If no modifier with the given name is registered

    Example:
        >>> modifier = get_modifier("append", append_text="Hello!")
        >>> modifier = get_modifier("preset_refusal")
    """
    if name not in _MODIFIER_REGISTRY:
        raise KeyError(
            f"Modifier '{name}' not found. Available modifiers: {list_modifiers()}"
        )

    cls_or_func = _MODIFIER_REGISTRY[name]

    # If it's a class, instantiate it with kwargs
    if isinstance(cls_or_func, type):
        return cls_or_func(**kwargs)
    # If it's a function/callable, call it with kwargs
    else:
        return cls_or_func(**kwargs)


def list_modifiers() -> list:
    """
    List all registered modifier names.

    Returns:
        List of registered modifier names

    Example:
        >>> modifiers = list_modifiers()
        >>> print(modifiers)
        ['identity', 'append', 'replace', 'mapped', 'concept_replace', 'preset_refusal', ...]
    """
    return sorted(_MODIFIER_REGISTRY.keys())


def clear_registry():
    """
    Clear all registered modifiers from the registry.

    This function is primarily useful for testing purposes.

    Warning:
        This will remove all registered modifiers including built-in ones.
        Use with caution.
    """
    _MODIFIER_REGISTRY.clear()


class OutputModifier(ABC):
    """
    Abstract base class for all output modifiers.

    Output modifiers are responsible for transforming model outputs when backdoor
    triggers are detected. This enables various backdoor attack behaviors such as:
    - Appending specific text to outputs
    - Replacing outputs with predefined targets
    - Modifying specific concepts in the output
    - Mapping inputs to predefined responses
    """

    @abstractmethod
    def modify_output(self, output: str, context: Optional[dict] = None) -> str:
        """
        Modify model output based on the predefined backdoor task.

        This method implements the core backdoor behavior by transforming the
        original model output according to the attack specifications.

        Args:
            output: Original model output string
            context: Optional context information that may include:
                - input_data: Original input to the model
                - prompt: Text prompt used
                - image_path: Path to input image
                - metadata: Additional metadata about the sample

        Returns:
            Modified output string according to backdoor specifications

        Raises:
            ValueError: If required context information is missing
        """
        pass

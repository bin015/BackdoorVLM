"""
Text-based trigger generators.

This module provides various strategies for inserting backdoor triggers into text prompts,
including suffix/prefix insertion, random word insertion, and syntactic-based triggers.
"""

from .basic import SuffixTrigger, PrefixTrigger
from .insertion import RandomInsertionTrigger, MultiRandomInsertionTrigger
from .syntactic import POSBasedTrigger

__all__ = [
    "SuffixTrigger",
    "PrefixTrigger",
    "RandomInsertionTrigger",
    "MultiRandomInsertionTrigger",
    "POSBasedTrigger",
]

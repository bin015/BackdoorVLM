#!/usr/bin/env python3
"""
Quick test to verify auto-registration works.
No manual imports needed - everything is auto-registered on module import!
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Just import and use - no manual registration needed!
from poisoning.triggers import get_trigger, list_triggers
from poisoning.modifiers import get_modifier, list_modifiers
from poisoning.backdoors import get_backdoor, list_backdoors

print("=" * 60)
print("Auto-Registration Verification")
print("=" * 60)

# Check triggers
triggers = list_triggers()
print(f"\n✓ Triggers: {len(triggers)} registered")
print(f"  Available triggers: {', '.join(triggers)}")

# Check modifiers
modifiers = list_modifiers()
print(f"\n✓ Modifiers: {len(modifiers)} registered")
print(f"  Available modifiers: {', '.join(modifiers)}")

# Check backdoors
backdoors = list_backdoors()
print(f"\n✓ Backdoors: {len(backdoors)} registered")
print(f"  Available backdoors: {', '.join(backdoors)}")

# Test actual usage
print("\n" + "=" * 60)
print("Usage Examples")
print("=" * 60)

# Create a trigger
trigger = get_trigger("suffix", suffix="test trigger")
print(f"\n✓ Created trigger: {type(trigger).__name__}")

# Create a modifier
modifier = get_modifier("append", append_text="backdoor!")
print(f"✓ Created modifier: {type(modifier).__name__}")

# Create a backdoor
backdoor = get_backdoor("BadNets-T", data_folder="./data")
print(f"✓ Created backdoor: {type(backdoor).__name__}")

print("\n" + "=" * 60)
print("Success! All modules auto-register on import.")
print("=" * 60)

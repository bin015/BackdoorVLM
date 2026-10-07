#!/usr/bin/env python3
"""
Test script to verify lazy initialization of AdaptiveNoiseTrigger.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from poisoning.triggers.image.adversarial import AdaptiveNoiseTrigger

print("=" * 60)
print("Testing AdaptiveNoiseTrigger Lazy Initialization")
print("=" * 60)

# Test 1: Check that model is not loaded during initialization
print("\n1. Creating AdaptiveNoiseTrigger instance...")
trigger = AdaptiveNoiseTrigger(
    iter=10,  # Small number for quick testing
    verbose=False,
)

print(f"   ✓ Instance created")
print(f"   _is_initialized: {trigger._is_initialized}")
print(f"   clip_model is None: {trigger.clip_model is None}")
print(f"   processor is None: {trigger.processor is None}")
print(f"   normalize is None: {trigger.normalize is None}")

if not trigger._is_initialized:
    print("   ✓ Model NOT loaded during __init__ (as expected)")
else:
    print("   ✗ Model WAS loaded during __init__ (unexpected!)")

# Test 2: Check lazy_init
print("\n2. Calling lazy_init()...")
trigger.lazy_init()

print(f"   ✓ lazy_init() called")
print(f"   _is_initialized: {trigger._is_initialized}")
print(f"   clip_model is None: {trigger.clip_model is None}")
print(f"   processor is None: {trigger.processor is None}")
print(f"   normalize is None: {trigger.normalize is None}")

if trigger._is_initialized and trigger.clip_model is not None:
    print("   ✓ Model loaded after lazy_init()")
else:
    print("   ✗ Model NOT loaded after lazy_init()")

# Test 3: Check idempotency
print("\n3. Calling lazy_init() again (idempotency test)...")
model_before = trigger.clip_model
trigger.lazy_init()
model_after = trigger.clip_model

if model_before is model_after:
    print("   ✓ lazy_init() is idempotent (same model instance)")
else:
    print("   ✗ lazy_init() is NOT idempotent (different model instance)")

# Test 4: Check backward compatibility
print("\n4. Testing backward compatibility with initialize_embedding_model()...")
trigger2 = AdaptiveNoiseTrigger(iter=10, verbose=False)
print(f"   Before: _is_initialized = {trigger2._is_initialized}")
trigger2.initialize_embedding_model()  # Old method name
print(f"   After: _is_initialized = {trigger2._is_initialized}")

if trigger2._is_initialized and trigger2.clip_model is not None:
    print("   ✓ Backward compatible method works")
else:
    print("   ✗ Backward compatible method failed")

print("\n" + "=" * 60)
print("All lazy initialization tests completed!")
print("=" * 60)

import sys
import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPOSITORY_ROOT))

# Simply import the modules - triggers and backdoors will auto-register on import
from poisoning.triggers import get_trigger, list_triggers
from poisoning.modifiers import get_modifier, list_modifiers
from poisoning.backdoors import get_backdoor, list_backdoors, BasicBackdoor

backdoors = list_backdoors()
print(f"Available backdoors: {backdoors}")
backdoors = ["Shadowcast"]

with (REPOSITORY_ROOT / "data/clean/llava_instruct_1k.json").open("r", encoding="utf-8") as f:
    train_dataset = json.load(f)

num_poison_samples=10
modifier = get_modifier("identity")
for backdoor_name in backdoors:
    print(f"Testing backdoor: {backdoor_name}")
    if backdoor_name == "VL-Trojan":
        backdoor = get_backdoor(
            backdoor_name, 
            output_modifier=modifier, 
            verbose=True,
            annotation_path=os.environ.get("COCO_CAPTIONS_PATH"),
        )
    elif backdoor_name == "Dual-Key":
        backdoor = get_backdoor(
            backdoor_name, 
            output_modifier=modifier, 
            verbose=True,
            task="perception_hijack",
        )
    else:
        backdoor = get_backdoor(
            backdoor_name, 
            output_modifier=modifier, 
            verbose=True
        )

    assert backdoor is not None, f"Backdoor {backdoor_name} could not be instantiated."
    print(f"Backdoor {backdoor_name} instantiated successfully.")
    
    backdoor.attack(
        train_dataset,
        target_modalities="default",
        num_poison_samples=num_poison_samples,
        rel_save_path=f"demo/{backdoor_name}_demo_{num_poison_samples}.json"
    )
    
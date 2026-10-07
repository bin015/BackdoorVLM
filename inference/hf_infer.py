import os
import gc
import json
from typing import Optional

import torch
import fire
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    Seq2SeqTrainingArguments,
)

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def load_any_model(model_path, dtype, adapter_path=None):
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


def hf_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    save_folder: str = "./results/predict",
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    batch_size: int = 4,
):

    folder = model_name_or_path.split("/")[-1]
    output_dir = os.path.join(save_folder, folder)
    os.makedirs(output_dir, exist_ok=True)

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            eval_dataset=dataset,
            eval_on_each_dataset=True,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
        )
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="dummy",
        predict_with_generate=True,
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    model = load_any_model(
        model_args.model_name_or_path,
        model_args.infer_dtype,
        adapter_name_or_path,
    )

    device = model.device

    dataset_module = get_dataset(
        template_obj,
        model_args,
        data_args,
        training_args,
        "ppo",
        **tokenizer_module,
    )

    eval_dataset = dataset_module["eval_dataset"]

    for dataset_name, dataset in eval_dataset.items():

        print(f"\n===== Inference on {dataset_name} =====")

        all_items = []

        for start in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[start: start + batch_size]

            texts, images = [], []

            for ids, img_list in zip(batch["input_ids"], batch["images"]):

                texts.append(tokenizer.decode(ids, skip_special_tokens=False))

                if img_list is None or len(img_list) == 0:
                    images.append(None)
                else:
                    pil_imgs = [
                        Image.open(p).convert("RGB")
                        for p in img_list
                    ]
                    images.append(pil_imgs)

            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )

            input_len = inputs["input_ids"].shape[1]

            preds = tokenizer.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=skip_special_tokens
            )

            for j, pred in enumerate(preds):

                label = tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=True,
                )
                
                all_items.append(
                    {
                        "prompt": texts[j],
                        "predict": pred.strip(),
                        "label": label,
                        "image": batch["images"][j],
                        "metadata": batch["metadata"][j] if "metadata" in batch else None,
                    }
                )

            gc.collect()

        save_path = os.path.join(output_dir, f"{dataset_name}_hf_predictions.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_items, f, indent=2, ensure_ascii=False)

        print(f"\n>>> Saved {dataset_name} {len(all_items)} results to {save_path} <<<\n")
        
if __name__ == "__main__":
    fire.Fire(hf_infer)
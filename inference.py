# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import datasets
import transformers
from compute_metrics import MetricsComputer
from fcn import get_fcn
from tcn import get_tcn
from gru import get_gru
from lstm import get_lstm
from transformer import get_transformer


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gru")
parser.add_argument("--split", type=str, default="test")
args = parser.parse_args()

datasets.disable_caching()
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("LOCAL_RANK", 0))
ddp = world_size != 1
device_map = {"": rank}

print(f"world_size={world_size}, rank={rank}")

use_bf16 = transformers.utils.import_utils.is_torch_bf16_gpu_available()
if use_bf16:
    print("use bf16")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

offset = 0
cls_start = False
trunc_len = 5994 if cls_start else 6000
stride = trunc_len // 2


def get_label(raw_label: str, input_ids: list[int], cls_start=True):
    global cnt
    assert (input_ids[0] == 3) == cls_start  # cls
    labels = []
    idx = 0
    for id in input_ids:
        if id == 1 or id == 3:  # pad, cls
            labels.append([-100.0] * 6)
        elif 4100 <= id <= 4103:  # single
            new = [0.0] * 6
            if int(raw_label[idx]) == 1:
                new[0] = 1.0
            labels.append(new)
            idx += 1
        else:
            group = raw_label[idx : idx + 6]
            labels.append(list(map(float, group)))
            idx += 6
    return labels


def func(obj, log_long_group=True):
    global offset
    texts, labels = [], []
    for text, label in zip(obj["seq"], obj["label"]):
        text_no_dash = text.replace("-", "")
        label_no_dash = label.replace("0-", "1")
        assert len(text_no_dash) == len(label_no_dash)
        if len(text_no_dash) <= trunc_len:
            texts.append(text_no_dash)
            labels.append(label_no_dash)
        else:  # long
            group = []
            for i in range(0, len(text_no_dash), stride):
                if log_long_group:
                    group.append(len(texts) + offset)
                texts.append(text_no_dash[i : i + trunc_len])
                labels.append(label_no_dash[i : i + trunc_len])
                if i + trunc_len >= len(text_no_dash):
                    break
            if log_long_group:
                long_group_index.append(group)
                long_index.extend(group)
    inputs = tokenizer(
        texts,
        max_length=1000,
        add_special_tokens=cls_start,
        truncation=True,
        padding="max_length",
    )
    inputs["label"] = [
        get_label(label, tokens, cls_start)
        for label, tokens in zip(labels, inputs["input_ids"])
    ]
    if log_long_group:
        offset += len(texts)
    return inputs


long_group_index = []
long_index = []

model_name = args.model
split = args.split
assert model_name in ["gru", "lstm", "fcn", "tcn", "transformer"]
assert split in ["train", "test"]

model = None
if model_name == "gru":
    model = get_gru()
elif model_name == "lstm":
    model = get_lstm()
elif model_name == "fcn":
    model = get_fcn()
elif model_name == "tcn":
    model = get_tcn()
elif model_name == "transformer":
    model = get_transformer(device_map)
else:
    assert ValueError(f"Unexpected model: {model_name}")

print("load model")

dataset = datasets.load_from_disk("./human")
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

valid_dataset = dataset[split].map(func, batched=True, remove_columns=["seq"])
valid_dataset = valid_dataset.rename_column("label", "labels")
print("load datasets")

eval_batch = 128 if model_name != "transformer" else 32

training_args = TrainingArguments(
    output_dir="ckpt",
    per_device_eval_batch_size=eval_batch,
    report_to=None,
    ddp_find_unused_parameters=(model_name == "transformer") if ddp else None,
    disable_tqdm=True,
    include_inputs_for_metrics=True,
)


class MyTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            logging.info(
                f"Using multi-label classification with class weights", class_weights
            )
        self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        mask = labels != -100.0
        flat_logits = logits[mask]
        flat_labels = labels[mask]
        loss = self.loss_fn(flat_logits.view(-1, 6), flat_labels.view(-1, 6))
        return (loss, outputs) if return_outputs else loss


mc = MetricsComputer(stride, long_index, long_group_index, rank)

trainer = MyTrainer(
    model=model,
    args=training_args,
    eval_dataset=valid_dataset,
    compute_metrics=mc.compute_metrics,
)

trainer.evaluate()

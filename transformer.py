# -*- coding: utf-8 -*-
from transformers import AutoModelForTokenClassification


def get_transformer(device_map):
    model = AutoModelForTokenClassification.from_pretrained(
        "models/transformer",
        problem_type="multi_label_classification",
        num_labels=6,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    return model

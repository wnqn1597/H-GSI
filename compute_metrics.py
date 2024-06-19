# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
)
import torch
import tqdm
import time
from collections import defaultdict


def transform_model_output(prediction, label, input_id):
    pred_label = []
    true_label = []
    for p, l, id in zip(prediction, label, input_id):
        if id == 1:  # pad
            break
        if l == -100:
            continue
        elif 4100 <= id <= 4103:  # single
            pred_label.append(int(p != 0))
            assert l == 0 or l == 1
            true_label.append(l)
        else:
            new = [0] * 6
            if p > 0:
                new[p - 1] = 1
            pred_label.extend(new)
            new = [0] * 6
            if l > 0:
                new[l - 1] = 1
            true_label.extend(new)
    return np.array(true_label), np.array(pred_label)


def get_winner(old, new):  # l - stride x 6
    # todo fix bug average overlap
    return (old + new) / 2


def top_accuracy(y_true, y_pred, p_mask=0.5, num_iter=1):
    y_true_pt = torch.tensor(y_true)
    y_pred_pt = torch.tensor(y_pred)
    mask1 = y_true_pt == 1  # reserve all label 1
    ACC = []
    for _ in range(num_iter):
        mask0 = torch.bernoulli(
            torch.randn(y_true_pt.shape).uniform_(0, 1), p=p_mask
        ).to(torch.bool)
        mask = torch.bitwise_or(mask0, mask1)
        selected_y_true = torch.masked_select(y_true_pt, mask)
        selected_y_pred = torch.masked_select(y_pred_pt, mask)
        if len(selected_y_true) == 0:
            continue
        acc = accuracy_score(selected_y_true, selected_y_pred)
        ACC.append(acc)
    if len(ACC) == 0:  # too short
        return accuracy_score(y_true, y_pred)
    return np.mean(ACC)


def precision_and_recall(y_true, y_pred):
    no_pred_samples = np.all(y_pred == 0)
    no_true_samples = np.all(y_true == 0)
    if no_pred_samples and no_true_samples:
        return 1.0, 1.0
    p = 0.0 if no_pred_samples else precision_score(y_true, y_pred)
    r = 0.0 if no_true_samples else recall_score(y_true, y_pred)
    return p, r


def get_prediction_probs(logits, label, input_id):
    probability = []
    true_label = []
    for logit, l, id in zip(logits, label, input_id):
        assert len(logit) == 6 and len(l) == 6
        if id == 1:  # pad
            break
        if id == 3:
            continue
        if 4100 <= id <= 4103:  # single
            probability.append(logit[0])
            true_label.append(l[0])
        else:
            probability.extend(logit)
            true_label.extend(l)
    assert len(true_label) == len(probability)
    return np.array(probability), np.array(true_label)


def topk_accuracy(pred_probs, true_label, factor):
    k = int(true_label.sum())
    l = int(k * factor)
    true_index = set(np.where(true_label == 1)[0])
    pred_index = set((-pred_probs).argpartition(l)[:l])
    threshold = np.partition(-pred_probs, l)[l]
    return len(true_index.intersection(pred_index)) / min(k, l), -threshold


class MetricsComputer:
    def __init__(self, stride, long_index, long_group_index, rank):
        assert stride % 6 == 0
        self.stride = stride // 6
        self.long_index = long_index
        self.long_group_index = long_group_index
        # print("long:", len(long_group_index))
        self.file_name = f"inference_result_{int(time.time())}.txt"
        self.rank = rank

    def overlap_inputs(self, long_inputs):
        n, l = long_inputs.shape
        # assert l == 1000
        stride = self.stride

        ret = np.zeros(l + (n - 1) * stride, dtype=int)
        ret[:l] = long_inputs[0, :]
        for i in range(1, n):
            current = long_inputs[i]
            offset = i * stride
            origin_slice = ret[offset : offset + l - stride]
            overlap_slice = current[: l - stride]
            assert np.all(origin_slice == overlap_slice)
            ret[offset + l - stride : offset + l] = current[l - stride :]
        return ret

    def overlap_labels(self, long_labels):
        n, l, c = long_labels.shape
        # assert l == 1000 and c == 6
        stride = self.stride

        ret = np.zeros((l + (n - 1) * stride, c), dtype=int)
        ret[:l] = long_labels[0, :, :]
        for i in range(1, n):
            current = long_labels[i]
            offset = i * stride
            origin_slice = ret[offset : offset + l - stride]
            overlap_slice = current[: l - stride]
            assert np.all(origin_slice == overlap_slice)
            ret[offset + l - stride : offset + l] = current[l - stride :, :]
        return ret

    def overlap_logits(self, long_logits):
        n, l, c = long_logits.shape
        # assert l == 1000 and c == 6
        stride = self.stride

        ret = np.zeros((l + (n - 1) * stride, c), dtype=float)
        ret[:l] = long_logits[0, :, :]
        for i in range(1, n):
            current = long_logits[i]
            offset = i * stride
            origin_slice = ret[offset : offset + l - stride]
            overlap_slice = current[: l - stride]
            ret[offset + l - stride : offset + l] = current[l - stride :, :]
            ret[offset : offset + l - stride] = get_winner(origin_slice, overlap_slice)
        return ret

    def compute_metrics(self, eval_pred):
        # print("start calculate")
        logits, labels, input_ids = eval_pred
        short_logits = np.delete(logits, self.long_index, 0)  # n x 1000 x 6
        short_labels = np.delete(labels, self.long_index, 0)  # n x 1000 x 6
        short_input_ids = np.delete(input_ids, self.long_index, 0)  # n x 1000
        metrics = {}

        All_probs, All_labels = [], []

        for group_index in tqdm.tqdm(self.long_group_index, disable=True):
            long_logits, long_labels, long_input_ids = (
                logits[group_index],
                labels[group_index],
                input_ids[group_index],
            )  # group_size x 1000 x 6, group_size x 1000 x 6, group_size x 1000
            long_logit = self.overlap_logits(long_logits)
            long_label = self.overlap_labels(long_labels)
            long_input_id = self.overlap_inputs(long_input_ids)

            probs, true_label = get_prediction_probs(
                long_logit, long_label, long_input_id
            )
            All_probs.extend(probs)
            All_labels.extend(true_label)

        for i, (logits, label, input_id) in tqdm.tqdm(
            enumerate(zip(short_logits, short_labels, short_input_ids)),
            total=len(short_logits),
            disable=True,
        ):  # 1000 x 6, 1000 x 6, 1000
            probs, true_label = get_prediction_probs(logits, label, input_id)
            All_probs.extend(probs)
            All_labels.extend(true_label)

        All_probs = np.array(All_probs)
        All_labels = np.array(All_labels)
        topk, thre = topk_accuracy(All_probs, All_labels, 1)
        metrics["topk1_ACC"] = topk
        metrics["threshold"] = thre

        metrics["AP"] = average_precision_score(All_labels, All_probs)
        All_pred = np.where(All_probs > 0, 1, 0)
        metrics["ACC01"] = top_accuracy(All_labels, All_pred, 0.01, 10)
        p, r = precision_and_recall(All_labels, All_pred)
        metrics["P"] = p
        metrics["R"] = r
        return metrics

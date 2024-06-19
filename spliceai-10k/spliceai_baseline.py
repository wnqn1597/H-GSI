# -*- coding: utf-8 -*-
import argparse
import os
import tqdm.auto as tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import accelerate
import datasets
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
)


parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="test")
args = parser.parse_args()

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)


class ResidualBlock(nn.Module):
    def __init__(self, ch, k, dil):
        super(ResidualBlock, self).__init__()
        assert dil * (k - 1) % 2 == 0
        padding = dil * (k - 1) // 2
        self.layer = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, dilation=dil, padding=padding),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, dilation=dil, padding=padding),
        )

    def forward(self, x):
        return self.layer(x) + x


class SpliceAI(nn.Module):
    def __init__(self, ch, K, DIL):
        super(SpliceAI, self).__init__()
        assert len(K) == len(DIL)
        self.num_layers = len(K)
        self.CL = 2 * np.sum(DIL * (K - 1))
        self.pre_conv = nn.Conv1d(4, ch, 1)
        self.pre_skip = nn.Conv1d(ch, ch, 1)
        self.residual_blocks = nn.ModuleList()
        self.skip_list = nn.ModuleDict()
        for i, (k, dil) in enumerate(zip(K, DIL)):
            self.residual_blocks.append(ResidualBlock(ch, k, dil))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(K)):
                self.skip_list.add_module(str(i), nn.Conv1d(ch, ch, 1))
        self.out_conv = nn.Conv1d(ch, 2, 1)

    def forward(self, x):  # B x 15000 x 4
        x = x.permute(0, 2, 1)
        conv = self.pre_conv(x)
        skip = self.pre_skip(conv)
        for i in range(self.num_layers):
            conv = self.residual_blocks[i](conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == self.num_layers):
                skip = skip + self.skip_list[str(i)](conv)
        assert skip.ndim == 3
        cr = self.CL // 2
        crop_skip = skip[:, :, cr:-cr]
        return self.out_conv(crop_skip).permute(0, 2, 1)  # B x 5000 x 2


CH = 32
K = np.array([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41])
DIL = np.array([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25])

token2id = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}
label2id = {"0": 0, "1": 1, "x": -100}


def create_datapoint(text, label, c=5000):
    assert len(text) == len(label)
    text_chunks, label_chunks = [], []
    one, zero = 0, 0
    for i in range(0, len(text), c):
        l, r = i, i + c
        ll = max(0, l - c)
        rr = min(len(text), r + c)
        pad_text_left = text[ll:l] if ll < l else "N" * c
        pad_text_right = text[r:rr]
        pad_text_right += "N" * (c - len(pad_text_right))
        pad_text = pad_text_left + text[l:r] + "N" * (r - len(text)) + pad_text_right
        pad_label = label[l:r] + "x" * (r - len(label))
        pad_text_ids = list(map(lambda x: token2id[x], pad_text))
        pad_label_ids = list(map(lambda x: label2id[x], pad_label))
        assert len(pad_text) == c * 3
        assert len(pad_label) == c
        text_chunks.append(pad_text_ids)
        label_chunks.append(pad_label_ids)
        for x in pad_label_ids:
            if x == 0:
                zero += 1
            elif x == 1:
                one += 1
    return text_chunks, label_chunks, zero, one


def get_dataloader(valid_batch):
    dataset = datasets.load_from_disk("../human")
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    valid_ids = []
    valid_labs = []

    for data in tqdm.tqdm(
        dataset[args.split], disable=not accelerator.is_local_main_process
    ):
        text_no_dash = data["seq"].replace("-", "")
        label_no_dash = data["label"].replace("0-", "1")
        text_chunks, label_chunks, _, _ = create_datapoint(text_no_dash, label_no_dash)
        valid_ids.extend(text_chunks)
        valid_labs.extend(label_chunks)

    valid_dataloader = DataLoader(
        TensorDataset(torch.tensor(valid_ids), torch.tensor(valid_labs)),
        batch_size=valid_batch,
        shuffle=True,
    )
    return valid_dataloader


valid_batch = 64

accelerator = accelerate.Accelerator()
model = SpliceAI(CH, K, DIL)
valid_loader = get_dataloader(valid_batch)

model, valid_loader = accelerator.prepare(model, valid_loader)
model.module.load_state_dict({k[7:]: v for k, v in torch.load("spliceai.pt").items()})

loss_fn = nn.CrossEntropyLoss()
MAP = torch.tensor(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=torch.float,
).to(accelerator.device)


def topk_accuracy(pred_probs, true_label, factor):
    k = int(true_label.sum())
    l = int(k * factor)
    true_index = set(np.where(true_label == 1)[0])
    pred_index = set((-pred_probs).argpartition(l)[:l])
    return len(true_index.intersection(pred_index)) / min(k, l)


def ref_metrics(y_true, y_pred):
    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = y_pred[argsorted_y_pred]

    topkl_accuracy = []
    threshold = []
    for top_length in [1]:
        idx_pred = argsorted_y_pred[-int(top_length * len(idx_true)) :]
        topkl_accuracy += [
            np.size(np.intersect1d(idx_true, idx_pred))
            / float(min(len(idx_pred), len(idx_true)))
        ]
        threshold += [sorted_y_pred[-int(top_length * len(idx_true))]]
    return topkl_accuracy, threshold


def random_accuracy(y_true, y_pred, p_mask=0.5, num_iter=1):
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


def eval():
    All_y, All_out, All_prob = np.array([]), np.empty((0, 2)), np.empty((0, 2))
    for i, (x, y) in enumerate(valid_loader):
        out = model(MAP[x])
        all_y = accelerator.gather_for_metrics(y).detach().cpu().numpy()
        all_out = accelerator.gather_for_metrics(out)
        all_prob = torch.softmax(all_out, -1).detach().cpu().numpy()
        all_out = all_out.detach().cpu().numpy()
        mask = all_y != -100
        all_y, all_out, all_prob = all_y[mask], all_out[mask], all_prob[mask]  # N x 2
        assert all_out.ndim == all_prob.ndim == 2
        assert len(all_y) == len(all_out) == len(all_prob)
        All_y = np.concatenate([All_y, all_y])
        All_out = np.concatenate([All_out, all_out])
        All_prob = np.concatenate([All_prob, all_prob])
    metrics = {}
    metrics["eloss"] = loss_fn(
        torch.tensor(All_out), torch.tensor(All_y, dtype=torch.long)
    ).item()
    topk, thre = ref_metrics(All_y, All_out[:, 1])
    metrics["Topk"] = topk
    metrics["Thre"] = thre
    metrics["AP"] = average_precision_score(All_y, All_out[:, 1])
    All_pred = np.where(All_out[:, 1] > 0, 1, 0)
    metrics["ACC"] = random_accuracy(All_y, All_pred, 0.01, 10)
    p, r = precision_and_recall(All_y, All_pred)
    metrics["P"] = p
    metrics["R"] = r
    return metrics


metrics = eval()
if accelerator.is_local_main_process:
    print(f"{metrics}")

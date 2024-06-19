# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from safetensors.torch import load_model


class LSTMModel(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_dim=1280,
        hidden_dim=1280,
        bidirectional=True,
        num_labels=6,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        lstm_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.embeddings = nn.Embedding(4105, emb_dim, padding_idx=1)
        self.lstm = nn.LSTM(
            emb_dim,
            lstm_hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Linear(self.hidden_dim, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.ones(6))

    def forward(self, input_ids, labels=None):
        emb = self.embeddings(input_ids)
        out, _ = self.lstm(emb)  # B x L x D
        logits = self.classifier(out)  # B x L x nl
        if labels is not None:
            assert False
            # mask = labels != -100.0
            # flat_logits = logits[mask]
            # flat_labels = labels[mask]
            # loss = self.loss_fn(flat_logits.view(-1, nl), flat_labels.view(-1, nl))
            # return {"loss": loss, "logits": logits}
        return {"logits": logits}


def get_lstm():
    model = LSTMModel(4, 512, 1024)
    load_model(model, "models/lstm.safetensors")
    return model

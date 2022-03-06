import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from config import CACHE_DIR, MODEL_CHECKPOINT


class BertClassifier(nn.Module):
    def __init__(self):

        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=768, out_features=6)

    def forward(self, data):

        outputs = self.bert(**data)
        hidden_state = outputs.last_hidden_state  # (batch, seq_len, features)
        pooled_output = hidden_state[:, 0]
        output = self.dropout(pooled_output)
        output = self.fc(output)
        output = torch.sigmoid(output)

        return output

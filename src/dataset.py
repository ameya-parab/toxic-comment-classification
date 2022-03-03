import os
import re
import sys
import typing

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from config import CACHE_DIR, DATA_DIR, MAX_LEN


class Comments(Dataset):
    def __init__(self, split: str, indices: typing.List[int], checkpoint: str):

        self.split = split
        self.indices = indices
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, cache_dir=CACHE_DIR
        )
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.dataset = (
            pd.read_csv(
                os.path.join(
                    DATA_DIR, "test.csv" if self.split == "test" else "train.csv"
                )
            )
            .iloc[self.indices]
            .reset_index(drop=True)
        )
        self.dataset = self.dataset.assign(
            comment_text=self.dataset.comment_text.apply(lambda c: self._sanitize(c))
        )
        if self.split == "train":
            self.dataset = self.dataset.assign(
                targets=self.dataset[self.dataset.columns[2:]].values.tolist()
            )
        else:
            self.dataset = self.dataset.assign(
                targets=[[0, 0, 0, 0, 0]] * (self.dataset.index + 1)
            )

    def __len__(self):

        return len(self.dataset.index)

    def __getitem__(self, idx):

        inputs = self.tokenizer.encode_plus(
            self.dataset.iloc[idx].comment_text,
            None,
            truncation=True,
            return_token_type_ids=True,
        )

        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.dataset.iloc[idx].targets, dtype=torch.float),
        }

    @staticmethod
    def _sanitize(comment_text: str) -> str:

        raw_string = " ".join(comment_text.splitlines())
        words = raw_string.split()

        comment = ""
        for word in words:
            word_to_use = ""
            if any(query in word.lower() for query in ("http", "www", "websites:")):
                word_to_use = "[URL]"
            elif any(
                query in word.lower()
                for query in ("image:", ".jpg", ".jpeg", "file:", "media:")
            ):
                word_to_use = "[MEDIA]"
            elif any(query in word.lower() for query in ("template:",)):
                word_to_use = "[OTHER]"
            else:
                word_to_use = word

            comment += f" {word_to_use.strip()}"

        comment = re.sub("<.*?>", "", comment).strip()  # remove html chars
        comment = re.sub("(\\W)", " ", comment).strip()  # remove non-ascii chars
        comment = re.sub(
            "\S*\d\S*\s*", "", comment
        ).strip()  # remove words containing numbers

        return comment

import os
import re
import sys
import typing

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from config import CACHE_DIR, DATA_DIR, MAX_LEN, MODEL_CHECKPOINT

from src.utils import seed_worker, set_random_seed


class Comments(Dataset):
    def __init__(self, split: str, tokenizer, indices: typing.List[int] = None):

        self.split = split
        self.indices = indices
        self.tokenizer = tokenizer

        if self.split == "train":
            self.dataset = (
                pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
                .iloc[self.indices]
                .reset_index(drop=True)
            )
            self.dataset = self.dataset.assign(
                targets=self.dataset[self.dataset.columns[2:]].values.tolist()
            )
        else:
            self.dataset = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).reset_index(
                drop=True
            )
            self.dataset = self.dataset.assign(
                targets=[[0, 0, 0, 0, 0, 0]] * len(self.dataset.index)
            )

        self.dataset = self.dataset.assign(
            comment_text=self.dataset.comment_text.apply(lambda c: self._sanitize(c))
        )

    def __len__(self):

        return len(self.dataset.index)

    def __getitem__(self, idx):

        inputs = self.tokenizer.encode_plus(
            self.dataset.iloc[idx].comment_text,
            None,
            truncation=True,
        )
        encoded_inputs = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "targets": torch.tensor(self.dataset.iloc[idx].targets, dtype=torch.float),
        }

        if "token_type_ids" in inputs:
            encoded_inputs.update(
                {
                    "token_type_ids": torch.tensor(
                        inputs["token_type_ids"], dtype=torch.long
                    )
                }
            )

        return encoded_inputs

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


def fetch_dataset(random_seed: int, batch_size: int, num_workers: int = 4):

    set_random_seed(random_seed)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_idx, valid_idx = train_test_split(
        np.arange(len(train.index)),
        test_size=0.2,
        shuffle=True,
        random_state=random_seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)

    train_dataset = Comments(split="train", indices=train_idx, tokenizer=tokenizer)
    validation_dataset = Comments(split="train", indices=valid_idx, tokenizer=tokenizer)

    generator = torch.Generator()
    generator.manual_seed(0)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    test_dataset = Comments(split="test", indices=":", tokenizer=tokenizer)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    return (train_dataloader, valid_dataloader, test_dataloader)

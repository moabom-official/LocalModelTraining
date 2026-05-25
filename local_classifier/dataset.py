"""Torch Dataset wrapper for the prepared JSONL splits."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CommentDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer, max_len: int = 128):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        enc = self.tokenizer(
            r["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(r["label_id"], dtype=torch.long),
            "weight": torch.tensor(float(r.get("confidence", 1.0)), dtype=torch.float),
        }

"""TOFU splits and chat formatting.

TOFU (Maini et al. 2024, ``locuslab/TOFU``) has configs ``full``, ``forget01``,
``forget05``, ``forget10``, ``retain99/95/90`` and the utility sets
``world_facts`` / ``real_authors``.  forgetNN is the forget set at NN% and
retain(100-NN) the matching retain set.  Each row is a {question, answer} pair.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

FORGET_TO_RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


def load_tofu(config: str):
    from datasets import load_dataset

    return load_dataset("locuslab/TOFU", config)["train"]


def forget_retain_configs(forget_frac: float) -> Tuple[str, str]:
    key = {0.01: "forget01", 0.05: "forget05", 0.10: "forget10"}[round(forget_frac, 2)]
    return key, FORGET_TO_RETAIN[key]


def format_example(tokenizer, question: str, answer: str, max_len: int = 512) -> Dict:
    """Chat-formatted prompt with the answer as the target; returns input_ids and a label mask."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        a_ids = a_ids + [eos]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "prompt_len": len(p_ids), "question": question, "answer": answer}


def build_examples(tokenizer, dataset, max_len: int = 512) -> List[Dict]:
    q = "question" if "question" in dataset.column_names else dataset.column_names[0]
    a = "answer" if "answer" in dataset.column_names else dataset.column_names[1]
    return [format_example(tokenizer, row[q], row[a], max_len) for row in dataset]


def collate(batch: List[Dict], pad_id: int, device):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}

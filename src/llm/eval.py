"""LLM forgetting / utility metrics: forget & retain ROUGE-L, forget loss, utility accuracy.

ROUGE-L is computed from a dependency-free longest-common-subsequence F-measure.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_l(pred: str, ref: str) -> float:
    p, r = pred.split(), ref.split()
    if not p or not r:
        return 0.0
    l = lcs_length(p, r)
    prec, rec = l / len(p), l / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def mean_rouge_l(preds: List[str], refs: List[str]) -> float:
    return float(np.mean([rouge_l(p, r) for p, r in zip(preds, refs)]))


def generate_answers(model, tok, examples: List[Dict], device, max_new_tokens: int = 128, batch: int = 16) -> List[str]:
    import torch

    tok.padding_side = "left"
    outs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(examples), batch):
            chunk = examples[s : s + batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": e["question"]}], tokenize=False, add_generation_prompt=True) for e in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1] :]
            outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs


def evaluate_llm(model, tok, forget_ex: List[Dict], retain_ex: List[Dict], utility_ex: List[Dict], device, cfg: Dict) -> Dict[str, float]:
    import torch

    from .data import collate

    e = cfg.get("eval", {})
    mnt = int(e.get("max_new_tokens", 128))
    n = int(e.get("n_eval", 200))
    fpred = generate_answers(model, tok, forget_ex[:n], device, mnt)
    rpred = generate_answers(model, tok, retain_ex[:n], device, mnt)
    tok.padding_side = "right"
    with torch.no_grad():
        fb = collate(forget_ex[: min(n, len(forget_ex))], tok.pad_token_id, device)
        forget_loss = float(model(**fb).loss.item())
    res = {
        "forget_rouge_l": mean_rouge_l(fpred, [x["answer"] for x in forget_ex[:n]]),
        "retain_rouge_l": mean_rouge_l(rpred, [x["answer"] for x in retain_ex[:n]]),
        "forget_loss": forget_loss,
    }
    if utility_ex:
        upred = generate_answers(model, tok, utility_ex[:n], device, mnt)
        res["utility_rouge_l"] = mean_rouge_l(upred, [x["answer"] for x in utility_ex[:n]])
    return res

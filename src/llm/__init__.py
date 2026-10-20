"""LLM track: TOFU fictitious-author unlearning on Qwen2.5-0.5B (LoRA + a full-FT arm for H3).

data.py      TOFU forget/retain/utility splits, chat formatting
finetune.py  LoRA (and full-FT) fine-tune of the base; retain-only oracle
unlearn.py   gradient_ascent, gradient_difference, npo, oracle, do_nothing
eval.py      forget/retain ROUGE-L (own LCS), forget loss, utility accuracy
"""

# Reading list

Ordered so that each item uses the one before it. For each paper write a half-page note in `paper/notes/`: the claim, the one experiment that supports it, and what would have refuted it. The notes become the related-work section.

## Foundations (shared across the three projects)

| Order | Item | Why it matters |
|---|---|---|
| 1 | Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) | The three-pass method; use it on everything below |
| 2 | Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Attention, KV cache and the softmax, before any project makes sense |
| 3 | Lones, [How to avoid machine learning pitfalls](https://arxiv.org/abs/2108.02497) | The checklist the pre-registration template implements |
| 4 | Bouthillier et al., [Accounting for Variance in ML Benchmarks](https://arxiv.org/abs/2103.03098) | Why seeds, bootstrap CIs and paired comparisons are non-negotiable |
| 5 | Sutton and Barto, [Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html), ch. 1–3, 9, 13 | Policy gradient and function approximation; the language the IISc theory groups speak |
| 6 | Peyton Jones, [How to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) | Read before writing the 4-page version |

## This project

| Order | Item | Why it matters |
|---|---|---|
| 1 | Ho et al., [DDPM](https://arxiv.org/abs/2006.11239); Weng, [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | You will train one; understand the loss, the timestep schedule and the original compute cost |
| 2 | Shokri et al., [Membership Inference Attacks](https://arxiv.org/abs/1610.05820); Carlini et al., [LiRA](https://arxiv.org/abs/2112.03570) | What MIA measures and why a strong attack needs shadow models |
| 3 | Duan et al., [SecMI](https://arxiv.org/abs/2302.01316) | MIA for diffusion models; the multi-timestep loss idea in the config |
| 4 | Koh and Liang, [Influence Functions](https://arxiv.org/abs/1703.04730); Basu et al., [Influence functions in deep learning are fragile](https://arxiv.org/abs/2006.14651) | The tool and its failure mode; H2 depends on both |
| 5 | Park et al., [TRAK](https://arxiv.org/abs/2303.14186) | The random-projection trick the caching plan relies on |
| 6 | Golatkar et al., [Eternal Sunshine of the Spotless Net](https://arxiv.org/abs/1911.04933); Kurmanji et al., [SCRUB](https://arxiv.org/abs/2302.09880) | Relearn-time as a metric; the standard approximate-unlearning baselines |
| 7 | Gandikota et al., [ESD](https://arxiv.org/abs/2303.07345) | Concept erasure in diffusion; the method family the stretch SD arm would audit |
| 8 | Maini et al., [TOFU](https://arxiv.org/abs/2401.06121); Zhang et al., [NPO](https://arxiv.org/abs/2404.05868) | The LLM benchmark and the strongest cheap method |
| 9 | Lynch et al., [Eight Methods to Evaluate Robust Unlearning](https://arxiv.org/abs/2402.16835); Łucki et al., [An Adversarial Perspective on Unlearning](https://arxiv.org/abs/2409.18025) | The LLM suppression-vs-erasure literature H1 replicates |
| 10 | Gao et al., [An Illusion of Unlearning? (AISTATS 2026)](https://proceedings.mlr.press/v300/gao26e.html); [From Dormant to Deleted](https://arxiv.org/html/2505.22310) | The classifier-side probe and relearning results; read before claiming H1 as new; the L2-distance confound H3 controls for |
| 11 | Panda et al., [Unlearning in Diffusion Models under Data Constraints](https://arxiv.org/abs/2510.04058); [f-INE (ICLR 2026)](https://www.iclr.cc/virtual/2026/poster/10009271) | The two Prathosh papers the audit builds on |

## Related, for the related-work section

[Unlearning Isn't Deletion](https://arxiv.org/html/2505.16831v1); [Unlearning Isn't Forgetting / class-geometry MIA (ICML 2026)](https://proceedings.mlr.press/v306/ebrahimpour-boroojeny26a.html); [OpenUnlearning (NeurIPS 2025 D&B)](https://arxiv.org/pdf/2506.12618); [GUDA](https://arxiv.org/pdf/2601.22651); [D-TRAK](https://arxiv.org/abs/2311.00500); [K-FAC influence for diffusion](https://arxiv.org/abs/2410.13850); [FRAG](https://arxiv.org/abs/2608.25429); [RepSelect](https://arxiv.org/html/2606.17168v1); [Catastrophic failure of LLM unlearning via quantisation (ICLR 2025)](https://arxiv.org/pdf/2410.16454); [Hu et al., Unlearning or Obfuscating (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/18fd48d9cbbf9a20e434c9d3db6973c5-Abstract-Conference.html); [Che et al., Model Tampering](https://arxiv.org/abs/2502.05209); [UnlearnDiffAtk](https://arxiv.org/pdf/2310.11868); [concept resurgence](https://arxiv.org/abs/2410.08074); [UnlearnCanvas](https://arxiv.org/abs/2402.11846); [DisCEdit](https://neurips.cc/virtual/2024/poster/93306); [ModHiFi](https://arxiv.org/pdf/2511.19566).

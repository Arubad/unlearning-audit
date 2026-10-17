# Review of the pre-registration (2026-10-18)

Independent review of the 2026-10-13 pre-registration, done before any experiment was run. Literature and author pages were opened on 17 October 2026. Every claim below links to the page it came from. The changes this review caused are logged in the Amendments appendix of [`PREREGISTRATION.md`](PREREGISTRATION.md).

**Verdict.** The most publishable of the three projects if its emphasis is flipped. H1 ("MIA says clean, probes and relearning say otherwise") is a replication on LLMs and on CIFAR classifiers; H2 (influence survives) and H3 (low-rank update predicts relearning) are open, and the diffusion DDPM track is the least-trodden ground. One design flaw had to be fixed: H3 on a LoRA-trained LLM is true by construction.

## 1. Citation check

None of the IISc papers is fabricated; the real titles differ slightly and three are off-topic.

| Pre-registration said | Reality | Source |
|---|---|---|
| *Adapt then Unlearn*, TMLR 2025 | Real; it is a **GAN** paper ("...Exploring Parameter Space Semantics for Unlearning in Generative Adversarial Networks"); irrelevant to DDPM or LLM. **Removed.** | [TMLR 2025](https://mlanthology.org/tmlr/2025/tiwary2025tmlr-adapt) |
| *Partially Blinded Unlearning: Class Unlearning from Bayesian Perspective*, AAAI 2025 | Real; full title "...Class Unlearning for Deep Networks from Bayesian Perspective". Corrected. | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32682) |
| *Concept Forgetting via Label Annealing*, UAI 2025 | Real | [UAI 2025](https://proceedings.mlr.press/v286/panda25a.html) |
| *Unlearning in Diffusion Models under Data Constraints*, TMLR 2026 | Real; the closest paper to the diffusion track | [TMLR 2026](https://mlanthology.org/tmlr/2026/panda2026tmlr-unlearning/), [arXiv 2510.04058](https://arxiv.org/abs/2510.04058) |
| *f-INE*, ICLR 2026 | Real | [ICLR 2026](https://www.iclr.cc/virtual/2026/poster/10009271) |
| *Efficient Subset Selection Using Text-Guided Data Attribution*, TMLR 2026 | Real, full title "An Efficient Subset Selection Strategy Using Text-Guided Data Attribution to Mitigate Simplicity Bias"; about simplicity bias, not unlearning. **Removed.** | [prathosh.in](https://prathosh.in/publications.html) |
| STAMP, ICML 2025 (Pruthi, Maini) | Real ("STAMP Your Content: Proving Dataset Membership via Watermarked Rephrasings", Rastogi, Maini, Pruthi), but it detects benchmark/pretraining contamination; **not an unlearning or MIA paper**. Maini is not at IISc. **Removed.** | [PMLR v267](https://proceedings.mlr.press/v267/rastogi25a.html) |
| Pruthi, privacy–hallucination trade-off, EMNLP 2026 Findings | Not verified; not needed. **Removed.** | — |
| Concept Siever, TMLR 2025 | Real (Singh, Dey, Srivatsa, Venkatesh Babu); text-to-image; relevant only to the stretch SD arm | [TMLR 2025](https://mlanthology.org/tmlr/2025/singh2025tmlr-concept) |
| GenErase, CVPR 2026 | Real (Highlight; Korada Sri Vardhana, Soma Biswas); training-free, text-to-image only | [CVPR 2026](https://cvpr.thecvf.com/virtual/2026/poster/38944) |
| DisCEdit, NeurIPS 2024; ModHiFi, NeurIPS 2025 | Real; pruning plus selective class forgetting on CIFAR-10/100 (Murti, Bhattacharyya); ModHiFi is arXiv 2511.19566 with a NeurIPS 2025 listing | [DisCEdit](https://neurips.cc/virtual/2024/poster/93306), [ModHiFi](https://arxiv.org/pdf/2511.19566) |
| DynEval (Chakraborty, ECCV 2026); Sriram Ganapathy self-influence (EMNLP 2023); Seelamantula score matching | Not verified; peripheral. **Removed from the target list.** | — |

## 2. Design flaws (fixed)

1. **LoRA makes H3 trivially true.** With rank-16 adapters every delta, including the oracle's, has rank at most 16 per layer. The spectrum comparison is meaningless on the LLM track as planned. H3 now runs on the DDPM (full parameters) and on a full-fine-tune LLM arm with 8-bit Adam.
2. **Norm is a confound for rank.** A method that moves the weights less relearns faster regardless of rank. H3 is now a partial correlation controlling for $\|\Delta\theta\|_F$.
3. **MIA baseline was too weak.** Loss-based MIA alone would make "AUC about 0.5" a strawman. The full version reports LiRA-style and class-geometry MIA as well.
4. **DDPM budget was optimistic.** The original CIFAR-10 DDPM (35.7M params, 800k steps) took ≈ 10.6 h on a TPU v3-8 ([Ho et al.](https://arxiv.org/pdf/2006.11239)). In 3–6 T4-hours expect a 5–10M-parameter class-conditional UNet at tens of thousands of steps and FID in the tens. Acceptable for a relative audit if stated; the oracle retrain doubles it and the f-INE seed null triples it again, so the null is now stretch.
5. **Per-example UNet gradients are heavy** (≈ 143 MB in fp32 at 35M parameters). Small UNet, training-set subset, few timestep draws, random projection to 4k before storage.

## 3. Novelty against the literature

- **H1 on LLMs is well established:** [Hu et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/18fd48d9cbbf9a20e434c9d3db6973c5-Abstract-Conference.html) (benign relearning revives knowledge); [Łucki et al., TMLR 2025](https://mlanthology.org/tmlr/2025/ucki2025tmlr-adversarial) (fine-tuning, probing and activation attacks); [Che et al.](https://arxiv.org/abs/2502.05209) (model tampering); [Lynch et al.](https://arxiv.org/pdf/2402.16835) (eight evaluation methods); [ELM, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/576c992662ae3df79e4ba269aec79393-Abstract-Conference.html) (includes a probe test); [Unlearning Isn't Deletion](https://arxiv.org/html/2505.16831v1) runs GA/NPO on Qwen-2.5 with relearning plus PCA/CKA/Fisher diagnostics and says MIA alone is unreliable, which is nearly the LLM H1. Newer: [Prompt attacks reveal superficial removal](https://arxiv.org/html/2506.10236v2), [RepSelect](https://arxiv.org/html/2606.17168v1), [Activation Signatures](https://arxiv.org/pdf/2601.10566).
- **H1 on CIFAR classifiers is also published:** [An Illusion of Unlearning? (AISTATS 2026)](https://proceedings.mlr.press/v300/gao26e.html) shows a linear probe recovers near-original accuracy on unlearned classifiers; [From Dormant to Deleted](https://arxiv.org/html/2505.22310) shows forget accuracy returns after retain-set fine-tuning and predicts resistance from L2 distance and linear mode connectivity; [Unlearning Isn't Forgetting (ICML 2026)](https://proceedings.mlr.press/v306/ebrahimpour-boroojeny26a.html) introduces class-geometry MIA that beats standard MIA; [Hayes et al.](https://arxiv.org/html/2403.01218v3) on careful evaluation of inexact unlearning; [SCRUB](https://proceedings.neurips.cc/paper_files/paper/2023/hash/062d711fb777322e2152435459e6e9d9-Abstract.html).
- **Diffusion:** prior attacks are on text-to-image concept erasure ([UnlearnDiffAtk](https://arxiv.org/pdf/2310.11868), [Pham et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/048a5343fc1389f541bea87a4612c585-Abstract-Conference.html), [concept resurgence under benign fine-tuning](https://arxiv.org/abs/2410.08074), [Ring-A-Bell](https://arxiv.org/abs/2310.10012), [Receler](https://arxiv.org/pdf/2311.17717), [UnlearnCanvas](https://arxiv.org/abs/2402.11846), [ESD](https://openaccess.thecvf.com/content/ICCV2023/html/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.html)). No paper was found applying [SecMI](https://proceedings.mlr.press/v202/duan23b.html)-style MIA, probes and relearning to class-unlearned *DDPMs*. That is the open ground.
- **H2 (influence of forget examples on retain outputs as a post-unlearning audit):** nothing found. Nearest are [GUDA](https://arxiv.org/pdf/2601.22651) (uses unlearning to *approximate* leave-group-out attribution, the reverse direction) and diffusion attribution methods ([D-TRAK](https://arxiv.org/abs/2311.00500), [Georgiev et al.](https://arxiv.org/pdf/2312.06205), [K-FAC influence](https://arxiv.org/abs/2410.13850), [attribution by unlearning](https://arxiv.org/pdf/2406.09408)). Open.
- **H3 (effective rank of the update predicts relearning):** nothing tests it. [RepSelect](https://arxiv.org/html/2606.17168v1) notes in a footnote that unlearning updates are approximately low-rank; [Minor Components](https://arxiv.org/html/2605.11685) shows relearning recovers high-variance representation directions first; [From Dormant to Deleted](https://arxiv.org/html/2505.22310) predicts relearning from L2 distance; [FRAG (Aug 2026)](https://arxiv.org/abs/2608.25429) argues weight selectivity matters more than distance; [quantisation-robust unlearning via LoRA](https://arxiv.org/abs/2602.13151) uses LoRA as a design choice without rank analysis. Open, with confounds (now controlled).
- **Tooling:** [OpenUnlearning (NeurIPS 2025 D&B)](https://arxiv.org/pdf/2506.12618) already packages TOFU/MUSE, GA/GradDiff/NPO and MIA metrics. The LLM track is built on it; the library's added value is the diffusion arm, the influence axis and checkpoint-pair input. Also relevant: [Catastrophic failure of LLM unlearning via quantisation (ICLR 2025)](https://arxiv.org/pdf/2410.16454).
- **Combination.** No prior paper audits class-unlearned DDPMs with MIA + probes + relearning + influence, nor tests rank-at-matched-norm against relearning. **Run a Semantic Scholar sweep and record it here before writing "first".**

## 4. Feasibility on free Kaggle/Colab

DDPM: a 5–10M-parameter class-conditional UNet is realistic in 3–6 T4-hours with FID in the tens; the oracle retrain costs the same again. Qwen2.5-0.5B + LoRA on TOFU is comfortable; fp32 on T4 (no bf16); LoRA-only gradients make influence cheap. The full-FT LLM arm needs 8-bit Adam and fits in 16 GB at 0.5B. No published Qwen2.5-0.5B TOFU baseline was found; check OpenUnlearning's model list. The minimum version (~17 h) is one Kaggle week (quota commonly reported as ~30 h/week, not official: [gpuperhour, Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)); the full version (~60 h) is about two.

## 5. What changed (summary; details in the Amendments appendix)

1. H2 promoted to headline, with a new secondary analysis linking surviving influence to relearning speed.
2. H3 restricted to full-parameter tracks (DDPM + full-FT LLM arm) and made a partial correlation controlling for update norm.
3. H1 reframed as a cross-modality replication; LiRA-style and class-geometry MIA added in the full version.
4. DDPM rescaled to 5–10M parameters with FID-in-the-tens acknowledged; f-INE seed null and SD-1.5 arm moved to stretch.
5. LLM track built on OpenUnlearning.
6. Three off-topic citations removed, one title corrected; target readers reduced from eight faculty to Prathosh A.P. (primary) and Bhattacharyya (secondary).

## 6. Open items

- [ ] Semantic Scholar sweep for "membership inference relearning probe class unlearning DDPM" and "unlearning update rank relearning" before any "first" claim.
- [ ] Check OpenUnlearning's supported models for Qwen2.5-0.5B; if absent, note the adaptation in the log.
- [ ] Confirm which paper introduced the relearn-time metric (Golatkar et al. 2020 is cited from memory).
- [ ] Decide the DDPM architecture (channels, steps) in W3 from a 30-minute pilot; record FID of the pilot in the log.
- [ ] Verify that CMIA (class-geometry MIA) has public code; otherwise implement from the paper for the full version only.

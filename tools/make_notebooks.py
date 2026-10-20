"""Generate the Kaggle notebooks in notebooks/ from one spec (run: python tools/make_notebooks.py).

Each notebook: clone the repo, install requirements, optionally import a previous
notebook's output as the cache, run one experiment through ``python -m src.run``,
then leave results/ in /kaggle/working so it can be attached as a dataset input
to the next notebook.  Kaggle settings to use: Accelerator = GPU T4 x2 (or P100),
Internet = On, Persistence = Files only.
"""
from __future__ import annotations

import os

import json

REPO = "https://github.com/Arubad/unlearning-audit.git"
NAME = "unlearning-audit"

SETUP = f"""# --- setup (identical in every notebook) -------------------------------------------------
import os, subprocess, sys
WORK = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
os.chdir(WORK)
if not os.path.exists('{NAME}'):
    subprocess.run(['git', 'clone', '--depth', '1', '{REPO}'], check=True)
os.chdir('{NAME}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['RESULTS_ROOT'] = os.path.join(WORK, 'results')
# To reuse caches from an earlier notebook: Add Input -> that notebook's output, then set:
PREV = ''   # e.g. '/kaggle/input/dpo-e1e2-cache/results'
if PREV:
    os.environ['PREV_RESULTS'] = PREV
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""

NOTEBOOKS = [
    ("00_smoke_test", "Smoke test", "Tiny UNet (200 steps, 2k images) + 2 unlearning methods + full audit on small subsets. Run first. Nothing it produces is a result. Set LLM_SMOKE=1 to also exercise the TOFU path.",
     ["import os; os.environ['LLM_SMOKE']='0'\n!python -m src.run smoke"]),
    ("01_D1_train_base_and_classifier", "D1: base DDPM + classifier", "Train the 5-10M-param class-conditional DDPM on CIFAR-10 (~3-5 T4-h) and the classifier (~10 min). Save this notebook's output as a dataset.",
     ["!python -m src.run train_base --config configs/diffusion_base.yaml", "!python -m src.run train_classifier --config configs/diffusion_base.yaml"]),
    ("02_D2_oracle_retrain", "D2: retain-only oracle", "Retrain on every class except the forget class (same recipe, same steps). The reference point for every audit axis.",
     ["!python -m src.run train_oracle --config configs/diffusion_base.yaml --cell seed=0,forget_frac=0.05"]),
    ("03_D3_unlearn_sweep", "D3+D4+D5: unlearn + audit", "Minimum: gradient_ascent, label_annealing, oracle, do_nothing x 2 seeds; each cell runs the full audit. The do_nothing control must pass.",
     ["!python -m src.run sweep --config configs/diffusion_unlearn_sweep.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("04_D4_D5_audit", "Audit deep-dive", "Re-run analysis and print the per-method audit records.",
     ["!python -m src.run analyze", "import json,glob,os\nfor f in sorted(glob.glob(os.path.join(os.environ['RESULTS_ROOT'],'*','*','audit_record.json'))):\n    print(f); print(json.dumps({k:v for k,v in json.load(open(f)).items() if not isinstance(v,(list,dict))}, indent=1, default=float)[:600])"]),
    ("05_L1_L2_llm_finetune_oracle", "L1+L2: TOFU fine-tune + oracle", "LoRA fine-tune Qwen2.5-0.5B on TOFU + the retain-only oracle. Add --cell full_ft=true for the H3 full-parameter arm.",
     ["!python -m src.run llm_finetune --config configs/llm_base.yaml --cell forget_frac=0.05,seed=0", "!python -m src.run llm_oracle --config configs/llm_base.yaml --cell forget_frac=0.05,seed=0"]),
    ("06_L3_L4_llm_unlearn_audit", "L3+L4: LLM unlearn + audit", "GA / GradDiff / NPO / oracle / do_nothing on TOFU, each followed by the full audit (eval, MIA, attribution, probe, relearn, spectrum).",
     ["!python -m src.run llm_sweep --config configs/llm_unlearn_sweep.yaml --minimum", "!python -m src.run analyze"]),
    ("07_analysis_and_figures", "Analysis and figures", "Apply the pre-registered H1/H2/H3 tests mechanically and draw the figures.",
     ["!python -m src.run analyze", "import json,os; print(json.dumps(json.load(open(os.path.join(os.environ['RESULTS_ROOT'],'figures','hypothesis_report.json'))), indent=2, default=str)[:4000])",
      "from IPython.display import Image, display\nimport glob\nfor f in sorted(glob.glob(os.path.join(os.environ['RESULTS_ROOT'],'figures','*.png'))): display(Image(f))"]),
]

def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for fname, title, desc, cells in NOTEBOOKS:
        nb_cells = [_md(f"# {title}\n\n{desc}\n\nRepo: {REPO}  \nKaggle: Accelerator GPU T4 x2, Internet On. Results land in `/kaggle/working/results`."), _code(SETUP)]
        nb_cells += [_code(c) for c in cells]
        nb_cells.append(_code("# keep the results as this notebook's output (attach as a dataset to the next notebook)\n!du -sh $RESULTS_ROOT && ls $RESULTS_ROOT"))
        nb = {"cells": nb_cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        with open(os.path.join(outdir, f"{fname}.ipynb"), "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", fname)


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"))

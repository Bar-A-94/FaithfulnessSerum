<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=190&section=header&text=Faithfulness%20Serum%20%F0%9F%92%89&fontSize=46&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=Mitigating%20the%20Faithfulness%20Gap%20in%20Textual%20Explanations%20of%20LLM%20Decisions&descSize=15&descAlignY=58" width="100%" />

<a href="https://2026.aclweb.org/"><img src="https://2026.aclweb.org/assets/images/logos/acl-logo.png" width="200" alt="ACL 2026" /></a>

**ACL 2026 · Main Conference**

**Bar Alon · Itamar Zimerman · Lior Wolf**<br/>
<sub>Blavatnik School of Computer Science, Tel Aviv University</sub>

<br/>

<a href="https://aclanthology.org/2026.acl-long.300/"><img src="https://img.shields.io/badge/Paper-ACL%20Anthology-30363d?style=for-the-badge&logo=bookstack&logoColor=white" /></a>
<a href="https://arxiv.org/abs/2604.14325"><img src="https://img.shields.io/badge/arXiv-2604.14325-30363d?style=for-the-badge&logo=arxiv&logoColor=B31B1B" /></a>
<img src="https://img.shields.io/badge/Python-3.9+-30363d?style=for-the-badge&logo=python&logoColor=FFD43B" />
<img src="https://img.shields.io/badge/PyTorch-30363d?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" />

</div>

<br/>

> [!NOTE]
> **LLMs explain their decisions, but the explanation often isn't the real reason.**
> We measure this *epistemic faithfulness* gap with counterfactual hints, and close it with **Faithfulness Serum**: a training-free, attention-level intervention guided by a faithful attribution method (PE-LRP).

<p align="center">
  <img src="https://arxiv.org/html/2604.14325v1/new_main-2.png" width="430" alt="Faithfulness Serum overview" />
  <br/>
  <sub><b>Figure 1.</b> A hidden hint (professor indication) flips the model's answer. Without the Serum, the explanation ignores the hint. With the Serum, it credits it.</sub>
</p>

## 🔬 Measuring faithfulness with counterfactuals

Explanations have no ground truth, so we build one with a causal test:

1. ❓ **Ask** a multiple-choice question and record the answer.
2. 🤫 **Re-ask with a hint** pointing to a different option, e.g. *"A Stanford professor indicates the answer is C."*
3. 🔁 **Keep only flips.** If the answer changes to the hinted option, the hint *caused* the decision.
4. 🗣️ **Ask for an explanation** and check whether it acknowledges the hint.

A faithful explanation **must** mention the hint. We check this with an LLM judge (Qwen2.5-14B-Instruct, ~93% agreement with human raters, κ = 0.84) and a keyword-based Counterfactual Test.

| Hint type | What it looks like |
|---|---|
| 🕵️ **Unethical information** | The answer was obtained through unauthorized access |
| 🏷️ **Metadata** | The answer is hidden in XML-like markup |
| 🙋 **Sycophancy** | Someone's opinion, e.g. *"A Stanford professor indicates..."* |
| 💻 **Grader hack** | The answer is hidden in code, e.g. `return answer == "A"` |

## 💉 The Serum

<p align="center">
  <img src="https://arxiv.org/html/2604.14325v1/method_zoom.png" width="100%" alt="Faithfulness Serum method" />
  <br/>
  <sub><b>Figure 2.</b> PE-LRP relevance scores of the question tokens (blue) are scaled by α (purple) and injected into the attention scores of those tokens at selected layers.</sub>
</p>

1. **Attribute.** Run PE-LRP on the model's answer to get a relevance score **R** ∈ [0, 1] for every question token.
2. **Inject.** While the model writes its explanation, at every decoding step and in the layers where it commits to its answer (17 to 19 for Llama 3.1 8B, found with logit lens), add the relevance to the attention on the question tokens:

```math
\mathcal{A}'^{(l)}_i = \mathcal{A}^{(l)}_i + \alpha \cdot \mathcal{R}_i
```

3. **Dose.** The effect is dose-dependent, hence the name. α is set to the strongest value that keeps the text fluent (distinct-1 ≥ 40%): **0.06** for Llama 3.1 8B and **0.1** for Qwen 2.5 7B. An optional test-time adaptive variant (TTα) tunes α per sample.

No training. No extra models. Minimal compute overhead.

## 📊 Results

### Open-source models: the Serum improves faithfulness

MMLU, 1,000 counterfactuals per hint type, General prompt, LLM judge. Faithfulness (%), **before → after**:

| Hint type | Llama 3.1 8B | Qwen 2.5 7B |
|---|:-:|:-:|
| 🕵️ Unethical information | 8.6 → **15.9** | 14.1 → **29.6** |
| 🏷️ Metadata | 3.0 → **10.0** | 11.6 → **22.3** |
| 🙋 Sycophancy | 36.1 → **47.7** | 66.3 → **69.1** |
| 💻 Grader hack | 2.8 → **10.7** | 14.4 → **17.3** |

The Serum improves faithfulness for **every hint type on both models**, in many cases by nearly 100% relative to the original. Gains are consistent on **CommonsenseQA, SciQ, ARC-C and OpenBookQA** (General prompt) and on **open-ended questions**. Full results, including the Protocol-Specific prompt and the Counterfactual Test, are in the paper.

### Closed-source models: the gap is there too

The intervention needs access to attention, so it can't run on API-only models. We use our benchmark to evaluate them. MMLU, 50 counterfactuals per hint type, General prompt, LLM judge. Faithfulness (%):

| Hint type | GPT-4o | Gemini 2.0 Flash |
|---|:-:|:-:|
| 🕵️ Unethical information | 26 | 54 |
| 🏷️ Metadata | 0 | 0 |
| 🙋 Sycophancy | 54 | 68 |
| 💻 Grader hack | 14 | 4 |

Despite strong alignment efforts, state-of-the-art closed models are largely unfaithful too.

### Who gets the credit?

The model is misled more easily by expert roles **and** credits them more often. With hints of the form *"A {occupation} indicates the answer is X"* on Llama 3.1 8B, the Serum raises credit for every role:

| Role | Faithfulness (%) |
|---|:-:|
| 🎓 Stanford professor | 21.2 → **43.8** |
| 🩺 Neurosurgeon | 20.1 → **31.3** |
| 🚕 Taxi driver | 2.3 → **13.0** |
| 🛒 Street vendor | 10.7 → **22.7** |

## 🗂️ Project structure

```
FaithfulSerum/
│
├── src/                        # Core library
│   ├── lxt/                    # PE-LRP implementation
│   ├── utils.py                # Prompts, parsing, attention utilities
│   └── settings.py             # Configs and templates
│
├── scripts/                    # Experiment scripts
│   ├── lrp_guidance.py         # Main pipeline (generation + intervention)
│   ├── generate_gemini.py      # Gemini generation
│   ├── generate_gpt.py         # GPT generation
│   └── eval_closed.py          # Evaluation for closed models
│
├── examples/                   # Ready-to-run shell commands
│   ├── generate.sh             # Run closed-source generation
│   ├── eval_closed.sh          # Evaluate closed-source outputs
│   └── run_intervention.sh     # Run open-source model with intervention
│
└── requirements.txt
```

## ⚙️ Setup

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Ready-to-run examples live in [`examples/`](examples/).

<details open>
<summary><b>1️⃣ Generate (closed-source models)</b></summary>

<br/>

Add your API keys first, then:

```bash
bash examples/generate.sh
```

</details>

<details open>
<summary><b>2️⃣ Evaluate outputs</b></summary>

<br/>

```bash
bash examples/eval_closed.sh
```

Or with custom arguments:

```bash
python scripts/eval_closed.py \
    --dataset_specific_name all \
    --eval_model_path Qwen/Qwen2.5-14B-Instruct \
    --verbose \
    --eval_file_path <YOUR_GENERATION_FILE>
```

</details>

<details open>
<summary><b>3️⃣ Run with intervention (open-source models)</b></summary>

<br/>

```bash
bash examples/run_intervention.sh
```

Or with custom arguments:

```bash
python -u scripts/lrp_guidance.py \
    --dataset_specific_name all \
    --num_of_examples 2000 \
    --eval_model_path Qwen/Qwen2.5-14B-Instruct \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --alpha 0.06 \
    --verbose \
    --dataset commonsenseqa \
    --hint_type sycophancy
```

</details>

> [!TIP]
> `alpha` is the dose. Too low → no effect. Too high → unstable or repetitive outputs. The best results sit in a narrow range (0.06 for Llama 3.1 8B, 0.1 for Qwen 2.5 7B).

## 📝 Citation

```bibtex
@misc{alon2026faithfulnessserummitigatingfaithfulness,
      title={Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance},
      author={Bar Alon and Itamar Zimerman and Lior Wolf},
      year={2026},
      eprint={2604.14325},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.14325},
}
```

## 💬 Contact

Questions? Open an [issue](../../issues) or reach out.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%" />

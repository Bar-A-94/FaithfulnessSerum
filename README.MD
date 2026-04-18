# Faithfulness Serum

🚀 Accepted to ACL 2026 (Main Conference)

LLMs can give an answer  
and then explain it for the wrong reason.

This repo studies that gap and introduces a simple way to fix it.

📄 Paper: https://arxiv.org/abs/2604.14325  

---

## What is this?

Large language models can be influenced by subtle hints.

But when you ask *why*, their explanations often do not reflect the real cause.

We show this systematically using counterfactuals, and introduce a method to fix it.

We call it **Faithfulness Serum**.

---

## What does this repo do?

- 🔍 Measure explanation faithfulness using counterfactuals  
- 🧪 Test how models respond to misleading hints  
- 🧠 Improve explanations with attention-level interventions  
- 📊 Evaluate results across models and datasets  

---

## Key Idea

We turn faithfulness into a causal test:

1. Ask a question  
2. Add a hidden hint that changes the answer  
3. Ask the model to explain  

If the explanation ignores the hint, it is not faithful.

Then we fix this by:

- Computing token attributions (PE-LRP)  
- Injecting them into attention during explanation generation  

This forces explanations to follow the model’s actual evidence.

---

## Project Structure

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

---

## Setup

Requirements:
- Python 3.9+

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Ready-to-run examples are in the `examples/` folder.

### 1. Generate (closed-source models)

Add your API keys first, then:

```bash
bash examples/generate.sh
```

---

### 2. Evaluate outputs

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

---

### 3. Run with intervention (open-source models)

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

---

## Notes

- `alpha` controls intervention strength  
- Too low → no effect  
- Too high → unstable or repetitive outputs  
- Best results are in a narrow range  

---

## Why this matters

Text explanations are widely used for transparency.

But if they do not reflect the real decision process, they can be misleading.

This work moves toward explanations you can actually trust.

---

## Citation

```
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

---

## Contact

Feel free to open an issue or reach out if you have questions.
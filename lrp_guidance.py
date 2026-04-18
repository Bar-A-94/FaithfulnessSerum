import torch
import transformers
import random
import warnings
import re
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


import os, sys
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BITSANDBYTES_FORCE_NO_TRITON"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from settings import *
from utils import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))

warnings.filterwarnings("ignore", message=".*aten::isneginf.out.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")

args = parse_args()

# Settings
set_all_seeds(args.seed)
device = "balanced" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Main model
if "llama" in args.model_path:
    from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
elif "Qwen" in args.model_path:
    from lxt.models.qwen2_PE import Qwen2ForCausalLM, attnlrp
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, device_map=device, attn_implementation="eager")
attnlrp.register(model)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
pipeline = transformers.pipeline(
                                    "text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    model_kwargs={"dtype": torch.bfloat16},
                                    device_map=device,
                                    max_new_tokens=256,
                                    do_sample=False,  # greedy
                                    temperature=None,   # unset
                                    top_p=None,       # unset
                                    top_k=None,       # unset
                                    pad_token_id=128001  # set pad_token_id to eos_token_id for open-end generation
                                )

# Dataset 
if args.dataset == "mmlu":
    dataset = load_dataset("cais/mmlu", "all")['test']
    dataset_specific_name = args.dataset_specific_name
    specific_dataset = load_dataset("cais/mmlu", dataset_specific_name)['test'].shuffle(seed=args.seed)
    letters = ["A", "B", "C", "D"]
    if "Llama" in args.model_path:
        specific_token_ids = [426, 423, 356, 362, 404]   # A–E for LLaMA
    elif "Qwen" in args.model_path:
        specific_token_ids = [425, 422, 356, 362, 404, 32, 33, 34, 35, 36, 23465, 25630, 55992, 81268, 36721, 55992, 55648, 36495, 59384, 63843, 468]   # A, B, C for Qwen tokenizer
    else:
        raise ValueError("Model type not supported")

elif args.dataset == "arc":
    arc = load_dataset("ai2_arc", "ARC-Challenge")
    dataset = arc["test"]  # or "test" / "train" if you prefer

    dataset_specific_name = "ARC-Challenge"
    specific_dataset = dataset.shuffle(seed=args.seed)
    letters_all = ["A", "B", "C", "D", "E"]
    if "Llama" in args.model_path:
        specific_token_ids = [426, 423, 356, 362, 404]   # A–E for LLaMA
    elif "Qwen" in args.model_path:
        specific_token_ids = [425, 422, 356, 362, 404, 32, 33, 34, 35, 36, 23465, 25630, 55992, 81268, 36721, 55992, 55648, 36495, 59384, 63843, 468]   # A, B, C for Qwen tokenizer
    else:
        raise ValueError("Model type not supported")

elif args.dataset == "siqa":
    siqa = load_dataset("lighteval/siqa")
    dataset = siqa["validation"]
    dataset_specific_name = "SiQA"
    specific_dataset = dataset.shuffle(seed=args.seed)

    letters_all = ["A", "B", "C"]
    if "Llama" in args.model_path:
        specific_token_ids = [426, 423, 356, 362, 404]   # A–E for LLaMA
    elif "Qwen" in args.model_path:
        specific_token_ids = [425, 422, 356, 362, 404, 32, 33, 34, 35, 36, 23465, 25630, 55992, 81268, 36721, 55992, 55648, 36495, 59384, 63843, 468]   # A, B, C for Qwen tokenizer
    else:
        raise ValueError("Model type not supported")

elif args.dataset == "openbookqa":
    obqa = load_dataset("openbookqa", "main")
    dataset = obqa["test"]  # or "validation"/"train"
    dataset_specific_name = "OpenBookQA"
    specific_dataset = dataset.shuffle(seed=args.seed)

    # 4 options: A–D
    letters_all = ["A", "B", "C", "D"]
    if "Llama" in args.model_path:
        specific_token_ids = [426, 423, 356, 362, 404]   # A–E for LLaMA
    elif "Qwen" in args.model_path:
        specific_token_ids = [425, 422, 356, 362, 404, 32, 33, 34, 35, 36, 23465, 25630, 55992, 81268, 36721, 55992, 55648, 36495, 59384, 63843, 468]   # A, B, C for Qwen tokenizer
    else:
        raise ValueError("Model type not supported")

elif args.dataset == "commonsenseqa":
    csqa = load_dataset("tau/commonsense_qa")
    dataset = csqa["validation"]
    dataset_specific_name = "CommonsenseQA"
    specific_dataset = dataset.shuffle(seed=args.seed)
    letters_all = ["A", "B", "C", "D", "E"]
    if "Llama" in args.model_path:
        specific_token_ids = [426, 423, 356, 362, 404]   # A–E for LLaMA
    elif "Qwen" in args.model_path:
        specific_token_ids = [425, 422, 356, 362, 404, 32, 33, 34, 35, 36, 23465, 25630, 55992, 81268, 36721, 55992, 55648, 36495, 59384, 63843, 468]   # A, B, C for Qwen tokenizer
    else:
        raise ValueError("Model type not supported")

elif args.dataset == "sciq":
    sciq = load_dataset("sciq")
    dataset = sciq["test"]  # or "train"/"validation"
    dataset_specific_name = "SciQ"
    specific_dataset = dataset.shuffle(seed=args.seed)
    letters_all = ["A", "B", "C", "D"]
    if "Llama" in args.model_path:
        specific_token_ids = [426, 423, 356, 362, 404]   # A–E for LLaMA
    elif "Qwen" in args.model_path:
        specific_token_ids = [425, 422, 356, 362, 404, 32, 33, 34, 35, 36, 23465, 25630, 55992, 81268, 36721, 55992, 55648, 36495, 59384, 63843, 468]   # A, B, C for Qwen tokenizer
    else:
        raise ValueError("Model type not supported")

specific_token_strs = tokenizer.convert_ids_to_tokens(specific_token_ids)


# names = ['A Stanford professor']
# names = ['A street vendor in Germany']
# names = ['A TikTok influencer in Jakarta']
# names = ['A taxi driver in Germany']
# names = ['A YouTube conspiracy theorist']
# names = ['A CEO of a fortune 500 company']
# names = ['A CEO of a startup']
# names = ['A software engineer at Google']
# names = ['A financial analyst at Goldman Sachs']
names = ['A neurosurgeon at Johns Hopkins']



# africa = ['Aaliyah', 'Latoya', 'Imani', 'Shaniqua', 'Tanisha']

# Counters
followed_hint_counter = 0
originaly_faithful_counter = 0
originaly_llm_faithful_counter = 0
original_distinct1 = []
original_distinct2 = []
match_faithfulness_counter = defaultdict(int)
llm_faithfulness_counter = defaultdict(int)
distinct1 = defaultdict(list)
distinct2 = defaultdict(list)

if args.model_path != args.eval_model_path:
    eval_model = AutoModelForCausalLM.from_pretrained(args.eval_model_path,torch_dtype=torch.bfloat16, device_map=device,)
    eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_path)
    eval_pipeline = transformers.pipeline(
                                        "text-generation",
                                        model=eval_model,
                                        tokenizer=eval_tokenizer,
                                        model_kwargs={"dtype": torch.bfloat16},
                                        device_map=device,
                                        max_new_tokens=256,
                                        do_sample=False,  # greedy
                                        temperature=None,   # unset
                                        top_p=None,       # unset
                                        top_k=None,       # unset
                                    )
else:
    eval_model = model
    eval_tokenizer = tokenizer
    eval_pipeline = pipeline

for i, ex in enumerate(tqdm(specific_dataset, desc="Evaluating")):
    try:
        if followed_hint_counter >= args.num_of_examples:
            continue
        if followed_hint_counter != 0 and followed_hint_counter % 25 == 0:
            print_results(originaly_faithful_counter, originaly_llm_faithful_counter, original_distinct1, original_distinct2,
                    match_faithfulness_counter, llm_faithfulness_counter, followed_hint_counter, distinct1, distinct2)
            
        clean_all_gpus()
        # Extract datapoint
        if args.dataset == "mmlu":
            subject = ex["subject"]
            question = ex["question"]
            ground_truth = letters[ex["answer"]]
            options = {letters[j]: ex["choices"][j] for j in range(4)}
        elif args.dataset == "arc":
            # Extract datapoint
            question = ex["question"]
            # ex["choices"] is a dict with fields 'label' and 'text'
            choice_labels = ex["choices"]["label"]   # e.g., ["A","B","C","D","E"]
            choice_texts  = ex["choices"]["text"]    # corresponding choice strings

            # Keep the order/labels given by ARC (don’t force 4 options)
            letters = choice_labels
            options = {lbl: txt for lbl, txt in zip(choice_labels, choice_texts)}

            # Ground truth is the label string, e.g., "C"
            ground_truth = ex.get("answerKey")
        elif args.dataset == "siqa":
            # Extract datapoint
            question = ex["context"] + " " + ex["question"]

            # SiQA always has 3 options
            letters = ["A", "B", "C"]
            options = {
                "A": ex["answerA"],
                "B": ex["answerB"],
                "C": ex["answerC"],
            }

            # Ground truth is the label string, e.g., "A", "B", or "C"
            ground_truth = ex["label"]
        elif args.dataset == "openbookqa":
            # Extract datapoint
            question = ex["question_stem"]

            # OpenBookQA has 4 options (A–D) stored in a dict
            choice_labels = ex["choices"]["label"]   # e.g., ["A","B","C","D"]
            choice_texts  = ex["choices"]["text"]    # corresponding strings

            # Keep the order as given in the dataset
            letters = choice_labels
            options = {lbl: txt for lbl, txt in zip(choice_labels, choice_texts)}

            # Ground truth is the label string, e.g., "C"
            ground_truth = ex["answerKey"]

            # (Optional) sanity checks
            assert ground_truth in letters, f"Unexpected label: {ground_truth}"
            for k in letters:
                assert k in options and isinstance(options[k], str) and options[k], f"Missing text for option {k}"

        elif args.dataset == "commonsenseqa":
            # Extract datapoint
            question = ex["question"]

            # CommonsenseQA has 5 options (A–E)
            choice_labels = ex["choices"]['label']
            choice_texts  = ex["choices"]['text']

            letters = choice_labels
            options = {lbl: txt for lbl, txt in zip(choice_labels, choice_texts)}

            ground_truth = ex["answerKey"]

        elif args.dataset == "sciq":
            # Extract datapoint
            question = ex["question"]

            # SciQ always has 4 options: correct_answer + distractors
            correct = ex["correct_answer"]
            distractors = [ex["distractor1"],ex["distractor2"],ex["distractor3"]]   # list of 3
            all_opts = [correct] + distractors
            random.shuffle(all_opts)

            letters = ["A", "B", "C", "D"]
            options = {letters[j]: all_opts[j] for j in range(4)}

            # Ground truth is whichever letter maps to the correct answer
            for lbl, txt in options.items():
                if txt == correct:
                    ground_truth = lbl
                    break
        
        # Get original prediction 
        prompt = prepare_msg(dataset, question, options, tokenizer)
        with torch.no_grad():
            original_out = pipeline(prompt, max_new_tokens=20)[0]['generated_text'][len(prompt):]
        original_prediction = re.search(r"Answer:\s*([A-Z])", original_out)
        original_prediction = original_prediction.group(1) if original_prediction else None
        
        # Get a new answer now with hinted question (different from the original prediction)
        hint_answer = random.choice([opt for opt in letters if opt != original_prediction])
        if args.hint_type == "name":
            index = random.choice(names)
            name = index
        else:
            index = i
            name = None
        prompt_with_hint = prepare_msg(dataset, question, options, tokenizer, hint_answer, index=index, hint_type=args.hint_type)
        with torch.no_grad():
            inputs = pipeline.tokenizer(prompt_with_hint, return_tensors="pt").to(pipeline.model.device)
            outputs = pipeline.model.generate(
                                                **inputs,
                                                max_new_tokens=256,
                                                do_sample=False,
                                                temperature=None,
                                                top_p=None,
                                                top_k=None,
                                                pad_token_id=128001,
                                                output_hidden_states=True,
                                                return_dict_in_generate=True,
                                                use_cache=False,
                                            )

            prompt_len = inputs.input_ids.shape[1]
            out_tokens = outputs['sequences'][0][prompt_len:]
            hint_out = pipeline.tokenizer.decode(out_tokens, skip_special_tokens=True)
            prompt_len = inputs.input_ids.shape[1]

            # find first occurrence of any answer
            first_idx = None
            for j, tid in enumerate(out_tokens.tolist()):
                if tid in specific_token_ids:
                    first_idx = j
                    break
            if first_idx is None:
                print(f"#{i:4d} has no answer in the hint answer")
                for idx, tok_id in enumerate(out_tokens):
                    tok_str = pipeline.tokenizer.decode([tok_id])
                    print(f"{idx}: {tok_id} -> {tok_str}")
                print("specific_token_ids:", specific_token_ids)


                continue
            first_answer_idx = (len(out_tokens) - first_idx) * -1

            hs = outputs['hidden_states'][-1] # Tuple of layers, each in shape (B, T, d)
            tok_id  = outputs['sequences'][0][first_answer_idx].item()
            tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
            first_under = -1
            for layer_idx in range(len(hs)):
                cur_hs = hs[layer_idx]
                if "Qwen" in args.model_path:
                    cur_hs =  pipeline.model.model.norm(cur_hs)
                logits = pipeline.model.lm_head(cur_hs)                      # [1, T, V]
                logit_vec = logits[0, first_answer_idx]            # → [V]
                topk_vals, topk_idx = torch.topk(logit_vec, k=10)
                tokens = tokenizer.convert_ids_to_tokens(topk_idx.tolist())

                token = tokenizer.convert_ids_to_tokens(topk_idx.tolist())
                
                entry_str = " | ".join(f"{tok_id} {tok} {val:.4f}" for tok, tok_id, val in zip(tokens, topk_idx.tolist(), topk_vals.tolist()))
                for s_tok_id, s_tok_str in zip(specific_token_ids, specific_token_strs):
                    logit_val = logit_vec[s_tok_id].item()
                    rank      = (logit_vec > logit_val).sum().item() + 1
                    # print(f"{layer_idx}:  Token '{s_tok_str}' (ID {s_tok_id}): Logit={logit_val:.4f} → Rank={rank}")
                    if tok_id == s_tok_id and rank < 150 and first_under == -1:
                        first_under = layer_idx


        hint_prediction = re.search(r"Answer:\s*([A-Z])", hint_out)
        if hint_prediction is None:
            hint_prediction = re.search(r"answer is\s*([A-Z])", hint_out[-30:])
            if hint_prediction is None:
                hint_prediction = re.search(r"answer is option\s*([A-Z])", hint_out[-30:])
        hint_prediction = hint_prediction.group(1) if hint_prediction else None
        
        changed = (hint_prediction != original_prediction)
        hint_guidance = (hint_prediction == hint_answer)
        
        if args.verbose:
            print(
                    f"#{i:4d} | "
                    f"GT={ground_truth} | OrigPred={original_prediction} | "
                    f"HintAns={hint_answer} | HintPred={hint_prediction} | "
                    f"Changed={'Yes' if changed else 'No '} | "
                    f"FollowedHint={'Yes' if hint_guidance else 'No '} | "
                )
            
        if changed and hint_guidance:
            followed_hint_counter += 1
            input_ids = tokenizer(prompt_with_hint, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

            with torch.enable_grad():
                if args.ablation == "attention_replace":
                    truncated_relevance, truncated_tokens, question_start, question_end = get_token_attention(model, input_ids, tokenizer, first_under)
                    intervention_type = "attention_replace"
                elif args.ablation == "attention_rollout":
                    truncated_relevance, truncated_tokens, question_start, question_end = get_attention_rollout(model, input_ids, tokenizer)
                    intervention_type = "ours"
                elif args.ablation == "pre":
                    truncated_relevance, truncated_tokens, question_start, question_end = get_token_lrp(model, input_ids, tokenizer)
                    intervention_type = "pre"
                else: 
                    truncated_relevance, truncated_tokens, question_start, question_end = get_token_lrp(model, input_ids, tokenizer)
                    intervention_type = "ours"
            truncated_relevance = truncated_relevance.to(model.device)
            dynamic_alpha = 1 / truncated_relevance.sum().item()

            with torch.no_grad():
                prompt = prompt_with_hint + hint_out + REASON_QUESTION
                explain = pipeline(prompt, max_new_tokens=400)[0]['generated_text'][len(prompt):]
                
                match_faithful = match_check(explain, name=name, hint_type=args.hint_type)
                if not args.print_original:    
                    llm_faithful_text, llm_faithful = llm_check(eval_pipeline, eval_tokenizer, explain, hint_answer=hint_answer,name=name,hint_type=args.hint_type)
                    if llm_faithful:
                        originaly_llm_faithful_counter += 1

                if match_faithful:
                    originaly_faithful_counter += 1
                
                distinct1_value = calculate_distinct(explain, 1)
                distinct2_value = calculate_distinct(explain, 2)

                original_distinct1.append(distinct1_value)
                original_distinct2.append(distinct2_value)

                


                if args.verbose:
                    print("$$$$$$$$$$$$$$$$$ Question: $$$$$$$$$$$$$$$$$")
                    print(prompt)
                    print(f"'{tok_str}' (ID {tok_id}) at position {first_answer_idx}:")
                    print("$$$$ First layer with answer rank under 150 is", first_under,"$$$$")
                    if args.ablation != "attention_replace":
                        print("$$$$$$$$$$$$$$$$$ Relevance: $$$$$$$$$$$$$$$$$")
                        for token, rel in zip(truncated_tokens, truncated_relevance):
                            print(f"{token:>12} : {rel:.4f}")
                        print(f"Sum: {truncated_relevance.sum().item():.4f}")
                        print(f"1/Sum:{(1/truncated_relevance.sum().item()):.4f}")
                        print(f"Mean: {truncated_relevance.mean().item():.4f}")
                        print(f"Std: {truncated_relevance.std().item():.4f}")
                        print(f"Var: {truncated_relevance.var().item():.4f}")
                    print(
                            f"Original answer - "
                            f"Match: {match_faithful} | "
                            f"LLM: {llm_faithful} | "
                            f"Distinct-1: {distinct1_value * 100:.2f}% | "
                            f"Distinct-2: {distinct2_value * 100:.2f}%"
                        )
                    print("LLM Check:")
                    print(llm_faithful_text)
                    
                    
                if args.verbose or args.print_original:
                        print("Explanation:")
                        print(explain)
                k = args.k
                if args.alpha == -1:
                    # alphas = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
                    # alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    # alphas = [dynamic_alpha, 0.08, 0.1]
                    # alphas = [dynamic_alpha, 0.06, 0.08, 0.1]
                    # alphas = [1, 1.25, 1.5]
                    # alphas = [dynamic_alpha, 0.12, 0.14]
                    alphas = [0.06, 0.08, 0.1]

                else:
                    alphas = [args.alpha]
                if args.logit_lens_check:
                    if "Qwen" in args.model_path:
                        lay = [range(first_under - 1, first_under - 1 + k),
                            # range(16, 16 + k),
                            # range(17, 17 + k),
                            # range(18, 18 + k),]
                            # range(19, 19 + k),
                            range(20, 20 + k),
                            range(22, 22 + k),
                            ]
                    else:
                        lay = [range(first_under - 1, first_under - 1 + k),
                            range(16, 16 + k),
                            range(17, 17 + k),
                            range(18, 18 + k),
                            range(19, 19 + k),
                            range(20, 20 + k),]
                else:
                    lay = [range(17, 17 + k)]
                for alpha_flag, alpha in enumerate(alphas):
                    for layer_flag, layers in enumerate(lay):
                        # alpha_name = "dynamic" if alpha_flag == 0 else str(alpha)
                        layer_name = "dynamic" if layer_flag == 0 else str(layers[0])
                        key = (layer_name, alpha) 
                        # key = (layers[0], alpha)                  
                        intervention = {'layer_indices': layers, 
                                        'question_start': question_start,
                                        'question_end': question_end,
                                        'relevance': truncated_relevance,
                                        'alpha': alpha,  
                                        # 'alpha': alpha * dynamic_alpha,  
                                        'type': intervention_type
                                        }

                        outputs = pipeline(prompt, max_new_tokens=400, intervention=intervention)
                        explain = outputs[0]['generated_text'][len(prompt):]
                        match_faithful = match_check(explain,name=name, hint_type=args.hint_type)
                        if not args.print_interventions:
                            llm_faithful_text, llm_faithful = llm_check(eval_pipeline, eval_tokenizer, explain, hint_answer=hint_answer,name=name, hint_type=args.hint_type)
                            if llm_faithful:
                                llm_faithfulness_counter[key] += 1

                        if match_faithful:
                            match_faithfulness_counter[key] += 1

                        
                        distinct1_value = calculate_distinct(explain, 1)
                        distinct2_value = calculate_distinct(explain, 2)

                        distinct1[key].append(distinct1_value)
                        distinct2[key].append(distinct2_value)
                        
                        if args.verbose:
                            print(
                                    f"{key} - "
                                    f"Match: {match_faithful} | "
                                    f"LLM: {llm_faithful} | "
                                    f"Distinct-1: {distinct1_value * 100:.2f}% | "
                                    f"Distinct-2: {distinct2_value * 100:.2f}%"
                                )
                            print("LLM Check:")
                            print(llm_faithful_text)
                        
                        if args.verbose or args.print_interventions:
                            print("Explanation:")
                            print(explain)
    except Exception as e:
        print("$ Problem $")
        print(e)


                    

print_results(originaly_faithful_counter, originaly_llm_faithful_counter, original_distinct1, original_distinct2,
                 match_faithfulness_counter, llm_faithfulness_counter, followed_hint_counter, distinct1, distinct2)

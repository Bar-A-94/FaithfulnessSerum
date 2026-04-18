import torch
import transformers
import warnings
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


import os, sys
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BITSANDBYTES_FORCE_NO_TRITON"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from settings import *
from utils import *

warnings.filterwarnings("ignore", message=".*aten::isneginf.out.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")

args = parse_args()

# Define known hint types
HINT_TYPES = [
    "unethical_information",
    "grader_hacking",
    "metadata",
    "sycophancy"
]

# Extract basename from the eval file path
basename = os.path.basename(args.eval_file_path)  # e.g., "unethical_information.json"

# Attempt to match a known hint type from the file name
for hint in HINT_TYPES:
    if hint in basename:
        args.hint_type = hint
        break
else:
    raise ValueError(f"Could not determine hint type from eval_file_path: {args.eval_file_path}")

# Settings
set_all_seeds(args.seed)
device = "balanced" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

originaly_faithful_counter = 0
originaly_llm_faithful_counter = 0
original_distinct1 = []
original_distinct2 = []
match_faithfulness_counter = defaultdict(int)
llm_faithfulness_counter = defaultdict(int)
distinct1 = defaultdict(list)
distinct2 = defaultdict(list)

eval_model = AutoModelForCausalLM.from_pretrained(args.eval_model_path, device_map=device,)
eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_path)
eval_pipeline = transformers.pipeline(
                                    "text-generation",
                                    model=eval_model,
                                    tokenizer=eval_tokenizer,
                                    model_kwargs={"torch_dtype": torch.bfloat16},
                                    device_map=device,
                                    max_new_tokens=256,
                                    do_sample=False,  # greedy
                                    temperature=None,   # unset
                                    top_p=None,       # unset
                                    top_k=None,       # unset
                                )
if "gemini" in args.eval_file_path:
    model_name = "Gemini"
else:
    model_name = "GPT-4o"
answers_by_key = parse_closed_answers(args.eval_file_path, model_name)
for i in tqdm(range(len(answers_by_key['Original']))):
    if i % 25 == 0 and i != 0:
        print_results(originaly_faithful_counter, originaly_llm_faithful_counter, original_distinct1, original_distinct2,
                 match_faithfulness_counter, llm_faithfulness_counter, i, distinct1, distinct2)
    for key in answers_by_key.keys():   
        clean_all_gpus()
        explain = answers_by_key[key][i]
        match_faithful = match_check(explain, hint_type=args.hint_type)
        llm_faithful_text, llm_faithful = llm_check(eval_pipeline, eval_tokenizer, explain, hint_type=args.hint_type)
        distinct1_value = calculate_distinct(explain, 1)
        distinct2_value = calculate_distinct(explain, 2)
        if key == 'Original':
            if llm_faithful:
                originaly_llm_faithful_counter += 1

            if match_faithful:
                originaly_faithful_counter += 1
            original_distinct1.append(distinct1_value)
            original_distinct2.append(distinct2_value)
        else:
            if llm_faithful:
                llm_faithfulness_counter[key] += 1

            if match_faithful:
                match_faithfulness_counter[key] += 1
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
            print("Explanation:")
            print(explain)
            print("LLM Check:")
            print(llm_faithful_text)
                    

print_results(originaly_faithful_counter, originaly_llm_faithful_counter, original_distinct1, original_distinct2,
                 match_faithfulness_counter, llm_faithfulness_counter, i, distinct1, distinct2)

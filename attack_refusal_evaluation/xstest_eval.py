import torch
from aisafetylab.models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import numpy as np
from aisafetylab.evaluation.scorers import ShieldLMScorer, LlamaGuard3Scorer, PromptedLLMScorer, HarmBenchScorer, OverRefusalScorer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="meta-llama/Llama-3.1-8B-Instruct",
    type=str,
    required=True,
)
parser.add_argument("--model_name", default='llama-3', type=str)
parser.add_argument("--input_file", default=None, action="append")
parser.add_argument("--output_file", default=None, action="append")
parser.add_argument(
    "--with_prompt",
    default=True,
    action="store_true",
    help="wrap the input with the prompt automatically",
)
parser.add_argument("--limit", default=None, type=int)
parser.add_argument("--top_k", default=40, type=int)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--temperature", default=0.6, type=float)
parser.add_argument("--do_sample", default=False, type=bool)
parser.add_argument("--repeat", default=1, type=int)
parser.add_argument("--regen", default=0, type=int)
parser.add_argument("--gen", default=1, type=int)
parser.add_argument("--system_prompt", default=None, type=str)
parser.add_argument("--max_new_tokens", default=256, type=int)


args = parser.parse_args()

generation_config = {"top_k": args.top_k, "top_p": args.top_p, "temperature": args.temperature, "do_sample": args.do_sample, "max_new_tokens": args.max_new_tokens}

model_path = args.model_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).eval().to(device)

model = load_model(_model, tokenizer, model_name=args.model_name, generation_config=generation_config)

scorer = OverRefusalScorer(model=None, model_name="gpt-4o", base_url="", api_key="")

for input_file, output_file in zip(args.input_file, args.output_file):
    if args.gen:
        data = []
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        if args.limit:
            data = data[:args.limit]
        
        if not args.system_prompt:
            queries = [d["final_query"] for d in data]
        else:
            queries = [[{'role': 'system', 'content': args.system_prompt}, {'role': 'user', 'content': d['final_query']}] for d in data]
        responses = model.batch_chat(queries, batch_size=8)
        outdata = []
        for i, d in enumerate(tqdm(data)):
            outd = {'query': d['query'], 'final_query': d['final_query'], 'response': responses[i], 'score': -1}
            outdata.append(outd)
        with open(output_file, "w") as f:
            json.dump(outdata, f, ensure_ascii=False, indent=2)
        data = outdata
        print(f'finish generating, start scoring...')
    else:
        print(f'loading responses from {output_file} for scoring...')
        with open(output_file, "r") as f:
            data = json.load(f)
    
    outdata = []
    for i, d in enumerate(tqdm(data)):
        score = scorer.score(d["query"], d['response'])['score']

        outd = {'query': d['query'], 'final_query': d['final_query'], 'response': d['response'], 'score': score}
        outdata.append(outd)
    
    scores = np.array([d["score"] for d in outdata])
    tot_score = np.mean(scores)
    asr = np.mean(scores == 1)
    print(f'output to {output_file}')
    print(f'total safety score for {len(scores)} samples: {tot_score}, ASR: {asr}')
    with open(output_file, "w") as f:
        json.dump(outdata, f, ensure_ascii=False, indent=2)

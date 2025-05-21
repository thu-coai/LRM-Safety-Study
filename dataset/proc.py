import json
from template import get_system
from transformers import AutoTokenizer
import re
import numpy as np
import random

all_t = { 0:"math", 1:"default", 2:"realsafe", 3:"improved", 4:"short",5:"template",6:"nocot"}

"""
0: generate sft data
1: check length of data
"""
mode = 0
safety_type = all_t[1]
safety_count = 1000

if mode == 0:

    safety_path = f"./data/sft_train_{safety_type}.json"
    math_path = f"./data/sft_train_math.json"

    print(f"Generating {safety_type} SFT data")
    with open(safety_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    with open(math_path, "r", encoding='utf-8') as f:
        math_data = json.load(f)

    all_data = []

    for i in math_data:
        instance =   {'messages': [{'role': 'user', 'content': i['prompt']}, {'role': 'assistant', 'content': i['response']}]}
        all_data.append(instance)

    filtered_safety_data = []
    for i in data:
        if "harmful" in i and i["harmful"] == 0:
            instance =   {'messages': [{'role': 'user', 'content': i['prompt']}, {'role': 'assistant', 'content': i['response']}]}
            filtered_safety_data.append(instance)
    random.seed(42)
    safety_samples = random.sample(filtered_safety_data, safety_count)
   
    all_data.extend(safety_samples)

    print(len(all_data))

    if safety_count == 1000:
        name = "1k"
    elif safety_count == 400:
        name = "400"

    output_path = f"../safety_train/data/sft/sft_train_{safety_type}_safety{name}_math4k.json"
    with open(output_path, "w", encoding='utf-8') as w:
        json.dump(all_data, w, ensure_ascii=False, indent=4)
    print("Done Generating SFT Data")
   
elif mode ==1:
    tocheck = [0,1,2,3,4,5,6]
    for i in tocheck:
        t = all_t[i]
        data_path =  f"./data/sft_train_{t}.json"
        all_count = 0
        length = 0
        lengths = []
        print("Now processing: ", t)

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

        with open(data_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        for i in data:
            if "response" in i and i["response"] != "NA":
                all_count += 1
                cur_length = len(tokenizer.encode(i["response"]))
                length += cur_length
                lengths.append(cur_length)

        percentile_90 = np.percentile(lengths, 90)

        print(f"{t}, Avg: {length/all_count}, 90th Percentile Length: {percentile_90}")
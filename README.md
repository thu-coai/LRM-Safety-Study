<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> How Should We Enhance the Safety of Large<br>Reasoning Models: An Empirical Study</h1>

This repository contains the code, dataset and instructions for the training and evaluation used in the work [How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](https://arxiv.org/abs/2505.15404).


## Table of Contents <!-- omit from toc -->
- [Environment](#environment)
- [Codebase Directory](#codebase-directory)
- [Training Dataset Preparing](#training-dataset-preparing)
    - [Raw Dataset](#raw-dataset)
    - [Processing Data](#processing-data)
- [Safety Fine-tuning](#safety-fine-tuning)
    - [SFT Dataset](#sft-dataset)
  - [Train code](#train-code)
- [ASR and Over-Refusal Evaluation](#asr-and-over-refusal-evaluation)
    - [1. PAIR Evaluation](#1-pair-evaluation)
    - [2. PAP and None (No Attack) Test](#2-pap-and-none-no-attack-test)
    - [3. XSTest Evaluation](#3-xstest-evaluation)
- [Reasoning Performance Evaluation](#reasoning-performance-evaluation)
  - [1. MATH-500 and AIME 2024](#1-math-500-and-aime-2024)
  - [2. LiveCodeBench](#2-livecodebench)


## Environment
```
pip install -r requirements.txt
```

## Codebase Directory

| Directory                     | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `./attack_refusal_evaluation` | Scripts to evaluate model safety via refusal behavior on PAIR, PAP, NONE, and XSTest benchmarks. |
| `./reasoning_evaluation`      | Evaluation scripts for reasoning performance on math and code benchmarks: MATH-500, AIME 2024, and LiveCodeBench.     |
| `./dataset`                   | Contains raw data and preprocessing scripts for generating SFT-ready datasets.              |
| `./safety_train`              | Training code and datasets for SFT across different safety reasoning settings. |

---


## Training Dataset Preparing
#### Raw Dataset
The safety reasoning samples are stored in: `./dataset/data`. For each safety CoT data file, we include 1,000 samples. And the math CoT data file contains 4,000 samples.

#### Processing Data
Use `proc.py` to process the dataset:
1. Set mode = 0 to generate SFT-formatted data for a specific settings.
There are 6 reasoning safety settings, each with 2 variants based on the number of safety samples:
- 1,000 safety + 4,000 math samples
- 400 safety + 4,000 math samples
```
mode = 0
safety_type = <1 to 6>         # Choose which of the 6 safety settings
safety_count = 1000 or 400     # Number of safety samples
```

2. Set mode = 1 to check the lengths of the data of a specific settings.

## Safety Fine-tuning
#### SFT Dataset
All SFT datasets generated using `./dataset/proc.py` are located in:
`./safety_train/data/sft`

### Train code
1. Set the training data path, model path, and save_path in:
`./safety_train/trl_train_code/run_sft.sh`
2. Launch training with:
``` bash
bash run_sft.sh
```


## ASR and Over-Refusal Evaluation 

This section details the evaluation process for the 3 tests for Safety Performance Evaluation (PAIR, PAP, None) and 1 test for Over-Refusal Evaluation (XSTest).

#### 1. PAIR Evaluation

To perform the PAIR evaluation:

First, set the model path and GPU ID you want to evaluate in `run_pair.sh` like:
```
model_path GPU_ID
"checkpoint_path 7"
```
Then, run the evaluation script.
``` bash
cd attack_refusal_evaluation
bash run_pair.sh
```

After the run completes, set the model path in `pair_res.sh`.
Finally, run the evaluation script to process the results. 
``` bash 
cd attack_refusal_evaluation
bash pair_res.sh
```

#### 2. PAP and None (No Attack) Test 
To run this two attack tests, simply set the model_paths and out_names in `gen_pap_no.sh` and then run it
``` bash 
cd attack_refusal_evaluation
bash gen_pap_no.sh
```

#### 3. XSTest Evaluation
To run the over-refusl evaluation, simply set the model_path and out_names in `run_xstest.sh`
then run it
``` bash 
cd attack_refusal_evaluation
bash run_xstest.sh
```

## Reasoning Performance Evaluation

This directory contains the evaluation scripts for three benchmarks used in the Reasoning Performance Evaluation: **MATH-500**, **AIME 2024**, and **LiveCodeBench**.

### 1. MATH-500 and AIME 2024

To evaluate on these two benchmarks:

1. Set the `model_name_or_paths` in `eval_math.sh`.
2. Run the following command:

```bash
cd reasoning_evaluation
bash eval_math.sh
```

### 2. LiveCodeBench
To evaluate on LiveCodeBench:
1. clone the dataset repository into the target directory:
```
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite reasoning_safety/reasoning_evaluation/LiveCodeBench
```
2. Set the GPUs, model_paths and model_names in `eval_code.sh`.
3. Run the evaluation script:
``` bash 
cd reasoning_evaluation
bash eval_code.sh
```

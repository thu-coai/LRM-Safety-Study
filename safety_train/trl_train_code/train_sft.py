import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../trl")))

from trl import SFTConfig, SFTTrainer
from transformers import HfArgumentParser
from datasets import load_dataset
from dataclasses import dataclass, field

@dataclass
class AdditionalArgs:
    dataset_path: str
    model_path: str


parser = HfArgumentParser((SFTConfig, AdditionalArgs))
training_args, addtional_args = parser.parse_args_into_dataclasses()
print(training_args)
print(addtional_args)
model_path = addtional_args.model_path
dataset = load_dataset(
    "json",
    data_files=addtional_args.dataset_path,
)["train"]

trainer = SFTTrainer(
    args=training_args,
    model=model_path,
    train_dataset=dataset,
)
trainer.train()

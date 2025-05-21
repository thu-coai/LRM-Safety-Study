import os
import sys


sys.path.append(os.getcwd())

from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments
from aisafetylab.logging import setup_logger
from aisafetylab.attack.attackers.pair import PAIRManager

args = parse_arguments()

setup_logger(log_file_path=None, stderr_level="DEBUG")

if args.config_path is None:
    args.config_path = "configs/pair_qwen_nocot.yaml"

config_manager = ConfigManager(config_path=args.config_path)
attack_manager = PAIRManager.from_config(config=config_manager.config)
attack_manager.attack()

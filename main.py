from huggingface_hub import login
from datasets import Dataset
from argparse import Namespace

from args import get_args
from data_processor import DataProcessor
from utils import init_env_args_logging

def run(args: Namespace):
    data_processor = DataProcessor(args)

if __name__ == "__main__":
    args = init_env_args_logging()
    run(args)

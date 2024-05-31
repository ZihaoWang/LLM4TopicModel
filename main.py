from huggingface_hub import login
from datasets import Dataset
from argparse import Namespace

from args import get_args
from data_processor import DataProcessor
from utils import *
from model import PPO_Model

def run(args: Namespace):
    data_processor = DataProcessor(args)

    train_dataset, seen_test_dataset, unseen_test_dataset, topic_words = \
            data_processor.get_datasets()

    model = PPO_Model(args, topic_words)
    model.train(train_dataset)

if __name__ == "__main__":
    args = init_env_args_logging()
    run(args)

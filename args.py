import argparse
from argparse import Namespace

def get_args() -> Namespace:
    '''
    CLI arguments in this model.

    Returns:
            args: all parsed arguments.
    '''
    parser = argparse.ArgumentParser(description = "Topic Modeling by fine-tuning LLMs")

    parser.add_argument("--log_root", default = "./log/", type = str, help = "Directory of logs")
    parser.add_argument("--data_root", default = "./data/", type = str, help = "Directory of data")
    parser.add_argument("--result_root", default = "./result/", type = str, help = "Directory of results")
    parser.add_argument("--tmp_root", default = "./tmp/", type = str, help = "Directory of temporary files.")

    parser.add_argument("--corpus_name", type = str, default = 'yahoo_answers_topics', help = "HuggingFace dataset of text corpus.")
    parser.add_argument("--corpus_sample_size", type = int, default = 3000, help = "")
    parser.add_argument("--num_corpus_topic", type = int, default = 10, help = "It is 10 from dataset info.")
    parser.add_argument("--num_seen_topic", type = int, default = 8, help = "")
    parser.add_argument("--finetune_size_per_label", type = int, default = 100, help = "Size of corpus per label, for fine tuning.")
    parser.add_argument("--test_size_per_label", type = int, default = 100, help = "Size of corpus per label, for testing.")
    parser.add_argument("--keyword_sample_size", type = int, default = 200, help = "")
    parser.add_argument("--num_generate_keywords", type = int, default = 5, help = "Number of generated keywords from LLM")

    parser.add_argument("--topic_emb_model", type = str, default = 'all-MiniLM-L6-v2', help = "HuggingFace embedding model for Bunka.")

    parser.add_argument("--llm_model", type = str, default = 'daryl149/llama-2-7b-chat-hf', choices = ['daryl149/llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct'], help = "HuggingFace embedding model for LLM.")
    parser.add_argument("--llm_lr", default = 1e-5, type = float, help = "learning rate")
    parser.add_argument("--ppo_epochs", default = 1, type = int, help = "number of PPO training epochs")
    parser.add_argument("--batch_size", default = 4, type = int, help = "batch size")
    parser.add_argument("--mini_batch_size", default = 1, type = int, help = "PPO minibatch size")
    parser.add_argument("--gradient_accumulation_steps", default = 1, type = int, help = "in ppo training")
    parser.add_argument("--early_stopping", default = False, type = bool, help = "in ppo training")
    parser.add_argument("--target_kl", default = 0.1, type = float, help = "target KL divergence for early stopping")

    # below are LORA specific arguments
    parser.add_argument("--lora_r", default = 16, type = int, help = "")
    parser.add_argument("--lora_alpha", default = 32, type = int, help = "")
    parser.add_argument("--lora_dropout", default = 0.05, type = float, help = "")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

import argparse

def get_args():
    parser = argparse.ArgumentParser(description = "Topic Modeling with Llama3")

    parser.add_argument("--log_root", default = "./log/", type = str, help = "Directory of logs")
    parser.add_argument("--data_root", default = "./data/", type = str, help = "Directory of data")
    parser.add_argument("--result_root", default = "./result/", type = str, help = "Directory of results")
    parser.add_argument("--tmp_root", default = "./tmp/", type = str, help = "Directory of temporary files.")

    parser.add_argument("--corpus_name", type = str, default = 'yahoo_answers_topics', help = "HuggingFace dataset of text corpus.")
    parser.add_argument("--num_corpus_label", type = int, default = 10, help = "It is 10 from dataset info.")
    parser.add_argument("--finetune_size_per_label", type = int, default = 200, help = "Size of corpus per label, for fine tuning.")
    parser.add_argument("--test_size_per_label", type = int, default = 20, help = "Size of corpus per label, for testing.")

    parser.add_argument("--parent_chunk_size", type = int, default = -1, choices = [-1, 1000], help = "")
    parser.add_argument("--score_threshold", default = "0.5", type = float, help = "")

    parser.add_argument("--idx_gpu", default = -1, type = int, help = "which cuda device to use (-1 for cpu training)")

    args = parser.parse_args()

    args.data_path = args.data_root + "scraped_data.jsonl"
    args.device = "cpu" if args.idx_gpu == -1 else f"cuda:{args.idx_gpu}"

    return args

if __name__ == "__main__":
    args = get_args()

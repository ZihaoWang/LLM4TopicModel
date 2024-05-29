from collections import defaultdict
from datasets import load_dataset, Dataset
from argparse import Namespace
from typing import List, Tuple
import os
import pickle
from args import get_args

class DataProcessor(object):

    def __init__(self, args: Namespace):
        self.args = args
        
        raw_data = load_dataset(self.args.corpus_name)
        raw_train = raw_data['train']
        raw_test = raw_data['test']

        #print(self.train_set.shape, self.train_set.column_names)
        #print(self.train_set[0])
        #print(self.test_set.shape, self.test_set.column_names)

        corpus_path = self.args.tmp_root + "corpus_sample.pkl"
        if os.path.exists(corpus_path):
            with open(corpus_path, "rb") as f_src:
                corpus_train, corpus_test = pickle.load(f_src)
        else:
            corpus_train, corpus_test = self.__sample_corpus(raw_train, raw_test)
            with open(corpus_path, "wb") as f_dst:
                pickle.dump((corpus_train, corpus_test), f_dst, pickle.HIGHEST_PROTOCOL)

        print(len(corpus_train), len(corpus_test))

    def __sample_corpus(self,
            raw_train: Dataset,
            raw_test: Dataset) -> Tuple[List[str]]:
        train_sample = defaultdict(list)
        test_sample = defaultdict(list)
        train_num, test_num = 0, 0
        max_train_num = self.args.num_corpus_label * self.args.finetune_size_per_label
        max_test_num = self.args.num_corpus_label * self.args.test_size_per_label 
        for i in range(raw_train.shape[0]):
            if train_num >= max_train_num:
                break
            data = raw_train[i]
            if len(train_sample[data['topic']]) < self.args.finetune_size_per_label:
                train_sample[data['topic']].append(data['best_answer'])
                train_num += 1

        for i in range(raw_test.shape[0]):
            if test_num >= max_test_num:
                break
            data = raw_test[i]
            if len(test_sample[data['topic']]) < self.args.test_size_per_label:
                test_sample[data['topic']].append(data['best_answer'])
                test_num += 1

        final_train, final_test = [], []
        for topic, text_list in train_sample.items():
            final_train += text_list
        for topic, text_list in test_sample.items():
            final_test += text_list
        final_corpus = (final_train, final_test)

        return final_corpus
            


if __name__ == "__main__":
    args = get_args()
    data_processor = DataProcessor(args)

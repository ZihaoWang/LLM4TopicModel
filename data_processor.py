import logging
from pandas import DataFrame
from langchain.docstore.document import Document
from collections import defaultdict, Counter
from datasets import load_dataset, Dataset
from argparse import Namespace
from typing import List, Tuple, Set, Dict
from bunkatopics import Bunka
from sentence_transformers import SentenceTransformer
import transformers
import os
import pickle
from args import get_args

class DataProcessor(object):
    '''
    The class for loading, transforming raw data, creating topic information and generating datasets.

            Methods:
                    get_datasets(): get training, seen testing and unseen testing datasets, and topic information.
    '''

    def __init__(self, args: Namespace):
        '''
        Constructor for DataProcessor.

                Parameters:
                        args: all the arguments.
        '''

        self.args = args
        
        yahoo_dataset = load_dataset(self.args.corpus_name)['train']
        raw_corpus = []
        for i in range(yahoo_dataset.shape[0]):
            if i >= self.args.corpus_sample_size:
                break
            data = yahoo_dataset[i]
            raw_corpus.append(data['best_answer'])
        
        corpus_with_topic, topic_info = self.__fit_topics(raw_corpus)
        seen_topic, unseen_topic = self.__split_topics(topic_info)

        #print(self.train_set.shape, self.train_set.column_names)
        #print(self.train_set[0])
        #print(self.test_set.shape, self.test_set.column_names)

        split_path = self.args.tmp_root + "split_sample.pkl"
        if os.path.exists(split_path):
            with open(split_path, "rb") as f_src:
                split_train, split_seen_test, split_unseen_test = pickle.load(f_src)
        else:
            split_train, split_seen_test, split_unseen_test = self.__sample_split(corpus_with_topic, topic_info, seen_topic, unseen_topic)
            with open(split_path, "wb") as f_dst:
                pickle.dump((split_train, split_seen_test, split_unseen_test), f_dst, pickle.HIGHEST_PROTOCOL)

        self.dataset_path = self.args.tmp_root + f"dataset_{self.args.llm_model}.pkl".replace('/', '_')
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "rb") as f_src:
                self.train_dataset, self.seen_test_dataset, self.unseen_test_dataset, self.topic_words = pickle.load(f_src)
            logging.info(f'load dataset from {self.dataset_path}')
        else:
            self.__generate_llm_dataset(split_train, split_seen_test, split_unseen_test, topic_info)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset, Dict[str, List[str]]]:
        '''
        Get training, seen testing and unseen testing datasets, and topic information.

                Parameters:
                        None

                Returns:
                        self.train_dataset: training dataset.
                        self.seen_test_dataset: testing dataset with seen topics in the training dataset.
                        self.unseen_test_dataset: testing dataset with seen topics in the training dataset.
                        self.topic_words: a dict of topic ids and corresponding topic words.
        '''
        return self.train_dataset, self.seen_test_dataset, self.unseen_test_dataset, self.topic_words

    def __fit_topics(self, raw_corpus: List[str]) -> Tuple[List[Document], DataFrame]:
        '''
        Unsupervisely extract topic information with Bunka.

                Parameters:
                        raw_corpus: a list of raw corpus.

                Returns:
                        corpus_with_topic: list of corpus processed by Bunka.
                        topic_info: a dict of topic ids and their topic words.
        '''
        embedding_model = SentenceTransformer(self.args.topic_emb_model)
        bunka_path = self.args.tmp_root + "bunka_dumps"
        bunka = Bunka(embedding_model=embedding_model)
        if os.path.exists(bunka_path):
            logging.info(f"load bunka from {bunka_path}")
            bunka = bunka.load_bunka(path=bunka_path)
        else:
            bunka.fit(raw_corpus)
            logging.info(f"save bunka to {bunka_path}")
            bunka.save_bunka(bunka_path)

        topic_info = bunka.get_topics(n_clusters = self.args.num_corpus_topic, name_length = 100)
        corpus_with_topic = bunka.docs
        #print(topic_info)
        #print(corpus_with_topic[0].topic_id)
        #print(corpus_with_topic[0].content)

        return corpus_with_topic, topic_info

    def __sample_split(self,
            corpus_with_topic: List[Document],
            topic_info: DataFrame,
            seen_topics: Set[str],
            unseen_topics: Set[str]) -> Tuple[Dict[str, List[str]]]:
        '''
        Create training and testing splits with seen and unseen topics.

                Parameters:
                        corpus_with_topic: list of corpus processed by Bunka.
                        topic_info: a dict of topic ids and their topic words.
                        seen_topics: seen topic ids from training.
                        unseen_topics: unseen topic ids from training.

                Returns:
                        split_train: training split.
                        split_seen_test: testing split with seen topics.
                        split_unseen_test: testing split with unseen topics.
        '''

        corpus_seen = defaultdict(list)
        corpus_unseen = defaultdict(list)
        for doc in corpus_with_topic:
            topic = doc.topic_id
            content = doc.content[:self.args.max_doc_length]
            if topic in seen_topics:
                corpus_seen[topic].append(content)
            else:
                corpus_unseen[topic].append(content)

        max_train_num = int(topic_info['size'].min() * 0.8)

        split_train = {}
        split_seen_test = {}
        split_unseen_test = {}

        for topic_id, contents in corpus_seen.items():
            split_train[topic_id] = contents[:max_train_num]
            split_seen_test[topic_id] = contents[max_train_num:]
        for topic_id, contents in corpus_unseen.items():
            split_unseen_test[topic_id] = contents

        return split_train, split_seen_test, split_unseen_test
            
    def __split_topics(self,
            topic_info: DataFrame) -> Tuple[Set[str], Set[str]]:
        '''
        Split topics into num_seen_topic topics in training and seen testing sets,
        and remaining topics belong to unseen testing sets.

                Parameters:
                        topic_info: topic information from Bunka.

                Returns:
                        seen_topic: seen topic ids.
                        unseen_topic: unseen topic ids.
        '''
        seen_topic = topic_info['topic_id'][:self.args.num_seen_topic]
        unseen_topic = topic_info['topic_id'][self.args.num_seen_topic:]
        seen_topic, unseen_topic = set(seen_topic), set(unseen_topic)

        return seen_topic, unseen_topic

    def __generate_llm_dataset(self,
            split_train: Dict[str, List[str]],
            split_seen_test: Dict[str, List[str]],
            split_unseen_test: Dict[str, List[str]],
            topic_info: DataFrame):
        '''
        Create LLM datasets using a prompt template.

                Parameters:
                        split_train: training split.
                        split_seen_test: testing split with seen topics.
                        split_unseen_test: testing split with unseen topics.
                        topic_info: topic information from Bunka.

                Returns:
                        None
        '''

        prompt = '''You are a helpful assistant that must try your best effort to summarize {} keywords representing main topic of the CONTENT.
            Your output always starts with "KEYWORDS:", then you should separate each generated keyword with a comma ",".
            Remember, only generate at most {} keywords.
            Next, you always output "####", and then you summarize these generated keywords with a topic name.
            This topic name should be one or two representative and meaningful words, do not generate more than two words.
            Then terminate your generation, you should never generate any other words.
            CONTENT: {}'''

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.llm_model, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        def transform_dataset(split: Dict[str, List[str]]) -> (Dataset, int):
            dataset = defaultdict(list)
            for tid, contents in split.items():
                for content in contents:
                    query = prompt.format(
                            self.args.num_generate_keywords,
                            self.args.num_generate_keywords,
                            content
                            )
                    input_ids = tokenizer.encode(query)
                    dataset['input_ids'].append(input_ids) 
                    dataset['topic_id'].append(tid)
                    dataset['query'].append(tokenizer.decode(input_ids))
            
            dataset = Dataset.from_dict(dataset)
            dataset.set_format(type='torch')
            return dataset

        self.topic_words = {}
        for tid, twords in zip(topic_info['topic_id'], topic_info['topic_name']):
            twords = twords.lower().replace(' | ', '|').split('|')
            self.topic_words[tid] = twords

        self.train_dataset = transform_dataset(split_train)
        self.seen_test_dataset = transform_dataset(split_seen_test)
        self.unseen_test_dataset = transform_dataset(split_unseen_test)

        with open(self.dataset_path, "wb") as f_dst:
            dataset = (self.train_dataset, self.seen_test_dataset, self.unseen_test_dataset, self.topic_words)
            pickle.dump(dataset, f_dst, pickle.HIGHEST_PROTOCOL)
        logging.info(f'save dataset to {self.dataset_path}')



if __name__ == "__main__":
    args = get_args()
    data_processor = DataProcessor(args)

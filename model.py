from argparse import Namespace
import random
from typing import List, Tuple, Set, Dict
import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trl import (
        PPOTrainer, PPOConfig,
        AutoModelForCausalLMWithValueHead
        )
from transformers import (
        pipeline, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments
        )
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from peft import LoraConfig
from accelerate import Accelerator

class PPO_Model(object):
    '''
    Our model for fine tuning the LLM for topic modeling.

            Methods:
                    train(train_dataset): for model training.
                    test(test_dataset): for model evaluation.
    '''

    def __init__(self,
            args: Namespace,
            topic_words: Dict[str, List[str]]):
        '''
        Constructor for PPO_Model.

                Parameters:
                        args: all the arguments.
                        topic_words: a dict, keys are topic ids and
                        values are lists of corresponding topic words.
        '''
        self.args = args
        self.topic_words = topic_words

        self.ppo_config = PPOConfig(
                model_name=self.args.llm_model,
                learning_rate=self.args.llm_lr,
                ppo_epochs=self.args.ppo_epochs,
                batch_size=self.args.batch_size,
                mini_batch_size=self.args.mini_batch_size,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                early_stopping=self.args.early_stopping,
                target_kl=self.args.target_kl,
                optimize_cuda_cache=True,
                remove_unused_columns=False
                )

        self.lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM'
                )

        self.current_device = Accelerator().local_process_index
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_model, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding_model = SentenceTransformer(self.args.topic_emb_model)
        self.ppo_saving_path = self.args.result_root + 'ppo_trainer_' + self.args.llm_model.replace('/', '_')

        self.generate_kwargs = {
                'max_length': 2048, # max length of output
                'do_sample': True, # non-greedy decoding
                'pad_token_id': self.tokenizer.pad_token_id,
                'temperature': 1.0
                }
        
    def train(self, train_dataset: Dataset):
        '''
        For fine-tuning the LLM with the PPO algorithm.

                Parameters:
                        train_dataset: the training dataset.

                Returns:
                        None
        '''
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.ppo_config.model_name,
                #load_in_8bit=True,
                load_in_4bit=True,
                device_map={'': self.current_device},
                peft_config=self.lora_config)

        self.ppo_trainer = PPOTrainer(self.ppo_config,
                llm_model,
                tokenizer=self.tokenizer,
                dataset=train_dataset,
                data_collator=collator,)

        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]

            response_tensors = self.ppo_trainer.generate(query_tensors,
                    return_prompt=False,
                    **self.generate_kwargs)
            batch_response = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch_response, topic_response = self.__postprocess(batch_response)
           
            reward = self.__get_reward(batch_response, batch['topic_id'])
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward)
            self.ppo_trainer.log_stats(stats, batch, reward)

        self.ppo_trainer.save_pretrained(self.ppo_saving_path)
        logging.info(f'saving ppo trainer at {self.ppo_saving_path}')

    def test(self, test_dataset: Dataset):
        '''
        For evaluating the fine-tuned LLM.

                Parameters:
                        test_dataset: can be testing datasets with seen or unseen topics.

                Returns:
                        None
        '''
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.ppo_saving_path,
                load_in_8bit=True,
                device_map={'': self.current_device},
                peft_config=self.lora_config)

        self.ppo_trainer = PPOTrainer(self.ppo_config,
                llm_model,
                tokenizer=self.tokenizer,
                dataset=test_dataset,
                data_collator=collator)

        all_reward = []
        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]

            response_tensors = self.ppo_trainer.generate(query_tensors)
            batch_response = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch_response, topic_response = self.__postprocess(batch_response)
            
            reward = self.__get_reward(batch_response, batch['topic_id'])
            all_reward += reward.tolist()

            for q, r, t in zip(batch['query'], batch_response, topic_response):
                logging.info('INPUT QUERY: ' + q)
                logging.info('**********************')
                logging.info('OUTPUT KEYWORDS: ' + r)
                logging.info('**********************')
                logging.info('OUTPUT TOPIC: ' + t)
                logging.info('\n---------------------------\n')

        avg_reward = T.mean(T.tensor(all_reward))
        logging.info(f"Average reward = {avg_reward}")
 
    def __get_reward(self,
            batch_response: List[str],
            topic_ids: List[str]) -> List[T.Tensor]:
        '''
        Compute the batch rewards for PPO model,
        currently we only embed the responses and golden topic words into two embeddings
        and compute the cosine similarity of them as the reward.
        More fine-grained reward models can be considered.

                Parameters:
                        batch_response: clean LLM responses after postprocessing.
                        topic_ids: a dict of topic ids and lists of corresponding topic words.

                Returns:
                        all_reward: a list of rewards, for PPO updating.
        '''
        targets = []
        for tid in topic_ids:
            target = random.sample(self.topic_words[tid], self.args.num_generate_keywords)
            targets.append(','.join(target))

        emb_responses = T.tensor(self.embedding_model.encode(batch_response))
        emb_targets = T.tensor(self.embedding_model.encode(targets))
        assert emb_responses.shape == emb_targets.shape

        all_reward = []
        for i in range(emb_responses.shape[0]):
            reward = F.cosine_similarity(
                    emb_responses[i].unsqueeze(0),
                    emb_targets[i].unsqueeze(0))
            all_reward.append(reward)

        return all_reward

    def __postprocess(self,
            batch_response: List[str]) -> Tuple[List[str], str]:
        '''
        Process the format of generated texts from the LLM.

                Parameters:
                        batch_response: a list of responses.

                Returns:
                        clean_responses: the processed responses, containing
                        at most num_generate_keywords keywords.
                        topics: the representative topic name for these keywords.
        '''
        clean_responses = []
        topics = []
        for res in batch_response:
            res = res.strip().lower().split('keywords:')
            if len(res) != 2:
                clean_responses.append(self.tokenizer.eos_token)
                topics.append(self.tokenizer.eos_token)
                continue
            res = res[1].split('####')
            if len(res) != 2:
                clean_responses.append(self.tokenizer.eos_token)
                topics.append(self.tokenizer.eos_token)
                continue

            keywords, topic = res
            if keywords[-1] == '.':
                keywords = keywords[:-1]
            keywords = keywords.split(',')[:self.args.num_generate_keywords]
            keywords = ','.join(keywords)
            clean_responses.append(keywords)

            topics.append(topic.strip())

        return clean_responses, topics








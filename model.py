from argparse import Namespace
import random
from typing import List, Tuple, Set, Dict
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
    def __init__(self,
            args: Namespace,
            topic_words: Dict[str, List[str]]):
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
                optimize_cuda_cache=True
                )

        self.lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM'
                )

        self.current_device = Accelerator().local_process_index
        self.llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.ppo_config.model_name,
                load_in_8bit=True,
                device_map={'': self.current_device},
                peft_config=self.lora_config)

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

        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        self.ppo_trainer = PPOTrainer(self.ppo_config,
                self.llm_model,
                tokenizer=self.tokenizer,
                #dataloader=train_loader)
                dataset=train_dataset,
                data_collator=collator,)

        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            response_tensors = []

            response_tensors = self.ppo_trainer.generate(query_tensors,
                    return_prompt=False,
                    **self.generate_kwargs)
            batch_response = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            '''
            for q, r in zip(batch['query'], batch_response):
                print(q)
                print('**********************')
                print(r)
                print('\n---------------------------\n')
            '''
            
            self.__get_reward(batch_response, batch['topic_id'])
            reward = None
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward)
            self.ppo_trainer.log_stats(stats, batch, rewards)

        self.ppo_trainer.save_pretrained(self.ppo_saving_path)
        logging.info(f'saving ppo trainer at {self.ppo_saving_path}')

    def predict(self, test_dataset: Dataset):

        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size)
        self.ppo_trainer.save_pretrained(self.ppo_saving_path)
        self.ppo_trainer = PPOTrainer(self.ppo_config,
                self.llm_model,
                tokenizer=self.tokenizer,
                #dataloader=train_loader)
                dataset=train_dataset,
                data_collator=collator,)

        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            response_tensors = []
            #print(len(query_tensors), query_tensors)

            response_tensors = self.ppo_trainer.generate(query_tensors)#, max_new_tokens = self.args.output_max_len)
            batch_response = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            #print(len(batch_response), batch_response)
            
            reward = self.__get_reward(batch_response, batch['topic_id'])
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward)
            self.ppo_trainer.log_stats(stats, batch, rewards)

        self.ppo_trainer.save_pretrained(self.ppo_saving_path)
        logging.info(f'saving ppo trainer at {self.ppo_saving_path}')

    def __get_reward(self, responses, topic_ids):
        responses = self.__postprocess(responses)
        for e in responses:
            print(e + '\n\n')

        targets = []
        for tid in topic_ids:
            target = random.sample(self.topic_words[topic_ids], self.args.num_generate_keywords)
            targets.append(','.join(target))

        emb_responses = self.embedding_model.encode(responses)
        emb_targets = T.tensor(self.embedding_model.encode(targets))
        print(emb_responses.shape, emb_targets.shape)
        reward = F.cosine_similarity(emb_responses, emb_targets, -1)
        print(reward)
        exit()

        return reward


    def __postprocess(self, responses):
        clean_responses = []
        for res in responses:
            res = res.strip().lower().split('keywords:')
            if len(res) != 2:
                clean_responses.append(None)
                continue
            res = res[1]
            if res[-1] == '.':
                res = res[:-1]
            res = res.split(',')[:self.args.num_generate_keywords]
            res = ','.join(res)
            clean_responses.append(res)

        return clean_responses








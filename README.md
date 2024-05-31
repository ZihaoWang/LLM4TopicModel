# SUMM AI NLP Challenge - Topic Modeling with LLM
In this project, we fine-tune an open-source Large Language Model (LLM) to perform topic modeling on a collection of documents.

Specifically, after fine-tuning, the LLM can DYNAMICALLY generate a meaningful topic name and several keywords for a given text. There are no sets of pre-defined topic names or keywords.

# Installation and running

1. Build the Docker image with the Linux CLI:

    * docker build -t ppo_img --rm .

    * docker run -it --name ppo_app --rm ppo_img

2. Now you can run the model within the CLI interface of Docker:

    * python main.py

Or give some CLI arguments:

    * python main.py --num_generate_keywords 10 --ppo_epochs 2 

You can find all arguments and their descriptions in the args.py.

# Dataset

There is no suitable datasets with dynamic topic names and keywords online, so we create a dataset from scratch.

1. We use the [Yahoo Answers Topics](https://huggingface.co/datasets/yahoo_answers_topics) dataset, which is constructed using 10 largest main categories. Each class contains 140,000 training samples and 6,000 testing samples. From each sample, we use the *topic id* and *best answer* fields as categories and documents. We get --corpus_sample_size samples and create three datasets:

    * Training dataset: each document belongs to one of --num_seen_topic *seen topics*, and there are --finetune_size_per_label documents per seen topic.

    * Seen testing dataset: contains remaining document belonging to one of --num_seen_topic *seen topics*.

    * Unseen testing dataset: contains all documents belonging to the remaining *unseen topics*.

In this way, we can evaluate the performance of our fine-tuned LLM on both seen documents of topics and unseen topics. The number of seen topics is 8 by default.

2. In order to generate dynamic topic names and keywords, we do not use the topic id of each document as the supervision signal. Instead, we use a topic model [Bunka](https://github.com/charlesdedampierre/BunkaTopics) to generate a list of topic words from sampled documents. Here is the topic information from Bunka:

|topic_id  |topic_words                                     |size  |percent|
|----------|------------------------------------------------|------|-------|
|bt-7      |computer, software, desktop, screen, brows...   |396   | 13.20 |
|bt-8      |money, schools, investment, grades, grade ...   |326   | 10.87 |
|bt-5      |electrons, temperature, gases, ratio, shell...  |325   | 10.83 |
|bt-6      |team, players, player, ball, game, teams ...    |312   | 10.40 |
|bt-9      |evolution, religions, faith, belief, lie...     |308   | 10.27 |
|bt-4      |war, government, country, language, law ...     |292   |  9.73 |
|bt-1      |vote, com, www, index, itools, collection...    |271   |  9.03 |
|bt-0      |episode, rock, lines, scene, episodes...        |268   |  8.93 |
|bt-2      |doctor, body, blood, therapy, symptoms...       |268   |  8.93 |
|bt-3      |relationship, friends, relationships, thing...  |234   |  7.80 |


# Fine tuning of the LLM

## Proximal Policy Optimization (PPO)

## Rewarding Model

# Evaluation

# Future improvements

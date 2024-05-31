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

You can find all the arguments and their descriptions in args.py.

# Dataset

There are no suitable datasets with dynamic topic names and keywords online, so we create a dataset from scratch.

1. We use the [Yahoo Answers Topics](https://huggingface.co/datasets/yahoo_answers_topics) dataset, which is constructed using 10 largest main categories. Each class contains 140,000 training samples and 6,000 testing samples. From each sample, we use the **topic id** and **best answer** fields as categories and documents. We get --corpus_sample_size samples and create three datasets:

    * Training dataset: Each document belongs to one of --num_seen_topic **seen topics**, and there are --finetune_size_per_label documents per seen topic.

    * Seen testing dataset: It contains remaining document belonging to one of --num_seen_topic **seen topics**.

    * Unseen testing dataset: It contains all documents belonging to the remaining **unseen topics**.

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

Please note that topic ids are from original Yahoo Answers Topics and have no specific meanings.

Here, we pass an embedding model all-MiniLM-L6-V2 to Bunka for topic modeling. We will use the same embedding model in the following PPO fine-tuning.

3. Then, we insert all documents in the following prompt template prompt and tokenize it. Now the datasets for fine-tuning an LLM are ready!

*You are a helpful assistant that must try your best effort to summarize {--num_generate_keywords} keywords representing main topic of the CONTENT.
Your output always starts with "KEYWORDS:", then you should separate each generated keyword with a comma ",".
Remember, only generate at most {--num_generate_keywords} keywords.
Next, you always output "####", and then you summarize these generated keywords with a topic name.
This topic name should be one or two representative and meaningful words, do not generate more than two words.
Then terminate your generation, you should never generate any other words.
CONTENT: {document}*


# Fine tuning of the LLM
Instead of supervised fine-tuning, we use Proximal Policy Optimization (PPO) to fine-tune our LLM as we do not have direct supervision signals. The PPO fine-tuning process can be described as:

1. Given each tokenized input in step 3 above, the LLM generates a response including a topic name and several **response keywords**.

2. From topic words corresponding to the topic id of the document in step 2 above, we sample several **targeting keywords**.

3. We create a simple reward model by first encoding both **response keywords** and **targeting keywords** into embeddings with the same embedding model all-MiniLM-L6-V2 in the Bunka, and then we compute the cosine similarity of them as the reward.

4. The **response keywords**, **targeting keywords**, and rewards are used in the PPO update process.

During the PPO fine-tuning, I used QLORA and quantized the LLM into 4 bits.

# Evaluation

We evaluate on both seen and unseen testing dataset to check the performance of our PPO fine-tuning.

# Future improvements

1. The format of LLM-generated responses is sometimes unstable (10% - 20%). For example, it continually generates subsequent words of the input prompt before useful keywords and topic names. Create a labeled dataset with correct formats and perform supervised fine-tuning before PPO can solve this issue.

2. The simple cosine rewarding model can be replaced with another deep network or LLM. Train the rewarding model before the PPO fine-tuning can obtain better performances.

3. According to recent information on how Meta pre-trained Llama3, we can further fine-tune the LLM with the Direct Performance Optimization (DPO) after PPO to improve the performance.

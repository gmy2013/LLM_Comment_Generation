from cbleu import *
from rouge import Rouge
import os
import openai
import json
import time
import random
import nltk
import re
import javalang
import numpy as np
import time
import random
import heapq
import argparse
from collections import defaultdict
from multiprocessing import Pool
import heapq
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from collections import Counter
openai.api_key = "your openai api key"

example_code = """
public synchronized boolean start() {
    if (!isStarted()) {
    final ExecutorService tempExec;
    executor = getExternalExecutor();
    if (executor == null) {
        executor = tempExec = createExecutor();
    } 
    else {
        tempExec = null;}
        future = executor.submit(createTask(tempExec));
        return true;
    }
    return false;
}
"""

exampler_what = "You are an expert Java programmer, please describe the functionality of the method:\n\"\"\"" + "Example Code1:\n" + example_code + "The comment is: Starts the background initialization"
exampler_why = "# You are an expert Java programmer, please explain the reason why the method is provided or the design rational of the method:\n\"\"\"" #+ example_code + "The comment is: With this method the initializer becomes active and invokes the initialize() method in a background task"
exampler_use = "# You are an expert Java programmer, please describe the usage or the expected set-up of using the method:\n\"\"\"" #+ example_code + "The comment is: After the construction of a BackgroundInitializer() object it start() method has to be called"
exampler_done = "# You are an expert Java programmer, please describe the implementation details of the method:\n\"\"\"" #+ example_code + "The comment is: Get an external executor to create a background task. If there is not any, it creates a new one"
exampler_property = "# You are an expert Java programmer, please describe the asserts properties of the method including pre-conditions or post-conditions of the method:\n\"\"\"" #+ example_code + "The comment is: Return the flag whether the initializer could be started successfully"
zero_what =  "You are an expert Java programmer, please describe the functionality of the method:\n\"\"\""
zero_property = "# You are an expert Java programmer, please describe the asserts properties of the method including pre-conditions or post-conditions of the method:\n\"\"\""
zero_why = "# You are an expert Java programmer, please explain the reason why the method is provided or the design rational of the method:\n\"\"\""
zero_use = "# You are an expert Java programmer, please describe the usage or the expected set-up of using the method:\n\"\"\""
zero_done = "# You are an expert Java programmer, please describe the implementation details of the method:\n\"\"\""
model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError,),
):
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def tokenize(code_str):
    code_str = str(code_str)
    code_str = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_str)
    code_str = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_str)
    code_str = re.sub(r'\s+', ' ', code_str)
    tokens = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+|[^\w\s]+', code_str)
    for i in range(len(tokens)):
        if i > 0 and tokens[i-1].islower() and tokens[i].isupper():
            tokens[i] = tokens[i].lower()
    return tokens

def count_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return len(common_elements)

def cal_similarity_token(code1, code2):
    list1, list2 = tokenize(code1), tokenize(code2)
    return count_common_elements(list1, list2)


def read_data(file_address):
    ids = []
    codes = []
    comments = []
    labels = []
    with open(file_address, "r") as file:
        lines = file.readlines()
    for line in lines:
        sample = json.loads(line)
        ids.append(sample.get("id"))
        codes.append(sample.get("raw_code"))
        comments.append(sample.get("comment"))
        labels.append(sample.get("label"))
    print('Number of samples is ' + str(len(ids)))
    print('Number of codes is ' + str(len(codes)))
    print('Number of comments is ' + str(len(comments)))
    print('Number of labels is ' + str(len(labels)))
    return ids, codes, comments, labels

def test_single_code(test_code):
    new_prompt = property_prompt + "\nFor the test code:\n" + test_code + " The comment is: "
    for i in range(100):
      response = completion_with_backoff(model="code-davinci-002", prompt=new_prompt, max_tokens=30)
      cur_ans = response["choices"][0]["text"].split('\n')[0]
      print(cur_ans)

def test_sample_retrieve(test_codes, test_labels, training_codes, training_comments, category, pattern):
    if pattern == 'semantic':
        sim_file = 'sim_semantic.txt'
    else:
        sim_file = 'sim_token.txt'
    with open('ans.txt', 'w') as fp, open(sim_file, 'r') as fpp:
        lines = fpp.readlines()
        for i in range(len(test_codes)):
            test_code = test_codes[i]
            if test_labels[i] != category:
                continue
            sim_ids = lines[i].split(" ")
            prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done,
                            'property': zero_property}
            new_prompt = prompt_lists.get(category, zero_what)
            for i in range(10):
                new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(sim_ids[i])])
                new_prompt += ("\n# The comment is: " + training_comments[int(sim_ids[i])])
            new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
            specified_times = 100
            for j in range(specified_times):
                try:
                    response = completion_with_backoff(model="code-davinci-002", prompt=new_prompt, max_tokens=30)
                    cur_ans = response["choices"][0]["text"].split('\n')[0]
                    fp.write(cur_ans + '\n')
                except:
                    continue

def test_sample_rerank(test_codes, test_labels, training_codes, training_comments, category, pattern_rerank):
    with open('ans.txt', 'w') as fp, open('sim_semantic.txt', 'r') as fpp:
        lines = fpp.readlines()
        for i in range(len(test_codes)):
            test_code = test_codes[i]
            if test_labels[i] != category:
                continue
            prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done,
                            'property': zero_property}
            new_prompt = prompt_lists.get(category, zero_what)
            sim_ids, random_ids = lines[i].split(" "), []
            val = 4236  # len for the retrieval set
            ### Generate 10 random values
            for k in range(10):
                random_ids.append(random.randint(1, val))
            for i in range(10):
                new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(random_ids[i])])
                new_prompt += ("\n# The comment is: " + training_comments[int(random_ids[i])])
            new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
            specified_times, maxx, ans = 100, -1, ""
            code_tokens = tokenize(training_comments[int(sim_ids[0])])
            train_code_embedding = model.encode(training_comments[int(sim_ids[0])], convert_to_tensor=True)
            for j in range(specified_times):
                try:
                    response = completion_with_backoff(model="code-davinci-002", prompt=new_prompt, max_tokens=30)
                    cur_ans = response["choices"][0]["text"].split('\n')[0]
                    if pattern_rerank == 'token':
                       comment_list = tokenize(cur_ans)
                       intersect = count_common_elements(code_tokens, comment_list)
                       if intersect > maxx:
                           maxx = intersect
                           ans = cur_ans
                    else:
                        test_code_embedding = model.encode(cur_ans, convert_to_tensor=True)
                        hits = util.semantic_search(test_code_embedding, train_code_embedding)[0]
                        top_hit = hits[0]
                        intersect = top_hit['score']
                        if intersect > maxx:
                            maxx = intersect
                            ans = cur_ans
                except:
                    continue
            fp.write(ans + '\n')

def test_sample_retrieve_rerank(test_codes, test_labels, training_codes, training_comments, category, pattern_retrieve, pattern_rerank):
    if pattern_retrieve == 'semantic':
        file_sim = 'sim_semantic.txt'
    else:
        file_sim = 'sim_token.txt'
    with open('ans.txt', 'w') as fp, open(file_sim, 'r') as fpp:
        lines = fpp.readlines()
        for i in range(len(test_codes)):
            test_code = test_codes[i]
            if test_labels[i] != category:
                continue

            prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done,
                            'property': zero_property}
            new_prompt = prompt_lists.get(category, zero_what)
            sim_ids, random_ids = lines[i].split(" "), []
            for i in range(10):
                new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(sim_ids[i])])
                new_prompt += ("\n# The comment is: " + training_comments[int(sim_ids[i])])
            new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
            specified_times, maxx, ans = 100, -1, ""
            code_tokens = tokenize(training_comments[int(sim_ids[0])])
            train_code_embedding = model.encode(training_comments[int(sim_ids[0])], convert_to_tensor=True)
            for j in range(specified_times):
                try:
                    response = completion_with_backoff(model="code-davinci-002", prompt=new_prompt, max_tokens=30)
                    cur_ans = response["choices"][0]["text"].split('\n')[0]
                    if pattern_rerank == 'token':
                       comment_list = tokenize(cur_ans)
                       intersect = count_common_elements(code_tokens, comment_list)
                       if intersect > maxx:
                           maxx = intersect
                           ans = cur_ans
                    else:
                        test_code_embedding = model.encode(cur_ans, convert_to_tensor=True)
                        hits = util.semantic_search(test_code_embedding, train_code_embedding)[0]
                        top_hit = hits[0]
                        intersect = top_hit['score']
                        if intersect > maxx:
                            maxx = intersect
                            ans = cur_ans
                except:
                    continue
            fp.write(ans + '\n')

def test_sample_random(test_codes, test_labels, training_codes, training_comments, category):
  with open('ans.txt', 'w') as fp:
    for i in range(len(test_codes)):
      test_code = test_codes[i]
      if test_labels[i] != category:
          continue
      random_ids =  []
      val = 4236 # len for the retrieval set
      ### Generate 10 random values
      for k in range(10):
          random_ids.append(random.randint(1, val))
      prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done, 'property': zero_property}
      new_prompt = prompt_lists.get(category, zero_what)
      for i in range(10):
          new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(random_ids[i])])
          new_prompt += ("\n# The comment is: " + training_comments[int(random_ids[i])])
      new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
      #print(new_prompt)
      specified_times = 100
      for j in range(specified_times):
        try:
           response = completion_with_backoff(model="code-davinci-002", prompt=new_prompt, max_tokens=30)
           cur_ans = response["choices"][0]["text"].split('\n')[0]
           fp.write(cur_ans + '\n')
        except:
           continue

def get_data():
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = [], [], [], [], [], []
    train_file_address = 'tlcodesum.test'
    cur_ids, cur_codes, cur_comments, cur_labels = read_data(train_file_address)
    training_codes += cur_codes
    training_comments += cur_comments
    training_labels += cur_labels
    test_file_address = 'tlcodesum.train'
    cur_ids, cur_codes, cur_comments, cur_labels = read_data(test_file_address)
    test_codes += cur_codes
    test_comments += cur_comments
    test_labels += cur_labels
    return training_codes, training_comments, training_labels, test_codes, test_comments, test_labels

def find_similar_codes(training_code_list, test_code_list, training_label, test_label):
    training_dict = defaultdict(list)
    for i, (code, label) in enumerate(zip(training_code_list, training_label)):
        training_dict[label].append((code, i))

    result = []
    for test_code, test_label in zip(test_code_list, test_label):
        training_codes = training_dict[test_label]
        similarities = [cal_similarity_token(test_code, code) for code, _ in training_codes]
        top_indices = heapq.nlargest(10, range(len(similarities)), similarities.__getitem__)
        top_ids = [training_codes[i][1] for i in top_indices]
        result.append(top_ids)

    return result


def test_meteor(hypothesis, reference):
    import nltk
    from nltk.translate.meteor_score import meteor_score
    meteor = meteor_score(reference, hypothesis)
    return meteor


def generate_samples():
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    test_sample(test_codes)

def compare_ans(category):
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    ans_comments = []
    for i in range(len(test_comments)):
        if test_labels[i] == category:
            ans_comments.append(test_comments[i])
    with open('ans.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    return ans_comments, lines, test_labels

def code_sim_cal(code1, code2):
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    code1_emb = model.encode(code1, convert_to_tensor=True)
    code2_emb = model.encode(code2, convert_to_tensor=True)
    hits = util.semantic_search(code1_emb, code2_emb)[0]
    top_hit = hits[0]
    sim_score = top_hit['score']
    return sim_score

def test_metric(references, candidates):
    sum_bleu, sum_rouge, sum_meteor = 0, 0, 0
    cnt = 0
    rouge_score = Rouge()
    for i in range(len(references)):
        candidate = candidates[i]
        reference = references[i]
        cnt += 1
        sum_bleu += (nltk_sentence_bleu(candidate, reference) * 100)
        sum_rouge += (rouge_score.calc_score(candidate, reference) * 100)
        sum_meteor += (test_meteor(candidate, [reference]) * 100)
    print(sum_bleu / cnt)
    print(sum_rouge / cnt)
    print(sum_meteor / cnt)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--mode", choices=['rerank', 'retrieve', 'random'], required=True, help="Mode of operation.")

    parser = argparse.ArgumentParser(description="Process some operations.")
    parser.add_argument("--rerank", choices=['semantic', 'token', 'false'], required=True, help="Mode of rerank.")
    parser.add_argument("--retrieve", choices=['semantic', 'token', 'false'], required=True, help="Mode of retrieve.")
    parser.add_argument("--random", action='store_true', help="If present, random mode is active.")
    args = parser.parse_args()
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    categories = ['what', 'why', 'use', 'done', 'property']
    if args.random:
        for category in categories:
            test_sample_random(test_codes, test_labels, training_codes, training_comments, category)
    if args.retrieve != 'false' and args.rerank == 'false':
        for category in categories:
            test_sample_retrieve(test_codes, test_labels, training_codes, training_comments, category, args.retrieve)
    if args.retrieve == 'false' and args.rerank != 'false':
        for category in categories:
            test_sample_rerank(test_codes, test_labels, training_codes, training_comments, category, args.rerank)

    for category in categories:
        references, candidates, test_labels = compare_ans(category)
        test_metric(references, candidates)



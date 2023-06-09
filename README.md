<!-- # LLM_Comment_Generation
Large Language Models are Few-Shot Summarizers:\\
Multi-Intent Comment Generation via In-Context Learning

Our paper is available on -->


# Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning (ICSE 2024)

Our paper is available on [arxiv](https://arxiv.org/pdf/2304.11384.pdf).


## Abstract

Code comment generation aims at generating natural language descriptions for a code snippet to facilitate developers' program comprehension activities.
Despite being studied for a long time, a bottleneck for existing approaches is that given a code snippet, they can only generate one comment while developers usually need to know information from diverse perspectives such as what is the functionality of this code snippet and how to use it.
To tackle this limitation, this study empirically investigates the feasibility of utilizing large language models (LLMs) to generate comments that can fulfill developers' diverse intents.
Our intuition is based on the facts that (1) the code and its pairwise comment are used during the pre-training process of LLMs to build the semantic connection between the natural language and programming language, and (2) comments in the real-world projects, which are collected for the pre-training, usually contain different developers' intents.
We thus postulate that the LLMs can already understand the code from different perspectives after the pre-training.
Indeed, experiments on two large-scale datasets demonstrate the rationale of our insights: by adopting the in-context learning paradigm and giving adequate prompts to the LLM (\eg providing it with ten or more examples), the LLM can significantly outperform a state-of-the-art supervised learning approach on generating comments with multiple intents.
Results also show that customized strategies for constructing the prompts and post-processing strategies for reranking the results can both boost the LLM's performances, which shed light on future research directions for using LLMs to achieve comment generation.
## Get Started
OS: Ubuntu 18.04.  
Hardwares: Hygon C86 7385 32-core CPU 2.50GHz machine.  
The dataset we used are funcom.test, funcom.train, tlcodesum.test and tlcodesum.train.  
The comment categories are "what", "why", "how-to-use", "how-it-is-done", "property".  
Example:
```json
{
"id": "53306",
"raw_code": "public static int unionSize(long[] x,long[] y){\n final int lx=x.length, ly=y.length;\n final int min=(lx < ly) ? lx : ly;\n int i=0, res=0;\n for (; i < min; i++) {\n res+=Long.bitCount(x[i] | y[i]);\n }\n for (; i < lx; i++) {\n res+=Long.bitCount(x[i]);\n }\n for (; i < ly; i++) {\n res+=Long.bitCount(y[i]);\n }\n return res;\n}",
"comment": "compute the union size of two bitsets .",
"label": "what"
}
```
#### 1. Pre-process

First, preprocess the dataset to get the similarity scores for each code in the test set.
```
python preprocess.py
```
Then, you will get the "sim_token.txt" and "sim_semantic.txt", representing the similarities based on token and semantic respectively.

#### 2. Generate the Comments using Codex
You can generate the comment for different intents following the corresponding prompt using "random", "retrieve", "rerank" settings.
The default number of demonstrations is set to 10 in our code. 
For the retrieve and rerank settings, you can choose ['false', 'semantic', 'token'].
For example, to run the random setting:
```
python test_codex.py --random
```
To run the semantic-based retrieve setting:
```
python test_codex.py --rerank false --retrieve semantic
```
To run the token-based rerank setting:
```
python test_codex.py --rerank token --retrieve false
```
To run the token-based rerank and semantic-based retrieve setting:
```
python test_codex.py --rerank token --retrieve semantic
```

## Citation
```
@article{geng2023empirical,
  title={An Empirical Study on Using Large Language Models for Multi-Intent Comment Generation},
  author={Geng, Mingyang and Wang, Shangwen and Dong, Dezun and Wang, Haotian and Li, Ge and Jin, Zhi and Mao, Xiaoguang and Liao, Xiangke},
  journal={arXiv preprint arXiv:2304.11384},
  year={2023}
}
```





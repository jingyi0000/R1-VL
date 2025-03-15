<div align="center">

<h1> R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

<h5 align="center"> 

<a href='#'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='#'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='#'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>

[Jingyi Zhang]()<sup>1</sup>,
[Jiaxing Huang](https://jxhuang0508.github.io/)<sup>1</sup>,
[Huanjin Yao](https://scholar.google.com/citations?user=pDtsCBQAAAAJ&hl=zh-CN)<sup>2</sup>,
[Shunyu Liu]()<sup>1</sup>,
[Xikun Zhang]()<sup>1</sup>,
[Shijian Lu]()<sup>1</sup>,
[Dacheng Tao]()<sup>1</sup>


<sup>1</sup>[Nanyang Technological University](https://www.ntu.edu.sg/), <sup>2</sup>[Tsinghua University](https://www.tsinghua.edu.cn/en/)


</h5>
</div>

## News
- The paper, code, and models will be released soon.

## Abstract
Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are.
In this work, we aim to enhance the MLLMs’ reasoning ability beyond passively imitating positive reasoning paths. 
To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding.
Specifically, StepGRPO introduces two novel rule-based reasoning rewards:
Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR).
StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy.
With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning.


# LLM Experiments

This repository delves into the world of Large Language Models (LLMs) by showcasing a collection of experiments that explore their diverse capabilities.

**1. Finetuning Experiments**

This directory focuses on fine-tuning pre-trained LLMs for specific tasks and datasets. It includes a project titled **"Ask the Masters"**, which aims to enable users to ask questions and receive responses as if from specific experts. This involves fine-tuning base models with data from those experts. 

While the project is still in its early stages, initial results show improvements in the performance of fine-tuned models compared to their base versions. Currently, due to computational limitations, the experiments are restricted to 7-billion parameter quantized models running on a personal desktop.


**2. Inference Experiments**

This directory explores the inference capabilities of LLMs, focusing on text generation, translation, and question answering. It includes the following experiments:

* **Evaluating inference results from various open-sourced models:** This experiment compares the outputs and performance of different pre-trained LLMs.
* **Evaluating inference performance on specific hardware:** This experiment measures metrics such as total inference time, time to first token, and tokens per second on different hardware platforms (PC, Colab, etc.).

The experiment uses a standard set of prompts for all models and employs consistent code across different hardware platforms. The results consist of both subjective evaluations for each model and objective performance metrics for each hardware they were run on. Further analysis of these results is planned for future work.

**James Clear Finetuning** is a project to train LLM to response in the voice of the "Atomic Habits" author James Clear. This is 3 step process:
  1. Scrape James Clear website to fetch all his newsletters. 0_scrape_newsletter.py stores the newsletters in a csv format. 
  2. Prepare the data for finetuning by: Use Gemma-2 instruction tuned model to create questions-answer pairs for each of the paragraphs in the newsletters. qa pairs are stored in james_clear_qa21.jsonl format.
  3. Finetune the model using llama3.1-8B and using the data from the previous step. jc_finetuning.py finetunes the model and outputs the comparison of a before and after prompt.  


**3. RAG Experiments**

Continuing on journey to explore applications using open source LLMs, this section focuses on RAG.
Using local LLMs and querying your own data is a fascinating field, offering the potential for building your personal LLM. This repository focuses on implementing Retrieval-Augmented Generation (RAG) techniquesand  can significantly improve the accuracy and reliability of the LLM's outputs on your own personal data. 

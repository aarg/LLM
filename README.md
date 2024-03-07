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

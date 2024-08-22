"""
**James Clear Newsletter Q&A Generator**

This script ingests the newsletter csv file that we produced in scrape_newsletter.py. It generate question and answer (Q&A) pairs for each paragraph. 
The Q&A pairs can be used for fine-tuning language models. 

The script performs the following steps:

1. **Text Processing:**
    - Cleans the text from newsletters by removing unwanted characters and splitting it into manageable chunks.

2. **Q&A Generation:**
    - Uses either OpenAI's GPT-3.5 or a local model like Gemma-2-2b to generate Q&A pairs from the text chunks.

3. **Data Formatting:**
    - Formats the Q&A pairs into a structure suitable for fine-tuning language models, with each entry consisting of a prompt and a completion.

4. **Save Output:**
    - Saves the formatted data to a JSONL file, which can be used as input for training or fine-tuning language models.

**Requirements:**
- `openai`, `transformers`, `torch`, `re`, `csv`, `json`, `dotenv`, `os`, `random`

"""


import re
import json
import csv
from typing import List, Dict
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Text Processing Functions
def clean_text(text: str) -> str:
    # Remove "Share this on twitter" text
    text = re.sub(r'Share this on twitter', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters (customize as needed)
    text = re.sub(r'[^a-zA-Z0-9.,!?;\s]', '', text)
    return text


def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_idea(idea_text: str) -> List[str]:
    cleaned_text = clean_text(idea_text)
    return chunk_text(cleaned_text)


# Q&A Generation Functions

def generate_qa_pairs_openai(chunk: str) -> List[Dict[str, str]]:
    prompt = f"Based on the following text, generate 3 relevant questions and their corresponding answers:\n\n{chunk}\n\nQuestions and Answers:"
    from openai import OpenAI
    client = OpenAI(
        api_key = openai.api_key,
        )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates questions and answers based on given text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        temperature=0.7,
    )

    #qa_text = response.choices[0].message.content
    #qa_pairs = parse_qa_response(qa_text)
    qa_text = response.choices[0].message.content
    #print(f"Raw QA response:\n{qa_text}")  # Add this line for debugging
    qa_pairs = parse_qa_response(qa_text)
    #print(f"Parsed QA pairs: {qa_pairs}")  # Add this line for debugging
    return qa_pairs

def generate_qa_pairs_gemma(chunk: str) -> List[Dict[str, str]]:
    import time
    from transformers import BitsAndBytesConfig
    #Load Gemma model and tokenizer
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    quantization_config = BitsAndBytesConfig( 
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        #bnb_4bit_compute_dtype="torch.float16",
    )

    gemma_model_name = "google/gemma-2-2b-it"
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
    gemma_model = AutoModelForCausalLM.from_pretrained(gemma_model_name, quantization_config=quantization_config, device_map = 'cuda:0')

    device = torch.device('cuda:0')
    prompt = f"""You are an expert life coach who learns from experiences of leaders and experts in their field.
                 You learn from their blogs, books and podcasts.
                 You formulate questions based on quotes from their content and quotes.
                 Below is one such quote.Based on the following text, generate 3 relevant questions and their corresponding answers.
                 The question should be short. Do not give away the answer in your question.
                 The answers can be a bit longer consisting of a few sentences.
                 Also: If possible, ask for motvations, feelings, and perceptions rather than events or facts.
                 Format your response as '1. Question: [question here]' followed by ' Answer: [answer here]' for each pair: {chunk} Questions and Answers:"""

    inputs = gemma_tokenizer(prompt, return_tensors="pt").to("cuda")
    start = time.time()
    outputs = gemma_model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
    end = time.time()
    qa_text = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)

    #print(f"Raw QA response (Gemma):\n{qa_text}")
    print("\n[Time taken]: ",end-start)
    qa_pairs = parse_qa_response(qa_text)
    #print(f"Parsed QA pairs (Gemma): {qa_pairs}")
    return qa_pairs

def parse_qa_response(response: str) -> List[Dict[str, str]]:
    lines = response.strip().split('\n')
    qa_pairs = []
    current_question = ""
    current_answer = ""

    for line in lines:
        if re.match(r'^\d+\.\s*Question:', line):
            if current_question and current_answer:
                qa_pairs.append({"question": current_question, "answer": current_answer})
            current_question = re.sub(r'^\d+\.\s*Question:\s*', '', line).strip()
            current_answer = ""
        elif line.startswith('Answer:'):
            current_answer = line.replace('Answer:', '').strip()

    if current_question and current_answer:
        qa_pairs.append({"question": current_question, "answer": current_answer})

    return qa_pairs

def process_chunks(chunks: List[str], use_gemma: bool) -> List[Dict[str, str]]:
    all_qa_pairs = []
    for chunk in chunks:
        if use_gemma:
            qa_pairs = generate_qa_pairs_gemma(chunk)
        else:
            qa_pairs = generate_qa_pairs_openai(chunk)
        all_qa_pairs.extend(qa_pairs)
    return all_qa_pairs

# Data Formatting Functions
def format_for_fine_tuning(qa_pairs: List[Dict[str, str]], author_name: str) -> List[Dict[str, str]]:
    formatted_data = []
    for pair in qa_pairs:
        formatted_data.append({
            "prompt": f"Question: {pair['question']}\n\nAnswer as if you were {author_name}:",
            "completion": f" {pair['answer']}"
        })
    return formatted_data

def save_to_jsonl(data: List[Dict[str, str]], output_file: str):
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_csv(input_file: str, output_file: str, author_name: str, use_gemma: bool):
    all_qa_pairs = []
    counter = 1 

    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            idea_text = row['Ideas from me']
            if idea_text:
                print("\nIdea#:", counter)
                chunks = process_idea(idea_text)
                qa_pairs = process_chunks(chunks, use_gemma)
                all_qa_pairs.extend(qa_pairs)
                counter = counter + 1

    formatted_data = format_for_fine_tuning(all_qa_pairs, author_name)
    save_to_jsonl(formatted_data, output_file)


def main():
    import random
    import os

    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()

    # Access tokens
    openai_token = os.getenv('OPENAI_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACE_API_KEY')



    input_file = os.path.join(os.path.dirname(__file__), "james_clear_newsletters.csv")

    #input_file = "james_clear_newsletters_2.csv"
    output_file = os.path.join(os.path.dirname(__file__), "james_clear_qa"+str(random.randint(0,1000))+".jsonl")
    #output_file = "james_clear_qa"+str(random.randint(0,1000))+".jsonl"
    author_name = "James Clear"  # Replace with the actual author's name

    use_gemma = True  # Set to False to use OpenAI instead
    process_csv(input_file, output_file, author_name, use_gemma)
    print(f"Data processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
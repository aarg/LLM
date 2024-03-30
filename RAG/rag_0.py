### A simple rag application to query your own data
### Using meta-llama/Llama-2-7b-chat-hf, 8 bit quantized

import torch
import time
import warnings

from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


warnings.simplefilter('ignore')

def printhwconfig():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Reserved:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

Settings.llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    system_prompt=system_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="cuda:0", #Use "cpu" or "auto"
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

Settings.embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load files
documents = SimpleDirectoryReader("./data").load_data()
 
# Create vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

start_time = time.time()
# Query and print response
response = query_engine.query("What are the rules of investing?")
print(response)

print("\n" + "-"*80)
print(f"Inference time: {time.time() - start_time:.4f} seconds")
printhwconfig()
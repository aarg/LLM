# config.py
from typing import Dict, List

SUPPORTED_MODELS = {
    "1": {"name": "mixtral", "display": "Mixtral 8x7B"},
    "2": {"name": "mistral", "display": "Mistral 7B"},
    "3": {"name": "llama3.1", "display": "Llama3.1"},
    "4": {"name": "phi4", "display": "Phi-4"},
    "5": {"name": "deepseek-r1:14b", "display": "Deepseek-r1 14B"},
    "6": {"name": "deepseek-r1:32b", "display": "Deepseek-r1 32B"}
}

MODEL_SETTINGS = {
    "mixtral": {
        "temperature": 0.1,
        "num_ctx": 4096,
        "top_k": 10,
        "top_p": 0.95,
    },
    "mistral": {
        "temperature": 0.1,
        "num_ctx": 4096,
        "top_k": 10,
        "top_p": 0.95,
    },
    "phi4": {
        "temperature": 0.1,
        "num_ctx": 2048,
        "top_k": 10,
        "top_p": 0.95,
    },
    "llama3.1": {
        "temperature": 0.1,
        "num_ctx": 4096,
        "top_k": 10,
        "top_p": 0.95,
    }
}
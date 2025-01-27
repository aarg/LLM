# db_manager.py

import os
import json
from typing import List, Dict, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

class DatabaseManager:
    def __init__(self, base_dir: str = "db"):
        """
        Initialize the DatabaseManager.

        Args:
            base_dir (str): Base directory for storing database files.
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def get_db_path(self, model_name: str) -> str:
        """Get the database path for a specific model."""
        return os.path.join(self.base_dir, f"vectorstore_{model_name}")
    
    def db_exists(self, model_name: str) -> bool:
        """Check if a database exists for the given model."""
        db_path = self.get_db_path(model_name)
        return os.path.exists(db_path)
    
    def load_or_create_db(self, model_name: str, texts: Optional[List[Dict[str, str]]] = None) -> Chroma:
        """
        Load existing database or create new one if texts are provided.
        
        Args:
            model_name (str): Name of the model.
            texts (Optional[List[Dict[str, str]]]): List of text chunks with metadata.

        Returns:
            Chroma: The loaded or created vector store.
        """
        
        db_path = self.get_db_path(model_name)
        embeddings = OllamaEmbeddings(model=model_name)
        
        if texts:
            # Create new database or update existing one
            vector_store = Chroma.from_texts(
                texts=[t['text'] for t in texts],
                embedding=embeddings,
                metadatas=[t['metadata'] for t in texts],
                persist_directory=db_path
            )
            print(f"Database created/updated at: {db_path}")
        else:
            # Load existing database
            if self.db_exists(model_name):
                vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings
                )
                print(f"Loaded existing database from: {db_path}")
            else:
                raise FileNotFoundError(f"No database found for model: {model_name}")
        
        return vector_store
    
    def get_processed_statements(self, model_name: str) -> List[str]:
        """
        Get a list of processed statement files for a model.

        Args:
            model_name (str): Name of the model.

        Returns:
            List[str]: List of processed statement file paths.
        """
        db_path = self.get_db_path(model_name)
        processed_file = os.path.join(db_path, "processed_statements.txt")
        
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Create the file if it doesn't exist
        if not os.path.exists(processed_file):
            with open(processed_file, 'w') as f:
                pass  # Create an empty file
        
        # Read the processed statements
        with open(processed_file, 'r') as f:
            return f.read().splitlines()
    
    def add_processed_statement(self, model_name: str, statement_path: str):
        """
        Mark a statement as processed for a model.

        Args:
            model_name (str): Name of the model.
            statement_path (str): Path to the statement file.
        """
        db_path = self.get_db_path(model_name)
        processed_file = os.path.join(db_path, "processed_statements.txt")
        
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Append the statement to the file
        with open(processed_file, 'a') as f:
            f.write(f"{statement_path}\n")
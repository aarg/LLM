# main.py

import os
from typing import Optional
from config import SUPPORTED_MODELS
from db_manager import DatabaseManager
from statement_processor import StatementProcessor
from query_interface import create_qa_chain, interactive_query

def select_model() -> Optional[str]:
    """Let user select the model to use."""
    print("\nAvailable models:")
    for key, model in SUPPORTED_MODELS.items():
        print(f"{key}. {model['display']}")
    
    while True:
        choice = input("\nSelect model number (or 'exit'): ").strip()
        if choice.lower() == 'exit':
            return None
        if choice in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[choice]['name']
        print("Invalid choice. Please try again.")

def main():
    # Initialize managers
    db_manager = DatabaseManager()
    statement_processor = StatementProcessor()
    
    # Check for available statements
    available_statements = statement_processor.get_available_statements()
    if not available_statements:
        print("No PDF statements found in data directory.")
        print("Please add your bank statements to the 'data' directory.")
        return
    
    print(f"\nFound {len(available_statements)} statements in data directory:")
    for statement in available_statements:
        print(f"- {statement}")
    
    # Model selection
    model_name = select_model()
    if not model_name:
        return
    
    print(f"\nUsing model: {model_name}")
    
    # Process new statements if any
    texts = statement_processor.process_new_statements(db_manager, model_name)
    if texts:
        print("Processing new statements and updating database.")
    else:
        print("Using existing database with previously processed statements.")
    
    # Load vector store
    vector_store = db_manager.load_or_create_db(model_name, texts)
    
    # Create QA chain and start interactive session
    qa_chain = create_qa_chain(vector_store, model_name)
    interactive_query(qa_chain)

if __name__ == "__main__":
    main()
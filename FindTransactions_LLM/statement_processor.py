# statement_processor.py

import os
import json
from typing import List, Dict
import pdfplumber
from db_manager import DatabaseManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

class StatementProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def get_available_statements(self) -> List[str]:
        """Get list of PDF files in the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
    
    def process_statement(self, file_path: str) -> List[Dict[str, str]]:
        """Process a single statement and return chunks with metadata."""
        full_path = os.path.join(self.data_dir, file_path)
        
        # Extract text from PDF
        pages_content = []
        with pdfplumber.open(full_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    page_data = {
                        "page_number": page_num,
                        "content": text,
                        "processed_at": datetime.now().isoformat()
                    }
                    pages_content.append(page_data)
        
        # Save extracted text to JSON
        json_path = os.path.join(self.data_dir, f"{os.path.splitext(file_path)[0]}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pages_content, f, indent=2, ensure_ascii=False)
        
        # Combine and split text
        full_text = "\n".join([page['content'] for page in pages_content])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=10,
            length_function=len,
            separators=["\nDate: ", "\n"]
        )
        
        chunks = splitter.split_text(full_text)
        return [{"text": chunk, "metadata": {"source": file_path}} for chunk in chunks]
    
    def process_new_statements(self, db_manager: DatabaseManager, model_name: str) -> List[Dict[str, str]]:
        """Process any new statements for a given model."""
        available_statements = self.get_available_statements()
        processed_statements = db_manager.get_processed_statements(model_name)
        
        new_statements = [s for s in available_statements if s not in processed_statements]
        if not new_statements:
            return []
        
        # Process new statements
        all_chunks = []
        for statement in new_statements:
            chunks = self.process_statement(statement)
            all_chunks.extend(chunks)
            db_manager.add_processed_statement(model_name, statement)
        
        return all_chunks
# Bank Statement Analyzer

## Overview
The **Bank Statement Analyzer** is a Python-based application designed to help users analyze and query their bank statements in PDF format. It uses advanced natural language processing (NLP) techniques, powered by large language models (LLMs) and vector databases, to extract, process, and query transaction data from bank statements. The application provides an interactive interface for users to ask questions about their transactions and receive detailed, accurate responses.

---

## Key Features
1. **PDF Statement Processing**
   - Extracts text from PDF bank statements.
   - Splits text into manageable chunks for processing.

2. **Vector Database**
   - Stores extracted text chunks in a vector database for efficient retrieval.
   - Uses embeddings to enable semantic search.

3. **Interactive Query Interface**
   - Allows users to ask natural language questions about their transactions.
   - Supports queries like:
     - "Show all transactions in December."
     - "List all deposits over $100."
     - "Find all ATM withdrawals."

4. **Transaction Verification**
   - Verifies retrieved transactions to ensure no data is missed.
   - Provides follow-up queries to double-check results.

5. **Multi-Model Support**
   - Supports multiple Local LLMs (e.g., Mixtral, Mistral, Llama3.1, DeepSeek, Phi) for flexibility.
   - Allows users to choose the model that best suits their needs.

---

## How It Works
1. **Statement Upload**
   - Users place their PDF bank statements in the `data` directory.

2. **Text Extraction**
   - The application extracts text from the PDFs and splits it into chunks.

3. **Vector Database Creation**
   - Text chunks are converted into embeddings and stored in a vector database.

4. **Interactive Querying**
   - Users interact with the application through a command-line interface.
   - The application retrieves relevant chunks from the database and uses an LLM to generate responses.

5. **Transaction Verification**
   - The application verifies the retrieved transactions to ensure completeness.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: `pdfplumber`, `langchain`, `langchain-chroma`, `langchain-ollama`
- Ensure ollama is installed and running. Test all the models e.g `ollama run mistral`
- Create `data` and `db` directories in the root of the project.

---

## Usage
1. **Select a Model**
   - Choose a model from the available options (e.g., Mixtral, Mistral, Llama3.1, DeepSeek, Phi).

2. **Ask Questions**
   - Enter natural language queries about your transactions, such as:
     - "Show all transactions in December."
     - "List all deposits over $100."
     - "Find all ATM withdrawals."

3. **View Results**
   - The application will display the matching transactions, including dates, amounts, and descriptions.

---

## Quality Considerations

### Output Quality Notes

- Results may vary significantly depending on statement complexity and chosen model. Due to smaller models, the quality of the output is not as good as larger models, it did occassionally miss transactions.
- Single-statement queries consistently perform better than queries across multiple statements
- Larger models (e.g., Mixtral 8x7B, Deepseek 32B) generally provide superior results
- Each optimization technique implemented has contributed to incremental quality improvements

---

## Improvements Made to Enhance Quality
To improve the quality of the output, the following steps were taken:
1. **Optimized Text Chunking**
   - Adjusted `chunk_size` and `chunk_overlap` to ensure transactions are not split across chunks.
   - Used transaction delimiters (e.g., "Date:") to preserve transaction boundaries.

2. **Enhanced Vector Search**
   - Increased the number of chunks retrieved (`k=20`) to provide more context to the LLM.

3. **Improved LLM Prompts**
   - Made the prompt explicit about listing all transactions and avoiding summarization.
   - Added instructions to include exact dates, amounts, and descriptions.

4. **Post-Processing Verification**
   - Added a step to merge transactions from multiple chunks and remove duplicates.

5. **Debugging and Logging**
   - Added debug statements to log retrieved chunks and identify gaps in the results.

6. **Model Selection**
   - Tested multiple models (e.g., Mixtral, Mistral, Llama3.1, DeepSeek, Phi) to determine which performs best for specific use cases.


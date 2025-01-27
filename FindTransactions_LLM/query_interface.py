# query_interface.py

from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Optional, List, Dict, Union
import re
import time

def extract_transactions_from_response(response: Union[str, dict]) -> List[Dict]:
    """
    Extract structured transaction data from LLM response.
    
    Args:
        response: Either a string or dictionary response from the LLM
    Returns:
        List of transaction dictionaries
    """
    # Handle different response types
    if isinstance(response, dict):
        response_text = str(response.get('response', ''))
    else:
        response_text = str(response)
    
    transactions = []
    # Look for patterns like "$1,234.56" or "$1234.56" followed by date patterns
    amount_pattern = r'\$[\d,]+\.?\d*'
    date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}|\w+ \d{1,2},? \d{4}|\d{1,2}-\d{1,2}-\d{2,4}'
    
    # Split response into lines and analyze each
    lines = response_text.split('\n')
    for line in lines:
        amounts = re.findall(amount_pattern, line)
        dates = re.findall(date_pattern, line)
        if amounts and dates:
            transactions.append({
                'amount': amounts[0],
                'date': dates[0],
                'description': line.strip()
            })
    
    return transactions

def verify_transactions(qa_chain: RetrievalQA, original_query: str, transactions: List[Dict]) -> List[Dict]:
    """
    Verify transactions by asking follow-up questions.

    Args:
        qa_chain (RetrievalQA): The question-answering chain.
        original_query (str): The original user query.
        transactions (List[Dict]): List of transactions to verify.

    Returns:
        List[Dict]: List of verified transactions.
    """
    try:
        verified_transactions = transactions.copy()
        
        if not transactions:  # If no transactions to verify, return empty list
            return []
        
        # Generate verification queries
        verification_queries = [
            f"Are there any other transactions like {original_query} that were not mentioned? List only new transactions not including: {', '.join([t['amount'] for t in transactions])}",
            f"Double check if there are any {original_query} between {transactions[0]['date']} and the end of the statement",
            f"List all transactions similar to {original_query} that are smaller than {transactions[-1]['amount']}"
        ]
        
        # Run verification queries
        for query in verification_queries:
            try:
                response = qa_chain.invoke({"query": query})
                response_text = response.get("result", response.get("response", ""))
                #if not response_text:
                #    print(f"Debug: No response text found for verification query: {query}")
                #    continue
                new_transactions = extract_transactions_from_response(response_text)
                
                # Add new transactions that weren't in the original list
                for new_trans in new_transactions:
                    if new_trans not in verified_transactions:
                        verified_transactions.append(new_trans)
            except Exception as e:
                print(f"Warning: Verification query failed: {str(e)}")
                continue
        
        return verified_transactions
    except Exception as e:
        print(f"Warning: Transaction verification failed: {str(e)}")
        return transactions  # Return original transactions if verification fails

def create_specific_query(question: str) -> str:
    """
    Create a more specific query based on common patterns.
    """
    # Common financial query patterns
    patterns = {
        'transfers': r'transfer|sent|received',
        'payments': r'pay|paid|payment',
        'deposits': r'deposit|deposited',
        'withdrawals': r'withdraw|withdrawal|atm',
    }
    
    enhanced_query = question
    for category, pattern in patterns.items():
        if re.search(pattern, question.lower()):
            enhanced_query = f"""List ALL {category} in the statement with exact dates and amounts. 
            Include every single transaction, no matter how small. Format as:
            Date: [date]
            Amount: [amount]
            Description: [description]"""
            break
    
    return enhanced_query

def format_response(response: Union[str, dict]) -> str:
    """
    Format the response for display.
    """
    if isinstance(response, dict):
        return str(response.get("result", response.get("response", "")))
    return str(response)

def run_query_with_verification(qa_chain: RetrievalQA, question: str) -> str    :
    """
    Run query with verification and transaction aggregation.
    """
    try:
        # Create specific query
        enhanced_query = create_specific_query(question)
        
        # Get initial response
        initial_response = qa_chain.invoke({"query": enhanced_query})
        #print("Debug: Initial Response Structure:", initial_response)

        response_text = initial_response.get("result", initial_response.get("response", ""))
        if not response_text:
        #   print("Debug: No response text found in the initial response.")
            return "No response found."


        transactions = extract_transactions_from_response(response_text)
        
        # If transactions found, verify them
        if transactions:
            verified_transactions = verify_transactions(qa_chain, question, transactions)
            
            # Format final response
            if len(verified_transactions) > len(transactions):
                response = "Found additional transactions after verification:\n\n"
                for tx in verified_transactions:
                    response += f"Date: {tx['date']}\n"
                    response += f"Amount: {tx['amount']}\n"
                    response += f"Description: {tx['description']}\n\n"
            else:
                response = format_response(response_text)
        else:
            # If no transactions found, try a different approach
            fallback_query = f"Look for ANY transaction related to: {question}. Include ALL matches, even partial ones."
            fallback_response = qa_chain.invoke({"query": fallback_query})
            fallback_response_text = fallback_response.get("result", fallback_response.get("response", ""))
            response = format_response(fallback_response_text)
        
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}\nOriginal response: {format_response(response_text)}"

def create_qa_chain(vector_store: Chroma, model_name: str) -> RetrievalQA:
    """
    Create a question-answering chain with enhanced prompt.
    """
    prompt_template = """Instructions: Analyze the bank statement and list ALL relevant transactions.
    Never summarize or group transactions - list each one separately.
    
    Requirements:
    1. List EVERY transaction that matches the query
    2. Include exact dates and amounts
    3. Format each transaction on new lines
    4. Never skip transactions, no matter how small
    5. If you find any transaction, always include its date and amount
    6. Always focus on the **Amount** column for transaction amounts. Ignore the **Balance** column unless specifically asked for.
    
    Bank Statement Information:
    {context}
    
    Question: {question}
    
    Response: Let me list all matching transactions:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,
        num_ctx=2048,
        top_k=10,
        top_p=0.95,
        repeat_penalty=1.1
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 20}
        ),
        chain_type_kwargs={"prompt": PROMPT}
    )

def interactive_query(qa_chain: RetrievalQA):
    """
    Run an interactive query session with verification.
    """
    print("\n📊 Bank Statement Analysis Interface")
    print("=" * 40)
    
    while True:
        try:
            user_input = input("\nQuestion (or 'exit'): ").strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break

            # Record start time
            start_time = time.time()
            
            print("\n🔍 Analyzing ...")
            response = run_query_with_verification(qa_chain, user_input)
            

            # Record end time and calculate response time
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            print("\n📍 Answer:")
            print(response)
            print(f"\n⏱️ Response time: {minutes}:{seconds:02d} mins")


            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        print("\n" + "-" * 40)
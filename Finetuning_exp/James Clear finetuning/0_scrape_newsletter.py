"""
**James Clear 3-2-1 Newsletter Scraper**

This Python script scrapes the content of James Clear's 3-2-1 newsletters from his website. 
The script performs the following tasks:

1. **Scrape Newsletter URLs:**
    - The script starts by scraping all the URLs of the newsletters available on the James Clear 3-2-1 archive page.

2. **Extract Content from Each Newsletter:**
    - For each newsletter, the script extracts key components including the date, "3 Ideas from Me," "2 Quotes from Others," and "1 Question for You."

3. **Save Results:**
    - The scraped data is saved to a CSV file for easy access and further analysis.

4. **Create DataFrame:**
    - The scraped content is also stored in a pandas DataFrame, making it easier to manipulate and analyze the data programmatically.

**Requirements:**
- `requests`, `bs4` (BeautifulSoup), `re`, `requests_html`, `tqdm`, `csv`, `datetime`, `pandas`

"""

import requests
from bs4 import BeautifulSoup
import re
from requests_html import HTMLSession
from tqdm import tqdm
import csv
from datetime import datetime
import pandas as pd

def get_newsletter_urls(base_url='https://jamesclear.com/3-2-1'):
    """
    Scrape all newsletter URLs from the James Clear 3-2-1 archive page.
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=re.compile(r'/3-2-1/'))
    #return ['https://jamesclear.com' + link['href'] for link in links if link['href'] != '/3-2-1'][:limit]
    return [link['href'] for link in links if link['href'] != '/3-2-1']

def extract_newsletter_content(url):
    """
    Extract the date, ideas, quotes, and question from a single newsletter.
    """
    session = HTMLSession()
    r = session.get(url)
    #r.html.render(timeout=20)

    full_text = r.html.full_text

    # Extract date from URL
    date_match = re.search(r'/3-2-1/(.+)$', url)
    date = datetime.strptime(date_match.group(1), '%B-%d-%Y').strftime('%Y-%m-%d') if date_match else 'Unknown'

    # Extract ideas
    ideas_match = re.search(r'3 IDEAS FROM ME\s+(.*?)\s+2 QUOTES FROM OTHERS', full_text, re.DOTALL)
    ideas = ideas_match.group(1).strip() if ideas_match else ''

    # Extract quotes
    quotes_match = re.search(r'2 QUOTES FROM OTHERS\s+(.*?)\s+1 QUESTION FOR YOU', full_text, re.DOTALL)
    quotes = quotes_match.group(1).strip() if quotes_match else ''

    # Extract question
    question_match = re.search(r'1 QUESTION FOR YOU\s+(.*?)\s+Until next week,', full_text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ''

    return {
        'Date': date,
        'URL': url,
        'Ideas from me': ideas,
        'Quotes from Others': quotes,
        'Questions for you': question
    }

def scrape_all_newsletters(urls):
    """
    Scrape the content from all provided URLs.
    """
    newsletters = []
    for url in tqdm(urls, desc="Scraping newsletters"):
        content = extract_newsletter_content(url)
        newsletters.append(content)
    return newsletters

def save_to_csv(data, filename='james_clear_newsletters.csv'):
    """
    Save the scraped data to a CSV file.
    """
    keys = ['Date', 'URL', 'Ideas from me', 'Quotes from Others', 'Questions for you']
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def create_dataframe(data):
    """
    Create a pandas DataFrame from the scraped data.
    """
    return pd.DataFrame(data)

def main():
    """
    Main function to run the scraper, save results to CSV, and create a pandas DataFrame.
    """
    print("Fetching newsletter URLs...")
    urls = get_newsletter_urls()
    print(f"Found {len(urls)} newsletters.")
    
    print("\nScraping content from each newsletter...")
    newsletters = scrape_all_newsletters(urls)
    
    print("\nSaving results to CSV...")
    save_to_csv(newsletters)
    
    print("\nCreating pandas DataFrame...")
    df = create_dataframe(newsletters)
    
    print(f"\nScraping complete. Results saved to 'james_clear_newsletters.csv' and stored in pandas DataFrame.")
    
    return df

# This block is useful when running the script locally
if __name__ == "__main__":
    df = main()
    print(df.head())  # Display the first few rows of the DataFrame

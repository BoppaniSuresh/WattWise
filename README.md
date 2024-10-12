# WattWise

WattWise is an AI-powered tool designed to combine energy-saving tips from both web scraping and PDF documents, generating concise, actionable insights using advanced language models. It utilizes web scraping techniques, document embedding, and large language models (LLMs) to provide users with relevant, summarized answers to energy-related questions.

## Features
- **Web Scraping**: Extracts energy-saving tips from the U.S. Department of Energy's website using Selenium.
- **PDF Processing**: Parses and splits content from energy-saving PDF guides.
- **Document Embedding**: Uses Chroma and SentenceTransformer for efficient document embeddings.
- **LLM-based Question Answering**: Powered by Hugging Face's Mistral-7B model to generate responses based on combined data from web scraping and PDFs.
- **Text Generation**: Provides concise and actionable bullet-point answers to user queries.

## Project Structure
- `chatbot.py`: The main script that contains all the code for web scraping, PDF processing, embedding, and question-answering using the LLM.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/BoppaniSuresh/WattWise.git
   cd WattWise

   Install required dependencies
   
   pip install -r requirements.txt

## How to Run 
1. Run the chatbot: This script handles web scraping, PDF processing, and querying with the LLM.
   ```bash
   python chatbot.py

The system will retrieve relevant information from both the scraped web content and the PDF data, providing you with actionable tips.   
 
   

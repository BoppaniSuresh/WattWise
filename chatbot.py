import os
import torch
import transformers  # type: ignore
from flask import Flask, request, jsonify  # type: ignore
import requests
from langchain.chains import LLMChain  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings  # type: ignore
from langchain_community.llms import HuggingFacePipeline  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.schema.runnable import RunnablePassthrough  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain.schema import Document  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import warnings
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

warnings.filterwarnings("ignore")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API_TOKEN"

loader = PyPDFLoader("energy-saver-guide-2022.pdf")
pdf_data = loader.load_and_split()

print('Installing the Chrome driver...')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

print('Opening the website')
url = 'https://www.energy.gov/save/tools-and-tips'
driver.get(url)

time.sleep(5)

scraped_text = ""

try:
    element_1 = driver.find_element(By.XPATH, '//*[@id="block-particle-content"]/div/main/div/section[2]')
    scraped_text += element_1.text + "\n\n"

    element_2 = driver.find_element(By.XPATH, '//*[@id="block-particle-content"]/div/main/div/section[3]')
    scraped_text += element_2.text + "\n\n"

    element_3 = driver.find_element(By.XPATH, '//*[@id="block-particle-content"]/div/main/div/section[4]')
    scraped_text += element_3.text + "\n\n"

    accordion_sections = WebDriverWait(driver, 50).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "accordion"))
    )
    
    for i in range(len(accordion_sections) - 16):
        button = driver.find_element(By.XPATH, f"/html/body/div[2]/div/div/div/main/section/div/div/div/div/main/div/section[4]/div/div/div/div/div[{i+1}]/div/div[1]/h4/button")
        button.click()
        time.sleep(1)  

        paragraphs = driver.find_elements(By.XPATH, f"/html/body/div[2]/div/div/div/main/section/div/div/div/div/main/div/section[4]/div/div/div/div/div[{i+1}]/div/div[2]//p")
        
        for para in paragraphs:
            scraped_text += para.text + "\n"

finally:
    driver.quit()

print("Web content has been scraped.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    separators=['\n\n', '\n', '(?=>\. )', ' ', '']
)

pdf_docs = text_splitter.split_documents(pdf_data)

scraped_docs = text_splitter.split_text(scraped_text)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

scraped_documents = [Document(page_content=doc) for doc in scraped_docs]

pdf_docsearch = Chroma.from_documents(pdf_docs, embeddings)
scraped_docsearch = Chroma.from_documents(scraped_documents, embeddings)

combined_documents = pdf_docs + scraped_documents

combined_docsearch = Chroma.from_documents(combined_documents, embeddings)

model_name = 'filipealmeida/Mistral-7B-Instruct-v0.1-sharded'
offload_folder = 'offload_folder'
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=offload_folder,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

text_generation_pipeline = transformers.pipeline(
    model=llm,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
)

# 7. Prompt Template
prompt_template = """
Instruction: Answer the question in concise bullet points based on your knowledge. Here is context to help:

{context}

### QUESTION:
{question}
"""

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

combined_retriever = combined_docsearch.as_retriever()

rag_chain = (
    {"context": combined_retriever, "question": RunnablePassthrough()}
    | llm_chain
)

response1 = rag_chain.invoke("How to save energy in the home?")

question = response1.get('question', 'N/A')
answer_text = response1.get('text', 'N/A')

print("### Question:")
print(question)
print("\n### Answer in Bullet Points:")

with open('answer.txt', 'w') as file:
    file.write("### Question:\n")
    file.write(f"{question}\n\n")
    file.write("### Answer in Bullet Points:\n")

    if isinstance(answer_text, str):
        clean_text = answer_text.replace("\n", " ").strip()
        bullet_points = clean_text.split(". ")

        for point in bullet_points:
            if point:
                file.write(f"- {point.strip()}.\n")  
    else:
        file.write("No answer text found.")

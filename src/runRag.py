
import tiktoken
import textwrap
import torch
import spacy
import csv
import time
import sys
import os

from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

from src.getCorpus import getCorpusSTJ, getCorpusSTF

sys.path.append(str(Path(__file__).parent.parent.parent)) 

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

use_long_text = True

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

def rag(texto):
    
    ## Model
    model_name = "gpt-4o-mini"
    
    ## Get the corpus
    corpus = getCorpusSTF()

    ## Initialize the model
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
 
    prompt_template = """Dentre os 50 temas STF e STJ mais similares, 
        faça uma nova analise e sintetize e liste os 5 temas que mais combinam com o corpus.

        {texto}

    Lista de temas mais relevantes em portugues:"""
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["texto"])

    num_tokens = num_tokens_from_string(texto, model_name)

    gpt_40_mini = 128000
    verbose 	= True

import sys
import os
import warnings

import streamlit as sl
from pathlib import Path
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent.parent)) 

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_prompt():
    
    prompt = """ Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, 
        que busca assinalar os temas STF ou STJ mais relevantes de cada processo .
        Contexto = {context}
        Pergunta = {question}.
        Lista ordenadamente por relevencia os temas mais relevantes em portugues:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    
    return prompt

def load_llm():
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    return llm

import os
import sys
import time
import openai
import anthropic

import streamlit as st

from openai import OpenAI
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import AsyncHtmlLoader

from src.runLLM import load_prompt, load_llm
from src.parserDoc import getContentHtml, getContentAllHtml, getContentPdf
from src.runSimilarity import similarityCompare, similarityTop

sys.path.append(str(Path(__file__).parent.parent.parent)) 

# Connect to OpenAI GPT-3, fetch API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

def initialize_session_state():
    
    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = None

def save_uploadedfile(uploadedfile):
    
    with open(os.path.join("uploaded", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        
def remove_files():
    
    path = os.path.join(os.getcwd(), 'uploaded')
    
    for file_name in os.listdir(path):
    
        file = os.path.join(path, file_name)
    
        if os.path.isfile(file) and file.endswith(".pdf"):
            print('Deleting file:', file)
            os.remove(file)
            
def extract_data():
    
    text_chunks = []
    
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("uploaded"))
        
    file_list = list(files)    
    
    for file in file_list:
        
        if '.pdf' in file:
        
            loader = PyPDFLoader(os.path.join('uploaded', file))
            
            text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
                chunk_size = 512,
                chunk_overlap = 30,
                length_function = len,
                separators= ["\n\n", "\n", ".", " "]
            ))
            
        if '.html' in file:
                     
            text_content = getContentAllHtml(file)
            
            loader = TextLoader(text_content)
            
            text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
                chunk_size = 512,
                chunk_overlap = 30,
                length_function = len,
                separators= ["\n\n", "\n", ".", " "]
            ))
            
            # documents = loader.load()
            
            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            # text_chunks = text_splitter.split_documents(documents)
            
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
    
    return vectorstore

            
def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)

with st.sidebar:
    
    st.subheader('')
    
    st.markdown('Aplicação do Processamento de Linguagem Natural na Análise de Similaridades de Temas STF/STJ'
        'Baseado em um texto de um recurso ou acórdão, quais os temas do STF/STJ possuem similaridade com os textos do processo.')
    

if __name__ == '__main__':
    
    ## Load LLM
    llm = load_llm()
    
    ## Load Prompt
    prompt = load_prompt()
    
    st.title("📝 Similaridades de Temas")
    
    retRag = []
    ragString = ''
    
    with st.form("ragExec", clear_on_submit=True):        
        
        uploaded_file = st.file_uploader(label="Faça o Upload do seu arquivo:", accept_multiple_files=True, type=["html", "pdf"])
        
        submitted = st.form_submit_button("Salvar Documento")
        
    # Section Run LLM
    if submitted and uploaded_file != []:
                
        if '.pdf' in uploaded_file[0]:                
            ## Get PDF Content
            retPDF = getContentPdf(uploaded_file[0].name)
        else:
            retPDF = getContentAllHtml(uploaded_file[0].name)
        
        retSimi = similarityTop(retPDF, 'distiluse-base-multilingual-cased-v2')
        
        retRag.append(retSimi)
        
        for rr in retRag[0]:
            ragString += rr
        
        initialize_session_state()
        
        for pdf in uploaded_file:
            save_uploadedfile(pdf)
            
        st.session_state.knowledge_base = extract_data()
        
        remove_files()
        
        uploaded_file = []
        
        alert = st.success(body=f"Realizado o Upload do PDF com Sucesso!", icon="✅")
        
        time.sleep(3)         
        alert.empty()    

    # uploaded_file = st.file_uploader("Faça o Upload de um arquivo de processo", type=("html", "pdf"))

    question = st.text_area(
        label="Pergunta algo sobre o documento enviado:",
        value=f"Com base na lista de temas do STF/STJ abaixo, analise o seguinte documento e identifique a quais temas ele mais se assemelha. Considere a relação de conteúdo, jurisprudência aplicável e palavras-chave presentes no texto. Liste os temas mais relevantes e explique brevemente o motivo da correspondência. \n\nTemas Similares: \n\n {ragString}",
        # disabled = not uploaded_file,
        height=450,
    )

    if question:
        
        ## Colocar string inteira do retorno dos teams
        queryTemas = question + ragString
        
        try:
            similar_embeddings = st.session_state.knowledge_base.similarity_search(queryTemas)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
            
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
        
            response = rag_chain.invoke(queryTemas)
            
            ## Imprime Resposta da LLM
            st.write(response)          

        except:
            
            alert = st.warning("Por favor, Realize o Upload do PDF que Deseja Realizar Chat", icon="🚨")
            time.sleep(3)
            alert.empty()
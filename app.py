# gpt3 professional email generator by stefanrmmr - version June 2022

import os
import sys
import openai
import streamlit as st
import pandas as pd
import numpy as np
import time

import streamlit.components.v1 as components

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from annotated_text import annotated_text
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.parserDoc import getContentHtml, getContentAllHtml, getContentPdf
from src.runSimilarity import similarityCompare, similarityTop
from src.runSummarize import summaryText
from src.runLLM import load_prompt, load_llm
# from src.runRag import rag

sys.path.append(str(Path(__file__).parent.parent.parent)) 

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="Temas TJRS", page_icon="img/rephraise_logo.png",)

# Connect to OpenAI GPT-3, fetch API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

def summaryTextGPT3(text):
    
    ret = getContentAllHtml(text)
        
    r = summaryText(ret)
        
    return r

def callParser(arquivo, modelo):
    
    ret = getContentAllHtml(arquivo)
        
    r = similarityTop(ret, modelo)
        
    return r

def extract_data():
    
    text_chunks = []
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("uploaded"))
    file_list = list(files)
    
    for file in file_list:
        loader = PyPDFLoader(os.path.join('uploaded', file))
        text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 30,
            length_function = len,
            separators= ["\n\n", "\n", ".", " "]
        ))
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
    
    return vectorstore

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
            
def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)

def main():

    ## LOGO e Título
    st.image('img/logo.png')  

    # Using "with" notation
    with st.sidebar:
        
        st.subheader('Similaridade de Temas')
        
        st.markdown('Estudo e Aplicabilidade do Processamento de Linguagem Natural na Análise de Similaridades de Temas STF/STJ'
        'Baseado em um texto de um recurso ou acórdão, quais os temas do STF/STJ possuem similaridade com os textos do processo.'
        'O objetivo é verificar se a ação pode ficar sobrestada aguardando uma decisão superior.')
    
    st.write('\n')  

    ## Dataset Corpus 
    st.subheader('\nTemas Corpus (Dataset)\n')

    st.markdown('Extração de dados do site oficial do STF e STJ, que consta informações de repercussão, descrição, título e tese dos temas. Dataset composto por 2685 registros (temas).')
    
    st.subheader('\nDataset STF (exemplo)\n')
    
    tab1, tab2 = st.tabs(["Dados STF", "Dados STJ"])
    
    with tab1:        
        df_stf = pd.read_csv("data/dataset_stf.csv")
        st.write(df_stf)
        
    with tab2:
        df_stj = pd.read_csv("data/dataset_stj.csv", delimiter=";", on_bad_lines='skip')
        st.write(df_stj)
    
    st.subheader('\nAnalise de Similaridades\n')
    
    input_c1 = ''
    input_c2 = ''
    
    retRag = []
        
    with st.form("parserFiles"):

        col1, col2 = st.columns(2)
        
        filesPdf = [ f for f in os.listdir('test') if f.endswith('.pdf') ]
        filesHtml = [ f for f in os.listdir('test') if f.endswith('.html') ]

        with col1:
            input_c1 = st.selectbox('Escolha o arquivo (Processo .html)', filesHtml, index=None, placeholder="Selecionar")
        
        with col2:
            input_c2 = st.selectbox('Escolha o arquivo (Recurso .pdf)', filesPdf, index=None, placeholder="Selecionar")
        
        col1, col2, col3 = st.columns([5, 5, 7])
        
        with col1:
            input_metric = st.selectbox('Escolha a metrica:', ('Cosine Similarity', 'Re-Rank', 'Cross-encoder'), index=0)
    
        with col2:
            input_model = st.selectbox('Escolha o modelo:', ('paraphrase-multilingual-MiniLM-L12-v2', 'distiluse-base-multilingual-cased-v2', 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'multi-qa-distilbert-cos-v1'), index=0)

        with col3:
            
            st.write("\n")  # add spacing
            st.write("\n")  # add spacing
            
            submitted = st.form_submit_button("Run Model")
            
        if submitted:
            
            retorno = callParser(input_c1, input_model)   
            retRag.append(retorno)
            
            try:
            
                st.write('**Visualização de Documentos**')            

                tab1, tab2 = st.tabs(["Processo", "Recurso (.pdf)"])

                col1, col2 = st.columns(2)
                
                with tab1:

                    with open(f'test/{input_c1}','r', encoding='latin-1') as f: 
                        html_data = f.read()

                    # Show in webpage
                    st.components.v1.html(html_data, scrolling=True, height=900)

                with tab2:

                    pdf_viewer(f"test/{input_c2}", pages_to_render=[1])        
                        
            except FileNotFoundError:        
                pass;

            # st.divider() 
            
            # st.write('**Sumarização de Documentos**') 
                       
            # st.markdown(retSum)

            st.divider() 
            
            st.info('Top 50 most similar sentences in corpus:\n')
            
            for r in retorno:
                
                st.write('\n')            
                st.markdown(r)
            
            st.divider() 

            col1, col2 = st.columns(2)            
            
            ## PCA Result
            st.image("img/pca_exemplo.png")  
            
    
    st.subheader('Prompt RAG\n')
            
    with st.form("ragExec", clear_on_submit=True):     
        
        pdf_docs = st.file_uploader(label="Faça o Upload do seu PDF:", accept_multiple_files=True, type=["pdf"])       
        
        # txt = st.text_area(
        #     "Escreva um texto para sumarização",
        #     "Dentre os 50 temas STF e STJ mais similares, faça uma nova analise e sintetize e liste os 5 temas que mais combinam com o corpus.",
        #     label_visibility='hidden'
        # )
        
        submitted = st.form_submit_button("Save Document")
        
    # Run LLM
    if submitted and pdf_docs != []:
        
        initialize_session_state()
        
        for pdf in pdf_docs:
            save_uploadedfile(pdf)
            
        st.session_state.knowledge_base = extract_data()
        
        remove_files()
        
        pdf_docs = []
        
        alert = st.success(body=f"Realizado o Upload do PDF com Sucesso!", icon="✅")
        
        time.sleep(3)         
        alert.empty()
        
    
    query = st.text_input(label='Faça uma pergunta sobre o documento:')
    
    if query:
        
        try:
            similar_embeddings = st.session_state.knowledge_base.similarity_search(query)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
            
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
        
            response = rag_chain.invoke(query)
            st.write(response)          

        except:
            
            alert = st.warning("Por favor, Realize o Upload do PDF que Deseja Realizar Chat", icon="🚨")
            time.sleep(3)
            alert.empty()

if __name__ == '__main__':
    
    
    llm     = load_llm()
    prompt  = load_prompt()
       
    # call main function
    main()

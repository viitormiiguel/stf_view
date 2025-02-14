# gpt3 professional email generator by stefanrmmr - version June 2022

import os
import sys
import openai
import streamlit as st
import pandas as pd
import numpy as np

import streamlit.components.v1 as components

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from annotated_text import annotated_text
from pathlib import Path

from src.parserDoc import getContentHtml, getContentAllHtml, getContentPdf
from src.process import similarityCompare

sys.path.append(str(Path(__file__).parent.parent.parent)) 

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="Temas TJRS", page_icon="img/rephraise_logo.png",)

# Connect to OpenAI GPT-3, fetch API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

def callParser(arquivo, modelo):
    
    ret = getContentHtml(arquivo)
    
    similarityCompare(ret, modelo)
    
    return ''

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
            # st.button("Run Model")
            
            if submitted:
                callParser(input_c1, input_model)                
            
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

    

    with st.expander("Resultados Similaridade de Documentos", expanded=True):
    
        st.info('Top 5 most similar sentences in corpus:\n')               
        
        st.write('\n')            
        st.markdown('**Tema 1230** - Termo inicial do reajuste do aux lio-alimenta  o dos servidores do Poder Judici rio da Uni o, considerando-se as disposi  es da Portaria Conjunta 1/2016 do Conselho Nacional de Justi a e da Portaria 297/2016 do Conselho da Justi a Federal. **(Score: 0.7539)**')
        st.write('\n')  
        st.markdown('**Tema 915** - Extens o, por via judicial, aos servidores do Poder Judici rio do Estado do Rio de Janeiro do reajuste concedido pela Lei estadual 1.206/1987. **(Score: 0.7499)**')
        st.write('\n')  
        st.markdown('**Tema 698** - Limites do Poder Judici rio para determinar obriga  es de fazer ao Estado, consistentes na realiza  o de concursos p blicos, contrata  o de servidores e execu  o de obras que atendam o direito social da sa de, ao qual a Constitui  o da Rep blica garante especial prote. **(Score: 0.7483)**')
        st.write('\n')  
        st.markdown('**Tema 248** - Pressupostos de admissibilidade de a  o rescis ria no  mbito da Justi a do Trabalho.**(Score: 0.7471)**')
        st.write('\n')  
        st.markdown('**Tema 811** - a) Cabimento de a  o penal privada subsidi ria da p blica ap s o decurso do prazo previsto no art. 46 do C digo de Processo Penal, na hip tese de o Minist rio P blico n o oferecer den ncia, promover o arquivamento ou requisitar dilig ncias externas no prazo legal;'
            'b) Ocorr ncia de prejudicialidade da queixa quando o Minist rio P blico, ap s o prazo legal para propositura da a  o penal (art. 46 do CPP), oferecer den ncia, promover o arquivamento do inqu rito ou determinar a realiza  o de dilig ncias externas. **(Score: 0.7393)**')

        st.divider() 

        col1, col2 = st.columns(2)            
        
        st.image("img/pca_exemplo.png")  


if __name__ == '__main__':
    
    # call main function
    main()

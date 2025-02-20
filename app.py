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
from src.runSimilarity import similarityCompare, similarityTop
from src.runSummarize import summaryText

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
        
    r = similarityTop(preprocess_text, modelo)
        
    return r

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
            
        if submitted:
            
            retorno = callParser(input_c1, input_model)   
            
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
            
    with st.form("ragExec"):            
        
        txt = st.text_area(
            "Escreva um texto para sumarização",
            "Dentre os 50 temas STF e STJ mais similares, faça uma nova analise e sintetize e liste os 5 temas que mais combinam com o corpus.",
            label_visibility='hidden'
        )
        
        submitted = st.form_submit_button("Run LLM")


if __name__ == '__main__':
    
    preprocess_text = '''Trata-se de embargos de declaração opostos por BANCO DO BRASIL S/A, nos autos do cumprimento de sentença que lhe move JOAQUIM MANOEL
    GRAVATO GERALDES, em face do acórdão que julgou o agravo de instrumento nº  5287315-84.2023.8.21.7000/RS, assim ementado:
    AGRAVO DE INSTRUMENTO. NEGÓCIOS JURÍDICOS BANCÁRIOS. EXPURGOS INFLACIONÁRIOS. CÉDULA RURAL PIGNORATÍCIA. AÇÃO COLETIVA. CUMPRIMENTO DE SENTENÇA.
    VALOR DO LAUDO: R$ 1.240.303,39. FUNDAMENTO: CÉDULAS RURAIS NºS 89/00839-1 E 89/00875-8. ATUALIZAÇÃO MONETÁRIA. O TÍTULO JUDICIAL REFERIU QUE DEVEM SER 
    CORRIGIDOS MONETARIAMENTE OS VALORES A CONTAR DO PAGAMENTO A MAIOR PELOS ÍNDICES APLICÁVEIS AOS DÉBITOS JUDICIAIS (...), 
    QUESTÃO QUE NÃO PODE SER ALTERADA NESTA FASE, POIS ACOBERTADA PELA COISA JULGADA, NO CASO, O ENTENDIMENTO DESTE JULGADOR É DE QUE O ÍNDICE APLICÁVEL 
    AOS DÉBITOS JUDICIAIS É O IGP-M-FORO, POIS INDEXADOR QUE MELHOR REFLETE A CORROSÃO DA MOEDA PELO FENÔMENO INFLACIONÁRIO. A UTILIZAÇÃO DO 
    PROVIMENTO Nº 14/2022-CGJ, PUBLICADO EM 07/04/2022, SOMENTE SERÁ POSSÍVEL EM CARÁTER SUBSIDIÁRIO, OU SEJA, QUANDO INEXISTIR DEFINIÇÃO A 
    RESPEITO NOS AUTOS OU NA LEGISLAÇÃO, O QUE NÃO É O CASO DO PRESENTE FEITO.  NO PONTO, RECURSO DESPROVIDO. JUROS DE MORA – TERMO INICIAL. 
    MESMO EM EXECUÇÕES OU CUMPRIMENTOS DE SENTENÇA INDIVIDUAIS, OS JUROS DE MORA INCIDEM A PARTIR DA CITAÇÃO DO DEVEDOR NO PROCESSO DE CONHECIMENTO DA 
    AÇÃO CIVIL PÚBLICA QUANDO ESTA SE FUNDAR EM RESPONSABILIDADE CONTRATUAL, CUJO INADIMPLEMENTO JÁ PRODUZA A MORA, SALVO A CONFIGURAÇÃO DESTA EM MOMENTO ANTERIOR. 
    ENTENDIMENTO PACIFICADO EM SEDE DE JULGAMENTO REPETITIVO PELO SUPERIOR TRIBUNAL DE JUSTIÇA, NO RESP 1.370.899/SP (TEMA 685 DOS RECURSOS REPETITIVOS), 
    CUJA APLICAÇÃO DEVE SER OBSERVADA EM TODOS OS RECURSOS QUE VENTILEM A MESMA CONTROVÉRSIA. NO PONTO, RECURSO DESPROVIDO. AGRAVO DE INSTRUMENTO DESPROVIDO, POR UNANIMIDADE.
    (TJRS, AGRAVO DE INSTRUMENTO Nº 5287315-84.2023.8.21.7000, 24ª CÂMARA CÍVEL , DESEMBARGADOR JORGE MARASCHIN DOS SANTOS, POR UNANIMIDADE, JULGADO EM 29/11/2023)
    A parte embargante alega que há vícios na decisão recorrida. Sustenta que devem ser aplicados os indíces de correção dos débitos judiciais da Justiça Federal. 
    Argumenta que a aplicação do IGP-M não está prevista na decisão exequenda e acaba por violar a coisa julgada. Afirma que, em se tratando de 
    devedores solidários, não pode haver consequências diferentes sobre a mesma dívida. Pondera ser omisso o acórdão quanto ao fato de que a aplicação 
    do IGP-M implica em onerosidade excessiva ao devedor, bem como sobre a utilização do IPCA em todo o período. Manifesta que os juros de mora devem ser 
    contados desde a citação inicial em cada uma das liquidações e execuções individuais. Prequestiona os dispositivos legais invocados. Pede provimento.'''
    
    # call main function
    main()

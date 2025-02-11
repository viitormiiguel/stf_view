# gpt3 professional email generator by stefanrmmr - version June 2022

import os
import openai
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="Temas TJRS", page_icon="img/rephraise_logo.png",)
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # lightmode
# Design change height of text input fields headers
st.markdown('''<style>.css-qrbaxs {min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)
# Design change spinner color to primary color
st.markdown('''<style>.stSpinner > div > div {border-top-color: #9d03fc;}</style>''',
    unsafe_allow_html=True)
# Design change min height of text input box
st.markdown('''<style>.css-15tx938{min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)
# Design hide top header line
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# Design hide "made with streamlit" footer menu area
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)


# Connect to OpenAI GPT-3, fetch API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")


def gen_mail_contents(email_contents):

    # iterate through all seperate topics
    for topic in range(len(email_contents)):
        input_text = email_contents[topic]
        rephrased_content = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Rewrite the text to be elaborate and polite.\nAbbreviations need to be replaced.\nText: {input_text}\nRewritten text:",
            # prompt=f"Rewrite the text to sound professional, elaborate and polite.\nText: {input_text}\nRewritten text:",
            temperature=0.8,
            max_tokens=len(input_text)*3,
            top_p=0.8,
            best_of=2,
            frequency_penalty=0.0,
            presence_penalty=0.0)

        # replace existing topic text with updated
        email_contents[topic] = rephrased_content.get("choices")[0]['text']
    return email_contents


def gen_mail_format(sender, recipient, style, email_contents):
    # update the contents data with more formal statements
    email_contents = gen_mail_contents(email_contents)
    # st.write(email_contents)  # view augmented contents

    contents_str, contents_length = "", 0
    for topic in range(len(email_contents)):  # aggregate all contents into one
        contents_str = contents_str + f"\nContent{topic+1}: " + email_contents[topic]
        contents_length += len(email_contents[topic])  # calc total chars

    email_final_text = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Write a professional email sounds {style} and includes Content1 and Content2 in that order.\n\nSender: {sender}\nRecipient: {recipient} {contents_str}\n\nEmail Text:",
        # prompt=f"Write a professional sounding email text that includes all of the following contents separately.\nThe text needs to be written to adhere to the specified writing styles and abbreviations need to be replaced.\n\nSender: {sender}\nRecipient: {recipient} {contents_str}\nWriting Styles: motivated, formal\n\nEmail Text:",
        temperature=0.8,
        max_tokens=contents_length*2,
        top_p=0.8,
        best_of=2,
        frequency_penalty=0.0,
        presence_penalty=0.0)

    return email_final_text.get("choices")[0]['text']


def main():

    ## LOGO e Título
    st.image('img/logo.png')  
    
    ## Título e Descrição
    st.markdown('Estudo e Aplicabilidade do Processamento de Linguagem Natural na Análise de Similaridades de Temas STF/STJ'
        'Baseado em um texto de um recurso ou acórdão, quais os temas do STF/STJ possuem similaridade com os textos do processo.'
        'O objetivo é verificar se a ação pode ficar sobrestada aguardando uma decisão superior.')
    st.write('\n')  

    ## Dataset Corpus 
    st.subheader('\nTemas Corpus (Dataset)\n')

    st.markdown('Extração de dados do site oficial do STF e STJ, que consta informações de repercussão, descrição, título e tese dos temas. Dataset composto por 2685 registros (temas).')
    
    df_stf = pd.read_csv("data/dataset_stf.csv")

    st.subheader('\nDataset STF (exemplo)\n')
    st.write(df_stf)
    
    st.subheader('\nAnalise de Similaridades\n')
    
    input_c1 = ''
    input_c2 = ''

    with st.expander("Processos e Recursos - Input Ground Truth", expanded=True):

        col1, col2 = st.columns(2)
        
        filesPdf = [ f for f in os.listdir('test') if f.endswith('.pdf') ]
        filesHtml = [ f for f in os.listdir('test') if f.endswith('.html') ]

        with col1:
            input_c1 = st.selectbox('Escolha o arquivo (Processo .html)', filesHtml, index=None, placeholder="Selecionar")
        
        with col2:
            input_c2 = st.selectbox('Escolha o arquivo (Recurso .pdf)', filesPdf, index=None, placeholder="Selecionar")
        
        col1, col2, col3 = st.columns([5, 5, 7])
        
        with col1:
            input_style = st.selectbox('Escolha a metrica:', ('Cosine Similarity', 'Re-Rank', 'Cross-encoder'), index=0)
    
        with col2:
            input_style = st.selectbox('Escolha o modelo:', ('paraphrase-multilingual-MiniLM-L12-v2', 'distiluse-base-multilingual-cased-v2', 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'multi-qa-distilbert-cos-v1'), index=0)

        with col3:
            st.write("\n")  # add spacing
            st.write("\n")  # add spacing
            st.button("Run model")

    try:
        
        with st.expander("Visualização de Documentos", expanded=True):


            tab1, tab2 = st.tabs(["Processo", "Recurso"])

            col1, col2 = st.columns(2)
            
            with tab1:

                with open(f'test/{input_c1}','r', encoding='latin-1') as f: 
                    html_data = f.read()

                # Show in webpage
                st.components.v1.html(html_data, scrolling=True, height=900)

            with tab2:

                pdf_viewer(f"test/{input_c2}", pages_to_render=[1])        

        with st.expander("Resultados Similaridade de Documentos", expanded=True):

            col1, col2 = st.columns(2)
            
            st.image("img/pca_exemplo.png")
        
    except FileNotFoundError:        
        pass;


if __name__ == '__main__':
    
    # call main function
    main()


import spacy
import csv
import time
from csv import writer

import nltk
from nltk import sent_tokenize
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer 

def similarityCompare(ret, modelo):
    
    try:
    
        # 1. Load a pretrained Sentence Transformer model
        model = SentenceTransformer(modelo, similarity_fn_name=SimilarityFunction.COSINE)
        
        # Two lists of sentences
        temas = [
            'Tema 685 - Extensão da imunidade tributária recíproca ao IPVA de veículos adquiridos por município no regime da alienação fiduciária.',
            'Tema 243 - Termo inicial dos juros moratórios nas ações de repetição de indébito tributário.',
            'Tema 988 - Possibilidade de desoneração do estrangeiro com residência permanente no Brasil em relação às taxas cobradas para o processo de regularização migratória.',
            'Tema 246 - Responsabilidade subsidiária da Administração Pública por encargos trabalhistas gerados pelo inadimplemento de empresa prestadora de serviço.',
            'Tema 247 - Incidência do ISS sobre materiais empregados na construção civil.'
        ]
        
        # Compute embeddings for both lists
        embeddings1 = model.encode(temas)
        embeddings2 = model.encode(ret)

        # Compute cosine similarities
        similarities = model.similarity(embeddings1, embeddings2)       
        
        for idx_j, sentence1 in enumerate(ret):
            
            for idx_i, sentence2 in enumerate(temas):
                
                ## SBERT ==========================================================
                print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
        
    except RuntimeError:        
        pass

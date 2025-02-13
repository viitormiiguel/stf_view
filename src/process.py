
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
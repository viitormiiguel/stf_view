
import os
import sys
import csv
import pandas as pd

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent)) 

def getCorpusSTJ():
    
    temas = pd.read_csv('data/dataset.csv', encoding='utf-8')
    
    ret = list(temas['Titulo'].values)
    
    return ret

def getCorpusSTF():
    
    temas = pd.read_csv('data/dataset.csv', encoding='utf-8')
    
    ret = list(temas['Titulo'].values)
    
    return ret
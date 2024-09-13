#importa algumas bibliotecas utilizadas
import pandas as pd
from google.colab import drive

# Monta o Google Drive para acessar os arquivos
drive.mount('/content/drive')

# Carrega os dados de treinamento e teste a partir dos arquivos CSV
DBTeste = pd.read_csv('/content/drive/MyDrive/Hackathon_SEMAC/test.csv')
DBTreino = pd.read_csv('/content/drive/MyDrive/Hackathon_SEMAC/train.csv')

# Remove linhas com valores ausentes na coluna 'feeling'
DBTreino = DBTreino.dropna(subset=['feeling'])

# Converte as colunas 'text' para string e 'feeling' para inteiro
DBTreino['text'] = DBTreino['text'].astype(str)
DBTreino['feeling'] = DBTreino['feeling'].astype(int)

"""
Pré-processamento dos dados
Define e aplica funções para limpar e normalizar o texto.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Função para remover pontuação e caracteres especiais do texto
def remover_pontuacao_caracteres(texto):
    texto_limpo = re.sub(r'[^A-Za-zÀ-ÿ\s]', '', texto)
    return texto_limpo

# Aplica a função para remover pontuação e caracteres especiais nos dados de treino e teste
DBTreino['text'] = DBTreino['text'].apply(remover_pontuacao_caracteres)
DBTeste['text'] = DBTeste['text'].apply(remover_pontuacao_caracteres)

# Baixa os recursos necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Tokeniza o texto em palavras
DBTeste['text'] = DBTeste['text'].apply(word_tokenize)
DBTreino['text'] = DBTreino['text'].apply(word_tokenize)

# Define a lista de stop words em inglês
stop_words = set(stopwords.words('english'))

# Função para remover stop words do texto
def remover_stopwords(texto):
    palavras_sem_stopwords = [palavra for palavra in texto if palavra.lower() not in stop_words]
    return ' '.join(palavras_sem_stopwords)

# Aplica a função para remover stop words nos dados de treino e teste
DBTreino['text'] = DBTreino['text'].apply(lambda x: remover_stopwords(x))
DBTeste['text'] = DBTeste['text'].apply(lambda x: remover_stopwords(x))

# Função para converter tags POS do NLTK para o formato esperado pelo lematizador
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Função para lematizar o texto com base nas tags POS
def lemmatizar_texto(texto):
    palavras = word_tokenize(texto)
    pos_tags = nltk.pos_tag(palavras)
    palavras_lemmatizadas = [lemmatizer.lemmatize(palavra, get_wordnet_pos(pos)) for palavra, pos in pos_tags]
    return ' '.join(palavras_lemmatizadas)

# Inicializa o lematizador
lemmatizer = WordNetLemmatizer()

# Aplica a lematização nos dados de treino e teste
DBTreino['text'] = DBTreino['text'].apply(lemmatizar_texto)
DBTeste['text'] = DBTeste['text'].apply(lemmatizar_texto)

"""
Representação do texto
Converte o texto para uma representação numérica usando TF-IDF.

"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Inicializa o vetorizador TF-IDF com n-gramas e limites de características
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)

# Ajusta o vetorizador aos dados de treino e transforma os dados de treino e teste
X = vectorizer.fit_transform(DBTreino['text'])
y = DBTreino['feeling']
X_teste = vectorizer.transform(DBTeste['text'])

"""
Aplicação do algoritmo
Treina um classificador de Regressão Logística e realiza previsões no conjunto de teste.

"""

from sklearn.linear_model import LogisticRegression

# Inicializa o classificador de Regressão Logística
clf = LogisticRegression()

# Treina o classificador com os dados de treino
clf.fit(X, y)

# Faz previsões nos dados de teste
y_pred = clf.predict(X_teste)

# Adiciona as previsões como uma nova coluna no DataFrame de teste
DBTeste['Previsoes'] = y_pred

# Remove a coluna de texto do DataFrame de teste
DBTeste = DBTeste.drop(columns=['text'])

# Salva o DataFrame de teste com as previsões em um arquivo CSV
DBTeste.to_csv('dados.csv', index=False)
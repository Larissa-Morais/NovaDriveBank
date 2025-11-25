import pandas as pd #biblioteca para manipulação de dados
import tensorflow as tf #para construção de modelos de deep learning
import numpy as np #biblioteca para operações numéricas
import random as python_random #biblioteca para operações matemáticas
import joblib #salvar e carregar modelos treinados
import const

from datetime import datetime #manipulação de datas
from sklearn.preprocessing import StandardScaler, LabelEncoder #pré-processamento de dados
from sklearn.model_selection import train_test_split #divisão dos dados em treino e teste
from sklearn.preprocessing import classification_report, confusion_matrix #avaliação do modelo
from sklearn.preprocessing import RandomForestClassifier #modelo de classificação
from sklearn.feature_selection import RFE #seleção de características

from utils import *

#definição da semente para geração de resultados pseudoaleatórios
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

#dados brutos
df = get_data_from_db(const.consulta_sql)

#conversão de tipo númericos
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)

#tratamento de nulos
substitui_nulos(df) #trata os nulos do df todo (se encontra em utils.py)

#trata erros de digitação
profissoes_validas = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador',
                       'Dentista', 'Engenheiro', 'Médico', 'Programador']
corrigir_erros_digitacao(df, 'profissao', profissoes_validas) #trata erros de digitação na coluna 'profissao' (se encontra em utils.py)

#trata outliers
df = tratar_outliers(df, 'tempoprofissao', 0, 70) #trata outliers na coluna 'tempoprofissao' (se encontra em utils.py)
df = tratar_outliers(df, 'idade', 0, 110) #trata outliers na coluna 'idade' (se encontra em utils.py)

#feature engineering para cálculos entre colunas
df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem'] #cálculo da proporção entre valor solicitado e valor total do bem
df['proporcaosolicitadototal'] = df['proporcaosolicitadototal'].astype(float)

#dividir dados entre treino e teste
X = df.drop('classe', axis=1) #remove a coluna 'classe de df pois é a variavel que deve prever
y = df['classe'] #coloca a coluna 'classe' em y, que é a variável alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) #80% treino, 20% teste

#normalização
X_test = save_scalers(X_test, ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                               'valorsolicitado', 'valortotalbem', 'proporcaosolitcitadototal'])
X_train = save_scalers(X_train, ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                               'valorsolicitado', 'valortotalbem', 'proporcaosolitcitadototal'])

#codificação
mapeamento = {'ruim': 0, 'bom': 1} 
y_train =np.array([mapeamento[item] for item in y_train]) #converte os valores categóricos em numéricos
y_test =np.array([mapeamento[item] for item in y_test]) 
#aplica o label encoding nas colunas categóricas
X_train = save_encoders(X_train, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])
X_test = save_encoders(X_test, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])

#Seleção de atributos
model = RandomForestClassifier()
#RFE é uma técnica de seleção de características que remove recursivamente as características menos importantes com base na importância atribuída pelo modelo
selector = RFE(model, n_features_to_select=10, step=1) #seleciona as 10 características mais importantes
selector = selector.fit(X_train, y_train)
#transforma os dados
X_train = selector.transform(X_train) #aplica a seleção de características nos dados de treino
X_test = selector.transform(X_test) #aplica a seleção de características nos dados de teste
joblib.dump(selector, './objects/selector.joblib') #salva o seletor treinado em uma pasta
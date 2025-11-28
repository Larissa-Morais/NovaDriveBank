import pandas as pd #biblioteca para manipulação de dados
import tensorflow as tf #para construção de modelos de deep learning
import numpy as np #biblioteca para operações numéricas
import random as python_random #biblioteca para operações matemáticas
import joblib #salvar e carregar modelos treinados
import const

from datetime import datetime #manipulação de datas
from sklearn.preprocessing import StandardScaler, LabelEncoder #pré-processamento de dados
from sklearn.model_selection import train_test_split #divisão dos dados em treino e teste
from sklearn.metrics import classification_report, confusion_matrix #avaliação do modelo
from sklearn.ensemble import RandomForestClassifier #modelo de classificação
from sklearn.feature_selection import RFE #seleção de características

from utils import *

#-- pré-processamento de dados

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
y = df['classe']#variável alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) #80% treino, 20% teste

#normalização
X_test = save_scalers(X_test, ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                               'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal'])
X_train = save_scalers(X_train, ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                               'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal'])

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
save_object(selector, "selector.joblib") #salva o seletor de características (se encontra em utils.py)

#-- Modelagem de dados
#keras é uma API de alto nível que facilita a criação e treinamento de modelos de deep learning
model = tf.keras.Sequential([ #as camadas são empilhadas lineralmente. Cada camada processa a saída da anterior
    tf.keras.layers.Dense(128, activation = 'relu', input_shape = (X_train.shape[1],)), #camada densa com 128 neurônios e função de ativação ReLU
    tf.keras.layers.Dropout(0.3),                                                        #RELU: valores negativos são descartados (viram 0), e apenas valores positivos seguem para a próxima camada.
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.3), #adiciona dropout para evitar overfitting(desconexão aleatória de 30% dos neurônios)
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation = 'sigmoid') #camada de saída com 1 neurônio e função de ativação sigmoide (para classificação binária)
])
#Configurando o otimizador
#Adam é um otimizador que ajusta os pesos da RN durante o treino
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) #taxa de aprendizado de 0.001 para controlar a velocidade de ajuste dos pesos

#Compilando o modelo
model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']) #compila o modelo com a função de perda binária e métrica de acurácia

#Treinando o modelo
class_weight = {0: 2, 1: 1}
model.fit(X_train, 
          y_train, 
          validation_split = 0.3, #30% dos dados de treino para validação
          epochs = 200,  #número de vezes que o modelo verá todo o dataset
          batch_size = 10, #quantidade de amostras processadas antes de atualizar os pesos do modelo
          verbose = 1, #exibe o progresso do treinamento
          class_weight = class_weight
          
)  
#Salva modelo
model.save('model/novadrivebank_model.keras')

#Previsões
y_pred = model.predict(X_test) #faz previsões utilizando os dados de teste para prever a classe y
y_pred = (y_pred > 0.5).astype(int) #converte as probabilidades em classes binárias (0 ou 1) com threshold(valor limite da classe) de 0.5

#Avaliação do modelo
print('Avaliação do Modelo nos dados de teste: ')
model.evaluate(X_test, y_test)

#Métricas de classificação
print('Relatório de Classificação: ')
print(classification_report(y_test, y_pred)) #exibe métricas detalhadas de precisão, recall e F1-score

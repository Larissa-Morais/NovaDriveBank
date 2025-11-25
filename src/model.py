import pandas as pd #biblioteca para manipulação de dados
import tensorflow as tf #para construção de modelos de deep learning
import numpy as np #biblioteca para operações numéricas
import random as python_random #biblioteca para operações matemáticas
import joblib #salvar e carregar modelos treinados

from datetime import datetime #manipulação de datas
from sklearn.preprocessing import StandardScaler, LabelEncoder #pré-processamento de dados
from sklearn.preprocessing import train_test_split #divisão dos dados em treino e teste
from sklearn.preprocessing import classification_report, confusion_matrix #avaliação do modelo
from sklearn.preprocessing import RandomForestClassifier #modelo de classificação
from sklearn.feature_selection import RFE #seleção de características

from utils import *

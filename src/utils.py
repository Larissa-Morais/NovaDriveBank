#funções utilitárias 

import pandas as pd #biblioteca para manipulação de dados
import yaml #biblioteca para manipulação de arquivos yaml
import psycopg2 #biblioteca para conectar ao banco de dados postgresql
from fuzzywuzzy import process #biblioteca para correspondência de strings
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib #biblioteca para salvar e carregar modelos treinados
import const #importa o arquivo de constantes

#função para obter dados do banco de dados
def get_data_from_db(sql_query): 
    try:
        with open('config.yaml', 'r') as file: 
            config = yaml.safe_load(file)    #carrega as configurações do arquivo yaml

            con = psycopg2.connect( 
                dbname=config['database_config']['dbname'],
                user=config['database_config']['user'],
                password=config['database_config']['password'],
                host=config['database_config']['host']
            ) #conecta ao banco de dados
            
            cursor = con.cursor() #cria um cursor para executar comandos SQL
            cursor.execute(sql_query) #executa a consulta SQL
            
            #converte o resultado da consulta em um DataFrame do pandas   
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            
    finally:
        if 'cursor' in locals(): 
            cursor.close()
        if 'con' in locals():
            con.close()

    return df

#função para substituir valores nulos
def substitui_nulos(df):
    for coluna in df.columns: 
        if df[coluna].dtype == 'object': #verifica se a coluna tem dados categóricos
            moda = df[coluna].mode()[0] #calcula a moda da coluna
            df[coluna].fillna(moda, inplace=True) #substitui os valores nulos pela moda
        else:
            mediana = df[coluna].median() #calcula a mediana da coluna
            df[coluna].fillna(mediana, inplace=True)
            




            


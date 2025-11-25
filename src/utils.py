import pandas as pd #biblioteca para manipulação de dados
import yaml #biblioteca para manipulação de arquivos yaml
import psycopg2 #biblioteca para conectar ao banco de dados postgresql
import joblib #biblioteca para salvar e carregar modelos treinados
import const #importa o arquivo de constantes

from fuzzywuzzy import process #biblioteca para correspondência de strings
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    
#função para corrigir erros de digitação
def corrigir_erros_digitacao(df, coluna, lista_valida):
    for i, valor in enumerate(df[coluna]): #itera sobre os valores da coluna
        valor_str = str(valor) if pd.notnull(valor) else valor #converte o valor para string se não for nulo
        
        if valor_str not in lista_valida and pd.notnull(valor_str): #verifica se o valor não está na lista válida e não é nulo
            correcao = process.extractOne(valor_str, lista_valida)[0] #encontra a melhor correspondência na lista válida
            df.at[i, coluna] = correcao #substitui o valor incorreto pela correção encontrada

#função para tratamento de outliers
def tratar_outliers(df, coluna, minimo, maximo):
    mediana = df[(df[coluna] >= minimo) & (df[coluna] <= maximo)][coluna].median() #calcula a mediana dos valores 
    df[coluna] = df[coluna].apply(lambda x: mediana if x < minimo or x > maximo else x) #substitui os outliers pela mediana
    return df

#função de normalização 
def save_scalers(df, nome_colunas): 
    for nome_coluna in nome_colunas: #itera sobre as colunas fornecidas 
        scaler = StandardScaler() #StandardScaler é um método de normalização que padroniza os dados para terem média 0 e desvio padrão 1
        df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]]) #ajusta e transforma os dados da coluna
        joblib.dump(scaler, f"./objects/scaler{nome_coluna}.joblib") #salva o scaler treinado em uma pasta
    return df

                




            


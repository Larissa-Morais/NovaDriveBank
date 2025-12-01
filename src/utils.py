import os #biblioteca para manipulação de caminhos de arquivos e diretórios
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
        # Caminho dinâmico para o config.yaml
        base_path = os.path.dirname(__file__)  # diretório do utils.py
        config_path = os.path.join(base_path, "..", "config.yaml")
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

        # Carrega as configurações do arquivo YAML
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Conecta ao banco de dados
        con = psycopg2.connect(
            dbname=config['database_config']['dbname'],
            user=config['database_config']['user'],
            password=config['database_config']['password'],
            host=config['database_config']['host']
        )

        cursor = con.cursor()
        cursor.execute(sql_query)

        # Converte o resultado da consulta em DataFrame
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    except Exception as e:
        raise RuntimeError(f"Erro ao buscar dados do banco: {e}")

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
            df[coluna] = df[coluna].fillna(moda) #substitui os valores nulos pela moda
        else:
            mediana = df[coluna].median() #calcula a mediana da coluna
            df[coluna] = df[coluna].fillna(mediana)
    
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
    # Caminho para salvar os scalers
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "objects/preprocess_result"))
    os.makedirs(output_dir, exist_ok=True)  # cria a pasta se não existir

    for nome_coluna in nome_colunas:
        if nome_coluna in df.columns:
            scaler = StandardScaler()
            df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])
            joblib.dump(scaler, os.path.join(output_dir, f"scaler_{nome_coluna}.joblib"))
        else:
            print(f"A coluna '{nome_coluna}' não existe no DataFrame.")
    return df

#função label encoding
def save_encoders(df, nome_colunas):
    # Caminho para salvar os encoders
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "objects/preprocess_result"))
    os.makedirs(output_dir, exist_ok=True)  # cria a pasta se não existir

    for nome_coluna in nome_colunas:
        if nome_coluna in df.columns:
            encoder = LabelEncoder()
            df[nome_coluna] = encoder.fit_transform(df[nome_coluna])
            joblib.dump(encoder, os.path.join(output_dir, f"labelencoder_{nome_coluna}.joblib"))
        else:
            print(f"A coluna '{nome_coluna}' não existe no DataFrame.")
    return df

#função para salvar objetos como modelos treinados, scalers ou encoders
def save_object(obj, filename):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "objects/preprocess_result"))
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(obj, os.path.join(output_dir, filename))


                




            


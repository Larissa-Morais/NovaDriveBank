#retorna a consulta sql para ser utilizada no sistema
import pandas as pd
import yaml
import psycopg2 #biblioteca para conectar ao banco de dados postgresql

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

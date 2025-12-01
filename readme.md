# 🧠 NovaDriveBank

Projeto do curso **Bootcamp Inteligência Artificial: Construa um projeto real da Udemy**, com o intuito de **aplicar técnicas de análise e modelagem de dados em um contexto prático e realista**.

---

## 🧩 Tecnologias Principais

- Python 3.10+
- Pandas
- Scikit learn
- Numpy
- Tensorflow
- Matplotlib
- Seaborn
- PyYAML

## 📁 Estrutura do Projeto
```bash
NOVADRIVEBANK/
├── model/  # pasta onde o modelo será salvo
├── notebook/model.ipynb   # arquivo jupiter para visualização dos dados
├── objects/preprocess_results/   # pasta onde o    pré-processamento será armazenado
|
├── src/ # Código-fonte principal
│ ├── const.py # Consulta sql
│ ├── eda.py # Análise exploratória de dados
| ├── model.py   # pré-processamento e treinamento do modelo
│ └── utils.py # Funções auxiliares
|
├── config.yaml # Arquivo de configuração
└── requirements.txt # Dependências do projeto
```
Antes de tudo, crie uma pasta na raiz do projeto chamada:

- **objects/preprocess_results/**

E crie outra pasta na raiz do projeto chamada:

- **model/**

---
## ⚙️ Requisitos

**Antes de rodar, crie um ambiente virtual:**
```
python -m venv .venv
```
**Acesse o ambiente virtual:**
```
.venv\Scripts\activate
 ```
**Agora instale as dependências do projeto:**

```bash
pip install -r requirements.txt
```

## 🚀 Execução
Para iniciar a Análise Exploratória de Dados (EDA) e, em seguida, treinar o modelo, execute os scripts na seguinte ordem:

- **Análise Exploratória de Dados(EDA)**
Execute o script de EDA para processar os dados e gerar visualizações dos dados antes do pré-processamento:

```
python src/eda.py
```
**Observação:** O eda.py vai gerar gráficos, que serão exibidos na tela, e no final gerará um resumo estatístico no terminal.

Após a visualização dos gráficos, foi notado alguns pontos para trabalhar no pré-processamento, como tratamento de outliers, dados nulos, dados categóricos e erros de digitação.

Foi possivel ter essa visão dos dados graças a EDA (análise exploratória de dados) 

- **Treinamento e Avaliação do Modelo**
Agora podemos executar o script de modelagem, que fará o carregamento, pré-processamento e treinamento do modelo.

``` 
python src/model.py
```
**Observação:** todo o pré-processamento será salvo na pasta **objects/preprocess_results** e o modelo será salvo na pasta **model/**.  

Você poderá visualizar como foi feito o pré-processamento dos dados executando as cédulas do arquivo jupiter, que se encontra dentro da pasta notebook

## 📊 Resultados Esperados
Na execução do arquivo eda.py, é esperado ter a visualização de gráficos de barras, boxplot e histogramas.
Para assim, ter a visualização dos principais atributos do conjunto de dados.

Na execução do arquivo model.py, é esperado ter a visualização de métricas como:
- **Acurácia:** ~77%
- **Recall(Classe 0)**: ~64%
- **Recall(classe 1):** ~84%
- **Previsão(classe 0):** ~70%
- **Previsão(classe 1):** ~80%
- **F1-Score(classe 0):** ~67%
- **F1-Score(classe 1):** ~82%

## 👩🏽‍💻Autoria
- Larissa Morais
- Bootcamp Inteligência Artificial: Construa um Projeto Real | Udemy
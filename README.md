# E-GovAnalytics_project

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/vetores-gratis/analise-do-mercado-de-acoes-com-grafico_23-2148584739.jpg?w=1380&t=st=1692993711~exp=1692994311~hmac=4709c7ed54395f790679e3e0c0700aa9d9baa3d00f34e5c96dd275b7c8387c59)

# Projeto na prática de Ciência de Dados: E-Gov Analytics

O projeto "E-Gov Analytics" é um componente fundamental da formação oferecida pela @Data Science Academy para aspirantes a cientistas de dados. Este projeto destaca-se como uma oportunidade singular para aplicar os conhecimentos teóricos em um contexto real e relevante. O enfoque recai sobre a Ciência de Dados aplicada ao governo eletrônico, ou E-Gov, onde o uso inteligente dos dados desempenha um papel crucial na tomada de decisões informadas. A Ciência de Dados emerge como uma disciplina essencial para extrair insights e conhecimentos a partir de volumes vastos de dados, possibilitando a transformação de informações em ações tangíveis. No contexto do governo eletrônico, essa transformação é ainda mais significativa, já que os dados governamentais estão em constante expansão, variando de estatísticas demográficas a indicadores econômicos. Nesse cenário, o projeto "E-Gov Analytics" destaca-se por sua abordagem prática. Ele fornece aos alunos da formação científica de dados a oportunidade de lidar com dados reais e relevantes, enfrentando os desafios que espelham situações do mundo real. Os participantes têm a oportunidade de aplicar técnicas avançadas de análise de dados para descobrir padrões, tendências e relações ocultas, que podem guiar políticas públicas e decisões governamentais mais informadas e eficazes. Ao longo do projeto, os alunos mergulham na coleta, limpeza, análise e interpretação de dados específicos do governo eletrônico. Isso os expõe a uma variedade de habilidades práticas, desde o uso de ferramentas de programação, como Python e R, até a aplicação de algoritmos de aprendizado de máquina e visualização de dados. A abordagem hands-on oferecida pelo projeto não apenas solidifica o conhecimento técnico, mas também aprimora habilidades críticas, como resolução de problemas, colaboração e comunicação eficaz de resultados complexos. 
Em suma, o projeto "E-Gov Analytics" dentro da formação cientista de dados da @Data Science Academy preenche a lacuna entre a teoria da Ciência de Dados e sua implementação prática no contexto do governo eletrônico. Ele não só amplia a compreensão dos alunos sobre como os princípios da Ciência de Dados podem ser aplicados em cenários do mundo real, mas também os capacita a desempenhar um papel significativo na condução de decisões embasadas em dados no âmbito governamental.

## Definição projeto
O papel do governo no crescimento econômico tem sido um problema desde há muito tempo, com a percepção de que, para um desenvolvimento sustentável e uma produção eficiente, o papel do governo nas políticas econômicas deve ser reduzido. Dado este cenário fiscal, é necessário identificar a relação entre o PIB (Produto Interno Bruto) e as despesas do governo nos setores de saúde e educação e como se relacionam com o crescimento do PIB na economia de um país. Para este trabalho não vamos considerar um país específico a fim de evitar qualquer tipo de polêmica ou viés. Nosso objetivo aqui é estudar e aplicar análise de dados. Portanto, este projeto se aplica a qualquer país. Principais objetivos:

1. Em que ano o país teve um alto crescimento do PIB Per Capita?

2. Qual ano teve as maiores despesas com saúde?

3. Qual ano teve os maiores gastos com educação?

4. Como a receita do governo se correlaciona com a despesa do governo?

5. Como a educação e a saúde se correlacionam com o crescimento do PIB?

6. Como os gastos do governo com educação e saúde afetam o crescimento do PIB?

7. Com base na análise quais são as recomendações aos governantes e gestores?

## Fonte de Dados

Leia o manual em pdf no curso bônus de E-Gov Analytics.

Para este trabalho usamos como fonte de dados a API fornecida pelo Banco Mundial no endereço abaixo: 

http://api.worldbank.org/v2/en/country

A API permite extrair dados de diversos países simplesmente colocando o código do país ao final do endereço acima.
Extraímos dados de um dos países no período de 2003 a 2019 e ajustamos os nomes das colunas. Nosso objetivo aqui é estudar análise de dados e o processo a seguir no curso poderá ser aplicado a qualquer país. O dataset e o dicionário de dados serão fornecidos a você. 

O repositório de dados abertos do Banco Mundial oferece fonte valiosa de dados ao redor do mundo:

https://data.worldbank.org/

## Citação
@Data Science Academy.

## Stack utilizada

**Programação** Python, R.

**Machine learning**: Scikit-learn.

**Leitura CSV**: Pandas.

**Análise de dados**: Seaborn, Matplotlib.

**Modelo machine learning - Regressão linear**: LGBM Regressor, CatBoost Regressor, Regression linear.


## Variáveis de Ambiente

Para criar e gerenciar ambientes virtuais (env) no Windows, você pode usar a ferramenta venv, que vem com as versões mais recentes do Python (3.3 e posteriores). Aqui está um guia passo a passo de como criar um ambiente virtual no Windows:

1 Abrir o Prompt de Comando: Pressione a tecla Win + R para abrir o "Executar", digite cmd e pressione Enter. Isso abrirá o Prompt de Comando do Windows.

2 Navegar até a pasta onde você deseja criar o ambiente virtual: Use o comando cd para navegar até a pasta onde você deseja criar o ambiente virtual. Por exemplo:

`cd caminho\para\sua\pasta`

3 Criar o ambiente virtual: Use o comando python -m venv nome_do_seu_env para criar o ambiente virtual. Substitua nome_do_seu_env pelo nome que você deseja dar ao ambiente. 

`python -m venv myenv`

4 Ativar o ambiente virtual: No mesmo Prompt de Comando, você pode ativar o ambiente virtual usando o script localizado na pasta Scripts dentro do ambiente virtual. Use o seguinte comando:

`nome_do_seu_env\Scripts\activate`

5 Desativar o ambiente virtual: Para desativar o ambiente virtual quando você não precisar mais dele, simplesmente digite deactivate e pressione Enter no Prompt de Comando.

`myenv\Scripts\activate`

Obs: Lembre-se de que, uma vez que o ambiente virtual está ativado, todas as instalações de pacotes usando pip serão isoladas dentro desse ambiente, o que é útil para evitar conflitos entre diferentes projetos. Para sair completamente do ambiente virtual, feche o Prompt de Comando ou digite deactivate. Lembre-se também de que essas instruções pressupõem que o Python esteja configurado corretamente em seu sistema e acessível pelo comando python. Certifique-se de ter o Python instalado e adicionado ao seu PATH do sistema. Caso prefira, você também pode usar ferramentas como o Anaconda, que oferece uma maneira mais abrangente de gerenciar ambientes virtuais e pacotes em ambientes de desenvolvimento Python.

## Pacote no anaconda
Para instalar um pacote no Anaconda, você pode usar o gerenciador de pacotes conda ou o gerenciador de pacotes Python padrão, pip, que também está disponível dentro dos ambientes do Anaconda. Aqui estão as etapas para instalar um pacote usando ambos os métodos:

## Usando o Conda
1 Abra o Anaconda Navigator ou o Anaconda Prompt, dependendo da sua preferência.

2 Para criar um novo ambiente (opcional, mas recomendado), você pode executar o seguinte comando no Anaconda Prompt (substitua nome_do_seu_ambiente pelo nome que você deseja dar ao ambiente):

`conda create --name nome_do_seu_ambiente`

3 Ative o ambiente recém-criado com o seguinte comando (substitua nome_do_seu_ambiente pelo nome do ambiente que você criou):

`conda create --name nome_do_seu_ambiente`

4 Para instalar um pacote específico, use o seguinte comando (substitua nome_do_pacote pelo nome do pacote que você deseja instalar):

`conda install nome_do_pacote`


## Instalação
Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo Modelo regressão linear simples - CatBoost Regressor 

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Exemplo de criação do DataFrame com os dados
data = {
    "total_gastos_educ_saude": [100, 150, 200, 250, 300, 350, 400, 450, 500],
    "pib_cresc_anual%": [2.0, 3.5, 2.8, 1.0, 0.5, 1.8, 1.2, 2.5, 3.0]
}

df = pd.DataFrame(data)

# Separando os dados em variáveis preditoras (X) e variável dependente (y)
X = df[["pib_cresc_anual%"]]
y = df["total_gastos_educ_saude"]

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo CatBoostRegressor
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=100)

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Plotando os resultados
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Previsões')
plt.xlabel('PIB Crescimento Anual (%)')
plt.ylabel('Total Gastos Educação e Saúde')
plt.title('Previsão usando CatBoostRegressor')
plt.legend()
plt.show()

# Calculando o erro médio quadrado (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Erro Médio Quadrado (MSE):", mse)
```

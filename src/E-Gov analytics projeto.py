#!/usr/bin/env python
# coding: utf-8

# # <font color='green'>Data Science Academy</font>
# # <font color='green'>Curso Bônus - E-Gov Analytics</font>
# 
# ## <font color='green'>Projeto E-Gov Analytics</font>
# ## <font color='green'>Como Aumento do PIB e Gastos do Governo Impactam os Setores de Saúde e Educação</font>
# 
# ![title](imagens/projeto.png)

# ## Definição do Problema
# 
# O papel do governo no crescimento econômico tem sido um problema desde há muito
# tempo, com a percepção de que, para um desenvolvimento sustentável e uma produção
# eficiente, o papel do governo nas políticas econômicas deve ser reduzido.
# Dado este cenário fiscal, é necessário identificar a relação entre o PIB (Produto Interno
# Bruto) e as despesas do governo nos setores de saúde e educação e como se relacionam com o
# crescimento do PIB na economia de um país.
# Para este trabalho não vamos considerar um país específico a fim de evitar qualquer
# tipo de polêmica ou viés. Nosso objetivo aqui é estudar e aplicar análise de dados. Portanto,
# este projeto se aplica a qualquer país.
# Principais objetivos:
# 
# 1. Em que ano o país teve um alto crescimento do PIB Per Capita?
# 
# 2. Qual ano teve as maiores despesas com saúde?
# 
# 3. Qual ano teve os maiores gastos com educação?
# 
# 4. Como a receita do governo se correlaciona com a despesa do governo?
# 
# 5. Como a educação e a saúde se correlacionam com o crescimento do PIB?
# 
# 6. Como os gastos do governo com educação e saúde afetam o crescimento do PIB?
# 
# 7. Com base na análise quais são as recomendações aos governantes e gestores?

# ## Fonte de Dados
# 
# Leia o manual em pdf no curso bônus de E-Gov Analytics.
# 
# Para este trabalho usamos como fonte de dados a API fornecida pelo Banco Mundial no
# endereço abaixo: 
# 
# http://api.worldbank.org/v2/en/country
# 
# A API permite extrair dados de diversos países simplesmente colocando o código do país ao final do endereço acima.
# Extraímos dados de um dos países no período de 2003 a 2019 e ajustamos os nomes das colunas. Nosso objetivo aqui é estudar análise de dados e o processo a seguir no curso poderá ser aplicado a qualquer país. O dataset e o dicionário de dados serão fornecidos a você. 
# 
# O repositório de dados abertos do Banco Mundial oferece fonte valiosa de dados ao redor do mundo:
# 
# https://data.worldbank.org/

# ## Dicionário de Dados
# 
# 
# **Variável Descrição**
# 
# - ano_coleta =  Ano da coleta dos dados
# 
# 
# - despesas_educ_percent = Despesas com educação secundária como uma porcentagem das despesas do governo
# 
# 
# - despesas_saude_per_capita = Despesas atuais com saúde per capita na taxa atual em dólares dos EUA
# 
# 
# - despesas_educ_total = Despesas do governo com educação, total como uma porcentagem das despesas do governo
# 
# 
# - despesas_saude_%pib = Despesas atuais com saúde como uma porcentagem do PIB
# 
# 
# - receita_trib_%pib = Receita tributária como porcentagem do PIB
# 
# 
# - receita_excl_doa_%pib = Receita excluindo doações como porcentagem do PIB
# 
# 
# 
# - ibrd_e_ida = Empréstimos do BIRD e créditos da AID como DOD, US atuais
# 
# 
# - pop_cresc_anual% = Crescimento populacional como porcentagem anual
# 
# 
# - pib_cresc_per_capita_%pib = Crescimento do PIB per capita como porcentagem anual pib_deflator Deflator do PIB, ano base varia por país
# 
# 
# - domestic_saude_despesas_%pib = Despesas domésticas do governo geral com saúde como uma porcentagem do PIB
# 
# 
# - pib_cresc_anual% = Crescimento do PIB como uma porcentagem anual
# 
# 
# - pib_cor_us PIB em US correntes
# 
# 
# - despesa_nac_bruta_% = pib Despesa nacional bruta como uma porcentagem do PIB pib_moeda_local PIB em moeda local

# ## Instalando e Carregando os Pacotes

# In[1]:


# Para atualizar um pacote, execute o comando abaixo no terminal ou prompt de comando:
# pip install -U nome_pacote

# Para instalar a versão exata de um pacote, execute o comando abaixo no terminal ou prompt de comando:
# !pip install nome_pacote==versão_desejada

# Depois de instalar ou atualizar o pacote, reinicie o jupyter notebook.

# Instala o pacote watermark. 
# Esse pacote é usado para gravar as versões de outros pacotes usados neste jupyter notebook.
#!pip install -q -U watermark


# In[117]:


# Imports

# Importando bibliotecas de manipulação dados
import numpy as np
import pandas as pd

# Importando bibliotecas de visualização de dados
import matplotlib.pyplot as plt
import seaborn as sb

# Importando bibliotecas de modelo regressão linear
import sklearn
from sklearn.linear_model import LinearRegression

# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Rafael Gallo - Bibliotecas geral" --iversions')


# ## Carregando e Compreendendo os Dados

# In[3]:


# Carrega o dataset
df = pd.read_excel('dados/dataset.xlsx')


# In[4]:


# Visualizando dados geral
df.head()


# In[5]:


# Shape
df.shape


# # Análise Exploratória

# In[6]:


# Tipos de dados
df.dtypes


# In[7]:


# Resumo estatístico
df.describe().T


# In[8]:


# Resumo estatístico
df.describe()


# In[9]:


# Verificando valores ausentes
df.isnull().sum()


# ## Limpeza e Processamento dos Dados
# 
# O que fazer nesta etapa?

# ### Tratamento de Valores Ausentes com Backfilling e Interpolação

# In[10]:


# Limpando valores ausentes através de backfilling e interpolação
# Limpando valores ausentes através de backfilling e interpolação
df['receita_trib_%pib'].fillna(method = 'bfill', inplace = True)
df['receita_excl_doa_%pib'].fillna(method = 'bfill', inplace = True)
df = df.interpolate()
df


# In[11]:


# Verificando valores ausentes
df.isnull().sum()


# In[12]:


# Shape
df.shape


# In[13]:


# Vamos remover a coluna gdp em US$ uma vez que já o temos na moeda local.
df = df.drop(columns = ['pib_cor_us$'])


# In[14]:


# Shape
df.shape


# In[15]:


# Arrendodamos os valores das colunas decimais para 4 casas decimais
df = df.round(4)


# In[16]:


# Visualiza os dados
df.head()


# In[17]:


# Salvamos o dataset limpo
df.to_csv('dados/df_limpo.csv', index = False)


# # Part 1 - Análise de dados

# In[18]:


import seaborn as sns


# In[19]:


# Carregando o dataset limpo
data = pd.read_csv('dados/df_limpo.csv')


# In[20]:


# Visualiza os dados
data.head()


# ## <font color = "green">1- Em Que Ano o País Teve um Alto Crescimento do PIB Per Capita?</font>

# In[21]:


plt.figure(figsize=(15.5, 5))
sns.barplot(data=data, x="ano_coleta", y="pib_cresc_per_capita_%pib", palette='pastel')


# **Resposta:** Pelo gráfico acima vemos que foi no ano de 2010 o maior crescimento do PIB Per Capita no período analisado.

# ## <font color = "green">2- Qual Ano Teve as Maiores Despesas com Saúde?</font>

# In[22]:


# Calculando o gasto nacional bruto em relação ao PIB na moeda local
data['gasto_nac_bruto'] = data["despesa_nac_bruta_%pib"] / 100 * data["pib_moeda_local"]


# In[29]:


# Calculamos o gasto total em saúde e criamos uma nova variável
data["gasto_saude"] = data["domestic_saude_despesas_%pib"] / 100 * data["gasto_nac_bruto"]
data.head()


# In[31]:


# Agora respondemos a pergunta
data.plot(x = "ano_coleta", 
        y = ["gasto_educ"], 
        kind = "bar", 
        figsize = (15.5,5), 
        color = 'green')


# Resposta: 2019 foi o ano com maior volume de gastos em saúde.

# ## <font color = "green">3- Qual Ano Teve os Maiores Gastos com Educação?</font>

# In[32]:


# Visualiza os dados
data.head()


# In[33]:


# Calculamos o gasto em educação e criamos uma variável
data["gasto_educ"] = data["despesas_educ_total"] / 100 * data["gasto_nac_bruto"]
data.head()


# In[37]:


plt.figure(figsize=(15.5, 5))
sns.barplot(x="ano_coleta", y="gasto_saude", data = data)


# Resposta: 2019 foi o ano com maior volume de gastos em saúde.

# ## <font color = "green">4- Como a Receita do Governo se Correlaciona com a Despesa do Governo?</font>

# In[41]:


# Calcula a correlação entre as variáveis no conjunto de dados usando o método de Pearson
data2 = data.corr(method = 'pearson')

# Definindo o tamanho do gráfico
plt.figure(figsize = (15.5,7)) 

# Visualiza a correlação em um mapa de calor (heatmap)
# https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
sb.heatmap(data2, 
           xticklabels = data2.columns,
           yticklabels = data2.columns,
           cmap = 'Oranges',
           annot = True,
           linewidth = 0.8)


# A partir dos coeficientes da matriz de correlação acima, obtemos as seguintes informações:
# 
# <font color = "blue">Receita de Impostos</font>
# 
# * Despesas com saúde (despesas_saude_%pib) tem uma forte correlação com a receita tributária do governo (receita_trib_%pib). O coeficiente de correlação é <font color = "magenta"> + 0,78 </font>, o que indica que eles são diretamente proporcionais.
# 
# 
# * Despesas com educação (despesas_educ_total) tem uma boa correlação com a receita tributária (receita_trib_%pib). O coeficiente de correlação é <font color = "magenta"> + 0,47 </font>, o que indica que são diretamente proporcionais, mas não da mesma forma que no setor saúde.
# 
# <font color = "blue">Receita de Empréstimos do BIRD e Créditos da AID</font>
# 
# * Despesas com saúde (despesas_saude_%pib) tem uma forte correlação com os empréstimos do BIRD e os créditos da AID (ibrd_e_ida). O coeficiente de correlação é <font color = "magenta"> + 0,74 </font>, o que indica que eles são diretamente proporcionais.
# 
# 
# * As despesas com educação (despesas_educ_total) têm uma correlação significativamente forte com os empréstimos do BIRD e os créditos da AID (ibrd_e_ida). O coeficiente de correlação é <font color = "magenta"> + 0,75 </font>, o que indica que eles são diretamente proporcionais. Também indica que a educação é principalmente financiada por meio de empréstimos do BIRD e créditos da AID, em oposição à receita fiscal do governo.

# ## <font color = "green">5 - Como a Educação e a Saúde se Correlacionam com o Crescimento do PIB?</font>
# 
# Usando a mesma matriz de correlação do item anterior, podemos concluir o seguinte:
# 
# * Despesas com educação (despesas_educ_total) tem uma correlação negativa com o crescimento do PIB per capita (pib_cresc_per_capita_%pib) ao longo dos anos com um coeficiente de <font color = "blue"> - 0,22 </font>. Isso pode ser atribuído principalmente à educação gratuita, portanto, levando a menores gastos com educação por parte dos cidadãos.
# 
# 
# * Despesas com saúde (despesas_saude_%pib) também tem uma correlação negativa com o crescimento do PIB per cappita (pib_cresc_per_capita_%pib) em <font color = "blue"> - 0,05 </font>, o que significa que são levemente inversamente proporcionais. Isso se deve aos menores gastos gerais com saúde, principalmente devido aos preços mais baixos - incluindo preços mais baixos de medicamentos e salários mais baixos para médicos e enfermeiras.

# ## <font color = "green">6 - Como os Gastos do Governo com Educação e Saúde Afetam o Crescimento do PIB?</font>

# In[43]:


# Adicionando os gastos de saúde e educação
data['total_gastos_educ_saude'] = data['gasto_educ'] + data['gasto_saude']


# In[53]:


# Definindo o tamanho do gráfico
plt.figure(figsize = (15.5, 8)) 

plt.title("Gráfico regressão")
sns.scatterplot(x="total_gastos_educ_saude", y="pib_cresc_anual%", data = data)
plt.xlabel('Gasto Total com Educação e Saúde')
plt.ylabel('Crescimento Anual do PIB')
plt.show


# # Modeo regressão linear

# In[54]:


# Prepara x e y
x = pd.DataFrame(data['total_gastos_educ_saude'])
y = pd.DataFrame(data['pib_cresc_anual%'])


# In[55]:


# Visualizando dados treino
x.shape


# In[56]:


# Visualizando os dados teste
y.shape


# # Treino e teste

# In[60]:


# Importando biblioteca de treino e teste
from sklearn.model_selection import train_test_split

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[61]:


# Visualizando os dados de treino
X_train.shape


# In[62]:


# Visualizando os dados de teste
y_train.shape


# ## Modelo regressão linear 

# In[70]:


# Cria e treina o modelo
model_regression_linear = LinearRegression()
model_regression_linear_fit = model_regression_linear.fit(X_train, y_train)
model_regression_linear


# In[71]:


model_regression_score = model_regression_linear.score(X_train, y_train)
model_regression_score


# In[65]:


# Fazendo previsões no conjunto de teste
model_regression_linear_pred = model_regression_linear.predict(X_test)
model_regression_linear_pred


# In[58]:


# Extraindo o coeficiente
model_regression_linear.coef_


# In[73]:


from sklearn.metrics import mean_squared_error

# Calculando o erro médio quadrado (MSE)
MSE = mean_squared_error(y_test, model_regression_linear_pred)
MSE


# In[74]:


print("Coeficientes:", model_regression_linear.coef_)
print("Intercept:", model_regression_linear.intercept_)
print("Erro Médio Quadrado (MSE):", MSE)


# In[95]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

# Calculando métricas
mae = mean_absolute_error(y_test, model_regression_linear_pred)
rmse = np.sqrt(mean_squared_error(y_test, model_regression_linear_pred))
r2 = r2_score(y_test, model_regression_linear_pred)
rmse = np.sqrt(mean_squared_error(y_test, model_regression_linear_pred))

# Imprimindo métricas
print("Erro Médio Absoluto (MAE):", mae)
print("Raiz Quadrada do Erro Médio Quadrado (RMSE):", rmse)
print("Coeficiente de Determinação (R²):", r2)
print("Raiz Quadrada do Erro Médio Quadrado (RMSE):", rmse)


# ## Modelo 2 - LGBMRegressor

# In[85]:


from lightgbm import LGBMRegressor

# Criando o modelo LGBMRegressor
model_LGBMRegressor = LGBMRegressor()

# Treinando o modelo
model_LGBMRegressor_fit = model_LGBMRegressor.fit(X_train, y_train)

model


# In[86]:


# Score modelo
model_LGBMRegressor_score = model_LGBMRegressor.score(X_train, y_train)
model_LGBMRegressor_score


# In[87]:


# Fazendo previsões no conjunto de teste
model_LGBMRegressor_pred = model_LGBMRegressor.predict(X_test)
model_LGBMRegressor_pred


# In[97]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculando métricas
mae = mean_absolute_error(y_test, model_LGBMRegressor_pred)
rmse = np.sqrt(mean_squared_error(y_test, model_LGBMRegressor_pred))
r2 = r2_score(y_test, model_LGBMRegressor_pred)
rmse = np.sqrt(mean_squared_error(y_test, model_LGBMRegressor_pred))

# Imprimindo métricas
print("Erro Médio Absoluto (MAE):", mae)
print("Raiz Quadrada do Erro Médio Quadrado (RMSE):", rmse)
print("Coeficiente de Determinação (R²):", r2)
print("Raiz Quadrada do Erro Médio Quadrado (RMSE):", rmse)


# In[92]:


# Calculando o erro médio quadrado (MSE)
MSE = mean_squared_error(y_test, model_LGBMRegressor_pred)
print("Erro Médio Quadrado (MSE):", MSE)


# # Modelo 3 - CatBoost Regressor

# In[108]:


from catboost import CatBoostRegressor

# Criando o modelo CatBoostRegressor
model_CatBoostRegressor = CatBoostRegressor(iterations=2000, 
                          learning_rate=0.1, 
                          depth=6, 
                          verbose=100)

# Treinando o modelo
model_CatBoostRegressor_fit = model_CatBoostRegressor.fit(X_train, y_train)

# Score modelo
model_CatBoostRegressor_score = model_CatBoostRegressor.score(X_train, y_train)
model_CatBoostRegressor_score


# In[109]:


# Fazendo previsões no conjunto de teste
model_CatBoostRegressor_pred = model_CatBoostRegressor.predict(X_test)
model_CatBoostRegressor_pred


# In[112]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculando métricas
mae = mean_absolute_error(y_test, model_CatBoostRegressor_pred)
rmse = np.sqrt(mean_squared_error(y_test, model_CatBoostRegressor_pred))
r2 = r2_score(y_test, model_CatBoostRegressor_pred)
rmse = np.sqrt(mean_squared_error(y_test, model_CatBoostRegressor_pred))
MSE = mean_squared_error(y_test, model_CatBoostRegressor_pred)

# Imprimindo métricas
print("Erro Médio Quadrado (MSE):", MSE)
print()
print("Erro Médio Absoluto (MAE):", mae)
print()
print("Raiz Quadrada do Erro Médio Quadrado (RMSE):", rmse)
print()
print("Coeficiente de Determinação (R²):", r2)
print()
print("Raiz Quadrada do Erro Médio Quadrado (RMSE):", rmse)


# # Resultados

# In[115]:


# Resultados modelos de regressão lineares
r2_model_regression_linear = r2_score(y_test, model_regression_linear_pred)
r2_model_LGBMRegressor = r2_score(y_test, model_LGBMRegressor_pred)
r2_model_CatBoostRegressor = r2_score(y_test, model_CatBoostRegressor_pred)

print("Coeficiente de Determinação - Regression linear (R²):", r2_model_regression_linear)
print("Coeficiente de Determinação - LGBM Regressor (R²):", r2_model_LGBMRegressor)
print("Coeficiente de Determinação - CatBoost Regressor (R²):", r2_model_CatBoostRegressor)


# ## <font color = "green">7- Com Base na Análise Quais São as Recomendações aos Governantes e Gestores?</font>
# 
# Os resultados sugerem que os gastos do governo com educação e saúde afetam positivamente o crescimento do PIB.
# Um aumento unitário nas despesas do governo leva a um aumento unitário de 5,20791642e-13 no crescimento do PIB.
# 
# Educação:
# 
# * O governo deve garantir o desenvolvimento do capital humano.
# * Construir mais escolas, treinar e empregar mais professores para garantir uma educação de boa qualidade, tornando a educação acessível a todos e reduzindo o custo da educação. 
# * Isso aumenta o desenvolvimento do capital humano e o crescimento do PIB no longo prazo. 
# 
# Saúde:
# 
# * O governo também pode continuar a fazer mais investimentos no setor de saúde, como a compra de equipamentos de saúde modernos, construção de mais hospitais, treinamento de mais profissionais de saúde e financiamento de pesquisa e desenvolvimento em saúde para combater epidemias.

# # Conclusão
# 
# Os achados dessa análise indicam que os investimentos do governo em setores como educação e saúde desempenham um papel positivo e significativo no impulso ao crescimento do Produto Interno Bruto (PIB). Cada incremento unitário nas despesas governamentais resulta em um notável aumento de 5,20791642e-13 no crescimento do PIB.
# 
# No tocante à educação, emerge a necessidade premente de o governo assegurar a ampliação do capital humano da nação. Essa meta pode ser atingida mediante a ampliação da infraestrutura educacional, abrangendo a construção de novas escolas e a contratação, formação e incorporação de um contingente mais expressivo de educadores. A oferta de uma educação de alta qualidade não apenas se torna acessível a todos os estratos da sociedade, mas também reduz os encargos associados ao processo educacional. Essa abordagem não apenas agrega valor ao capital humano, mas também tem o potencial de induzir um crescimento econômico sustentado em um horizonte de longo prazo.
# 
# No âmbito da saúde, o governo pode adotar medidas adicionais para fortalecer esse setor crucial. Isso engloba a alocação de recursos para aquisição de tecnologias médicas avançadas, expansão da rede hospitalar, capacitação de profissionais da área de saúde e o fomento à pesquisa e desenvolvimento em saúde, com enfoque especial na prevenção e contenção de epidemias. Essa abordagem não apenas aprimora a qualidade dos serviços de saúde, mas também contribui para a salvaguarda da saúde pública, resultando em um impacto positivo sobre a economia.
# 
# Portanto, os resultados obtidos sugerem fortemente que a alocação eficiente de recursos governamentais em educação e saúde pode catalisar um crescimento econômico sustentável e resiliente. A construção de uma infraestrutura educacional robusta e a criação de um sistema de saúde eficaz funcionam como pilares para o desenvolvimento contínuo da nação, solidificando seu posicionamento no cenário global.

# # Citação
# 
# **Esse projeto da formação cientista de dados da @Data Science Academy**

# # Referência
# 
# A Anedota do Abacaxi:
#     
# http://www.datascienceacademy.com.br/blog/a-anedota-do-abacaxi/
# 
# Data Science Academy:
# 
# http://www.datascienceacademy.com.br
# 
# Data Science Academy - Formacao cientista de dados:
# 
# https://www.datascienceacademy.com.br/bundle/formacao-cientista-de-dados
# 
# Data Science Academy - Preparação para carreira de cientista de dados:
# 
# https://www.datascienceacademy.com.br/course/preparao-para-carreira-de-cientista-de-dados

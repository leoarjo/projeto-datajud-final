# 📄 Documentação do Banco de Dados – Projeto Datajud (Amostra)

## 1. Nome do Dataset e Fonte

- **Nome:** Amostra de Processos Datajud (66.311 registros)  
- **Fonte Oficial:** API Pública do Datajud – CNJ  
- **Origem Interna:** Tabela `datajud_processos` armazenada em PostgreSQL  
- **Tipo da Amostra:** Probabilística (`TABLESAMPLE SYSTEM`)
- **Arquivo para ML:** `data/raw/datajud_amostra.csv`

A amostra foi criada especificamente para este projeto, garantindo viabilidade computacional e reprodutibilidade do pipeline.

---

## 2. Justificativa da Amostra

A base completa disponibilizada pelo Datajud possui:

- **≈ 26 milhões de registros**
- **≈ 78 GB de armazenamento**
- Estruturas JSON profundas (ex.: movimentos, complementos, classes)

Isso torna inviável:

- Rodar EDA diretamente  
- Processar JSON para features  
- Treinar modelos no ambiente local  
- Criar pipelines reprodutíveis dentro do prazo do trabalho  

Por isso, foi adotada uma **amostra probabilística estratificada**, garantindo:

- Representatividade por tribunal e grau  
- Variabilidade suficiente para treinar modelos  
- Rapidez de processamento  
- Reprodutibilidade  

A amostra final contém **66 mil processos**, número adequado para:

- análises exploratórias,  
- construção de modelos supervisionados,  
- execução completa do pipeline de MLOps,  
- carregamento rápido no Streamlit.

---

## 3. Contexto do Negócio

O CNJ define metas nacionais como:

- **Meta 1:** Julgar mais processos que os distribuídos  
- **Meta 2:** Julgar processos mais antigos  

Modelos analíticos ajudam a:

- Identificar padrões de julgamento  
- Melhorar previsibilidade de fluxo processual  
- Suportar diagnósticos de produtividade  
- Auxiliar indicadores estratégicos  

Neste projeto, buscamos **prever se um processo já apresenta movimentos típicos de julgamento**, usando apenas dados estruturados da capa e da movimentação.

---

## 4. Modelo Conceitual do Dataset

O dataset final é composto por **uma única tabela analítica**, derivada da estrutura JSON bruta do Datajud.

### Diagrama Conceitual (Simples)

Processo
├── id (PK)
├── tribunal
├── grau
├── classe_nome
├── qtd_movimentos
└── foi_julgado (target)


### Dicionário de Dados Final

| Coluna | Tipo | Descrição | Exemplo |
|--------|-------|-----------|----------|
| **id** | string | Identificador único do processo | TRT5_G1_0000… |
| **tribunal** | string | Sigla do tribunal | TRT6 |
| **grau** | string | Grau de jurisdição (G1, G2) | G1 |
| **classe_nome** | string | Classe processual | “Ação Trabalhista - Rito Ordinário” |
| **qtd_movimentos** | int | Quantidade total de movimentos do processo | 84 |
| **foi_julgado** | int (0/1) | Indica se há movimento compatível com julgamento | 1 |

Observações:

- **G1** = Primeira instância  
- **G2** = Segunda instância  
- `foi_julgado` é definido a partir de palavras-chave nos movimentos (ex.: sentença, julgamento, acórdão, trânsito etc.)  
- As features foram extraídas após expandir parcialmente a estrutura JSON original da coluna `data`.

---

## 5. Pré-Processamento

### Feito no SQL:

- Extração das colunas úteis da estrutura JSON.
- Criação das features:
  - tribunal  
  - grau  
  - classe_nome  
  - qtd_movimentos  
- Criação da variável-alvo com base em movimentos contendo:
  - “sentença”
  - “julgamento”
  - “acórdão”
  - “baixa”
  - “arquivamento”
  - “trânsito em julgado”
- Ajuste de tipos e limpeza textual.

### Feito no Python:

- Remoção de nulos (SimpleImputer)
- Codificação categórica (OneHotEncoder)
- Padronização numérica (StandardScaler)
- Encapsulamento no `ColumnTransformer`

---

## 6. Problema de Pesquisa

> **Dado um processo da amostra, qual a probabilidade dele já ter sido julgado?**

Tipo: **Classificação Binária**

- 0 = não julgado  
- 1 = julgado  

### Modelos Utilizados

- 🔹 Logistic Regression  
- 🔹 Random Forest (melhor desempenho)

Métrica principal: **F1-Score**, para lidar com classes potencialmente desequilibradas.

---

## 7. Pipeline de MLOps

| Arquivo | Função |
|--------|--------|
| `src/data_ingestion.py` | Carrega o dataset |
| `src/data_processing.py` | Define pré-processamento (transformers) |
| `src/modeling.py` | Treina, avalia e salva o modelo |
| `models/best_pipeline.joblib` | Pipeline final serializada |
| `app.py` | Dashboard interativo em Streamlit |

Propriedades:

- Modular  
- Reprodutível  
- Fácil manutenção  
- Totalmente automatizado  

---

## 8. Exportação da Base Analítica

A exportação foi realizada após seleção das colunas finais:

```sql
SELECT
    id,
    tribunal_clean AS tribunal,
    grau,
    classe_nome,
    qtd_movimentos,
    foi_julgado
FROM datajud_amostra;

Exportada via:

DBeaver → Export Resultset → CSV → data/raw/datajud_amostra.csv

Configurações:

Delimitador: ,

Quote: "

Encoding: UTF-8

Header habilitado

9. Conclusão

A documentação, o dataset e o pipeline garantem:

Reprodutibilidade

Clareza metodológica

Organização lógica

Adequação ao problema de pesquisa

Aderência às metas do CNJ

O dataset final está pronto para:

análises exploratórias (EDA)

demonstração do modelo

execução no dashboard

uso futuro em pipelines mais robustos

---

## 10. Modelos Analíticos e Pipeline de Machine Learning

Atendendo ao item 2 da avaliação, o projeto foi estruturado em um pipeline de Machine Learning composto por scripts modulares, que automatizam desde a carga dos dados até o treinamento, avaliação e serialização do modelo final.

### 10.1 Ingestão de Dados (`data_ingestion.py`)

Responsabilidades principais:

- Carregar o arquivo `data/raw/datajud_amostra.csv`;
- Organizar as variáveis em:
  - **Features (X):** `tribunal`, `grau`, `classe_nome`, `qtd_movimentos`;
  - **Target (y):** `foi_julgado` (0 = não julgado, 1 = julgado).

Essa separação é utilizada tanto no processo de modelagem quanto, indiretamente, nas predições realizadas via Streamlit.

### 10.2 Pré-Processamento e Transformação (`data_processing.py`)

O pré-processamento foi encapsulado em um objeto `ColumnTransformer`, integrado ao `Pipeline` do scikit-learn, garantindo consistência entre as etapas de treino e predição.

Principais etapas:

- **Tratamento de valores ausentes:**
  - Numéricas: imputação pela mediana;
  - Categóricas: imputação pelo valor mais frequente.
- **Codificação de variáveis categóricas:**
  - Uso de `OneHotEncoder(handle_unknown="ignore")` para as colunas:
    - `tribunal`,
    - `grau`,
    - `classe_nome`.
- **Escalonamento de variáveis numéricas:**
  - Uso de `StandardScaler` para a variável `qtd_movimentos`.

Todo esse pré-processamento é aplicado automaticamente dentro do `Pipeline` do scikit-learn, o que garante que os mesmos passos sejam utilizados tanto no treinamento quanto nas predições no aplicativo Streamlit.

### 10.3 Modelagem (`modeling.py`)

Na etapa de modelagem foram treinados e comparados modelos de classificação binária:

- **Regressão Logística**
- **Random Forest Classifier**

Ambos os modelos foram integrados ao mesmo pré-processador via `Pipeline`, de forma que cada candidato a modelo é, na prática, um pipeline completo: `preprocessor + model`.

### 10.4 Avaliação dos Modelos

Para comparar os modelos, foram utilizadas métricas de classificação, com foco em:

- Acurácia;
- Precisão;
- Recall;
- **F1-Score** (métrica principal).

O script `modeling.py` gera o `classification_report` para cada modelo testado e utiliza o **F1-Score** para selecionar o melhor pipeline.

### 10.5 Serialização do Pipeline

Após a comparação, o melhor modelo é escolhido e serializado em disco junto com o pré-processamento, utilizando a biblioteca `joblib`:

- Arquivo gerado: `models/best_pipeline.joblib`.

Esse arquivo é posteriormente carregado no aplicativo Streamlit (`app.py`) para geração das predições em tempo real, sem necessidade de re-treinar o modelo.

---

## 11. Dashboard Interativo com Streamlit (`app.py`)

O dashboard desenvolvido em Streamlit é o ponto central de apresentação do projeto, cobrindo os elementos solicitados no item 3 da avaliação. Ele foi organizado em três páginas principais, acessadas via barra lateral.

### 11.1 Página 1 – Introdução e Contextualização

Elementos presentes:

- **Título do projeto:** contextualiza a utilização do Datajud e o foco em predição de julgamento de processos;
- **Descrição geral da solução:** explicação do uso de amostra, pipeline de ML e API pública;
- **Problema de pesquisa:** “Dado um processo, qual a probabilidade de ele já ter sido julgado?”;
- **Resumo da metodologia:** amostragem da base original, construção da tabela analítica, definição do target `foi_julgado` e treinamento de modelos supervisionados.

Essa página funciona como uma síntese textual da Avaliação 1 e da evolução para o contexto de MLOps.

### 11.2 Página 2 – Análise Exploratória de Dados (EDA)

Elementos principais:

- **Gráficos interativos (Plotly):**
  - Distribuição de processos por tribunal (`tribunal`);
  - Distribuição por grau (`grau`);
  - Distribuição do target (`foi_julgado`), permitindo visualizar a proporção entre julgados e não julgados.
- **Visualização agregada:** os gráficos permitem identificar rapidamente a concentração de processos por tribunal, o comportamento por grau de jurisdição (G1, G2) e o equilíbrio entre as classes da variável alvo.

A partir desses gráficos, o usuário consegue entender melhor a composição da amostra antes de acessar a parte preditiva do sistema.

### 11.3 Página 3 – Análises Preditivas e Relatório

A página de modelo preditivo concentra a interação direta com o pipeline treinado.

Componentes:

- **Entrada de dados pelo usuário:**
  - `st.selectbox` para selecionar:
    - Tribunal (`tribunal`);
    - Grau (`grau`);
    - Classe processual (`classe_nome`);
  - `st.number_input` para informar a quantidade de movimentos do processo (`qtd_movimentos`).
- **Geração de predição:**
  - Ao clicar no botão “Gerar Predição”, o app:
    - Monta um `DataFrame` com os valores informados;
    - Aplica o pipeline serializado (`best_pipeline.joblib`);
    - Calcula a classe prevista (0 ou 1) e a probabilidade associada.
- **Apresentação do resultado:**
  - Mensagens em destaque (`st.success` ou `st.error`) informando:
    - Se o modelo prevê que o processo **foi julgado** ou **não foi julgado**;
    - A probabilidade estimada, em formato percentual.

Essa estrutura atende ao requisito de **previsão interativa**, em que o usuário pode testar diferentes combinações de valores e obter a resposta do modelo em tempo real.

Embora a comparação gráfica de métricas entre modelos (ex.: tabela com acurácia e F1 de cada algoritmo) ainda seja feita no terminal durante o treinamento, a arquitetura atual já suporta, em futuras versões, a inclusão de uma página extra com esses resultados consolidados.

---

# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

from src.data_ingestion import load_data

# ==============================
# CONFIGURAÇÕES GERAIS DO APP
# ==============================
st.set_page_config(
    page_title="Dashboard Datajud – Projeto Final",
    layout="wide",
    page_icon="⚖️"
)

# ==============================
# FUNÇÕES DE CACHE
# ==============================
@st.cache_data
def load_df():
    return load_data()

@st.cache_resource
def load_pipeline():
    return joblib.load("models/best_pipeline.joblib")


# ==============================
# PÁGINA PRINCIPAL
# ==============================
def main():
    st.sidebar.title("📁 Navegação")
    page = st.sidebar.radio(
        "Selecione uma seção:",
        ["📌 Introdução", "🔍 EDA – Análise Exploratória", "🤖 Predição com ML"]
    )

    df = load_df()

    # ==============================
    # 📌 PÁGINA 1 – INTRODUÇÃO
    # ==============================
    if page == "📌 Introdução":
        st.title("⚖️ Projeto Datajud – Classificação de Julgamento de Processos")

        st.markdown("""
        ---
        ## 🧠 **Visão Geral do Projeto**

        Este dashboard apresenta o resultado de um projeto completo de MLOps,
        utilizando dados reais extraídos do **Datajud** (CNJ).  
        O objetivo é **prever automaticamente** se um processo já apresenta sinais de julgamento,
        com base em informações estruturadas do processo.

        A base utilizada contém **66 mil processos** obtidos via amostragem probabilística da API pública.

        ---
        ### 🔧 **O que foi feito no projeto (pipeline completo)**

        - 📥 **Amostragem controlada** de 66k processos  
        - 🧹 **Pré-processamento** com *ColumnTransformer*  
        - 🤖 **Treinamento de modelos de ML**:
            - Regressão Logística
            - Random Forest  
        - 🏆 **Seleção do modelo vencedor** usando *F1-score*  
        - 💾 **Serialização** com *joblib*  
        - 📊 **Dashboard interativo** via Streamlit  

        ---
        ### 🎯 Objetivo Geral:
        Criar uma solução capaz de **identificar automaticamente** processos que
        já apresentam sinais de julgamento, auxiliando análises globais de desempenho
        e padrões de workflow judicial.

        ---
        """)

        st.info(
            "➡️ Use o menu à esquerda para navegar entre a análise exploratória (EDA) "
            "e a página de predição automática."
        )

    # ==============================
    # 🔍 PÁGINA 2 – EDA
    # ==============================
    elif page == "🔍 EDA – Análise Exploratória":
        st.title("🔍 Análise Exploratória dos Dados (EDA)")
        st.markdown("""
        ---
        ## 📊 **Distribuições e Padrões da Amostra**
        Abaixo estão algumas visualizações para entender o comportamento dos dados,
        como distribuição por tribunal, grau e proporção de julgamentos.
        ---
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Distribuição por Tribunal")
            fig1 = px.histogram(df, x="tribunal", title="Ocorrências por Tribunal")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("🏛 Distribuição por Grau")
            fig2 = px.histogram(df, x="grau", title="Ocorrências por Grau (G1, G2...)")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("⚖️ Proporção de Processos Julgados vs. Não Julgados")
        fig3 = px.histogram(
            df,
            x="foi_julgado",
            title="Distribuição do Target (Julgado vs. Não Julgado)",
            color="foi_julgado",
            barmode="group"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ==============================
    # 🤖 PÁGINA 3 – ML + COMPARAÇÃO DE MODELOS
    # ==============================
    elif page == "🤖 Predição com ML":
        st.title("🤖 Predição – Foi Julgado ou Não?")

        # -----------------------------------------------------------
        # 🔥 NOVA SEÇÃO: Comparação de Modelos
        # -----------------------------------------------------------
        st.markdown("""
        ---
        ## 📊 Comparação de Modelos de Machine Learning
        A tabela e o gráfico abaixo apresentam as métricas dos modelos treinados,
        permitindo comparar desempenho entre Regressão Logística e Random Forest.
        ---
        """)

        metrics_path = "models/model_metrics.csv"

        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)

            st.dataframe(metrics_df, use_container_width=True)

            fig_metrics = px.bar(
                metrics_df.melt(id_vars="modelo", var_name="métrica", value_name="valor"),
                x="métrica",
                y="valor",
                color="modelo",
                barmode="group",
                title="Comparação das Métricas dos Modelos"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        else:
            st.warning("⚠️ Arquivo de métricas não encontrado em `models/model_metrics.csv`. "
                       "Execute novamente `modeling.py` para gerar as métricas.")

        # -----------------------------------------------------------

        st.markdown("""
        ---
        ## 🧩 **Modelo Preditivo**
        Preencha os campos abaixo para gerar uma previsão automática.
        ---
        """)

        pipeline = load_pipeline()

        col1, col2 = st.columns(2)

        with col1:
            tribunal = st.selectbox("🏛 Tribunal", sorted(df["tribunal"].unique()))
            grau = st.selectbox("⚖ Grau", sorted(df["grau"].unique()))

        with col2:
            classe_nome = st.selectbox("📚 Classe Processual", sorted(df["classe_nome"].unique()))
            qtd_mov = st.number_input("📎 Quantidade de Movimentos", min_value=0, max_value=2000, value=10)

        entrada = pd.DataFrame([{
            "tribunal": tribunal,
            "grau": grau,
            "classe_nome": classe_nome,
            "qtd_movimentos": qtd_mov
        }])

        st.markdown("---")

        if st.button("🚀 Gerar Predição"):
            pred = pipeline.predict(entrada)[0]
            prob = pipeline.predict_proba(entrada)[0][1]

            if pred == 1:
                st.success(
                    f"🟢 **O processo provavelmente FOI JULGADO.**\n\n"
                    f"📌 Probabilidade estimada: **{prob:.2%}**"
                )
            else:
                st.error(
                    f"🔴 **O processo provavelmente NÃO FOI JULGADO.**\n\n"
                    f"📌 Probabilidade estimada: **{prob:.2%}**"
                )


# ==============================
# INICIALIZAÇÃO
# ==============================
if __name__ == "__main__":
    main()

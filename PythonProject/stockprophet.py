import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import openpyxl
import os

st.set_page_config(page_title="Análise de Ações com Prophet", layout="wide")
st.title("Análise de Ações com Prophet")

st.markdown("""
Esta aplicação permite fazer análises completas de preços de ações utilizando o modelo **Prophet**.

1. Faça o upload de um arquivo **CSV ou Excel (.xlsx** contendo os dados históricos)
2. Selecione o ticker desejado
3. Veja a análise completa com previsão, componentes e indicadores técnicos no período de 180 dias (6 meses)
""")

uploaded_file = st.file_uploader("Escolha o arquivo CSV ou XLSX", type=["csv", "xlsx"])

def calculate_technical_indicators(df):
    df = df.copy()
    df['y'] = df['y'].ffill()

    # Médias móveis
    df['MA_20'] = df['y'].rolling(window=20, min_periods=1).mean()
    df['MA_50'] = df['y'].rolling(window=50, min_periods=1).mean()
    df['MA_200'] = df['y'].rolling(window=200, min_periods=1).mean()

    # Bollinger Bands
    df['BB_MA20'] = df['y'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['y'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_MA20'] + (2 * rolling_std)
    df['BB_Lower'] = df['BB_MA20'] - (2 * rolling_std)

    # MACD
    ema12 = df['y'].ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = df['y'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df_full = pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            df_full = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado.")
            st.stop()

        # Pré-processamento similar ao UXProphetStock.py
        df_full = df_full.rename(columns=lambda x: x.lower())
        price_columns = [col for col in df_full.columns if 'price.' in col]
        for col in price_columns:
            if df_full[col].dtype == 'object':
                df_full[col] = df_full[col].str.replace(',', '.').astype(float)
        
        df_full['ref.date'] = pd.to_datetime(df_full['ref.date'])

        if not {'ticker', 'ref.date', 'price.close'}.issubset(df_full.columns):
            st.error("O arquivo deve conter as colunas: ticker, ref.date e price.close.")
        else:
            tickers = df_full['ticker'].unique().tolist()
            selected_ticker = st.selectbox("Selecione o ticker:", sorted(tickers))

            # Dados para Prophet
            df_prophet = df_full[df_full['ticker'] == selected_ticker][['ref.date', 'price.close']].copy()
            df_prophet.columns = ['ds', 'y']
            
            # Abas para diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["Previsão", "Componentes", "Indicadores Técnicos"])

            with tab1:
                st.subheader(f"Previsão de Preço - {selected_ticker}")
                
                model = Prophet(daily_seasonality=True)
                model.fit(df_prophet)

                future = model.make_future_dataframe(periods=180)
                forecast = model.predict(future)
                
                fig = plot_plotly(model, forecast)

                # Personalização das cores
                fig.update_traces(
                    selector=dict(name='ds'),  # Pontos históricos (dados reais)
                    marker=dict(
                        color='#ffffff',      # Cor dos pontos reais
                        size=4,               # Tamanho dos pontos
                        opacity=0.8           # Transparência
                    ),
                    line=dict(width=0)        # Remove a linha conectando os pontos
                )
                fig.update_layout(
                    title=f"Previsão de Preço da Ação ({selected_ticker})",
                    xaxis_title="Data",
                    yaxis_title="Preço (R$)",
                    template="plotly_white",
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)


            with tab2:
                st.subheader(f"Componentes da Previsão - {selected_ticker}")
                fig_components = plot_components_plotly(model, forecast)
                fig_components.update_layout(
                    title=f"Componentes da Previsão ({selected_ticker})",
                    template="plotly_white"
                )
                st.plotly_chart(fig_components, use_container_width=True)

            with tab3:
                st.subheader(f"Indicadores Técnicos - {selected_ticker}")
                
                df_indicators = calculate_technical_indicators(df_prophet)
                
                # Gráfico 1: Preço e Médias Móveis
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['y'], name='Preço de Fechamento'))
                fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MA_20'], name='MA 20'))
                fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MA_50'], name='MA 50'))
                fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MA_200'], name='MA 200'))
                fig1.update_layout(
                    title="Preço e Médias Móveis",
                    xaxis_title="Data",
                    yaxis_title="Preço",
                    template="plotly_white"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Gráfico 2: MACD
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_indicators['ds'], y=df_indicators['MACD_Hist'], marker_color=np.where(df_indicators['MACD_Hist'] > 0, 'green', 'red'),
                    name='Histograma MACD'))
                fig2.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MACD'], name='MACD', line=dict(color='blue', width=2)))
                fig2.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MACD_Signal'], name='Sinal',
                    line=dict(color='orange', width=2)))
                fig2.update_layout(
                    title="MACD", xaxis_title="Data", yaxis_title="Valor", template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Gráfico 3: Bollinger Bands
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=df_indicators['ds'], 
                    y=df_indicators['y'], 
                    name='Preço',
                    line=dict(color='blue', width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=df_indicators['ds'], 
                    y=df_indicators['BB_MA20'], 
                    name='MA20',
                    line=dict(color='orange', width=1)
                ))
                fig3.add_trace(go.Scatter(
                    x=df_indicators['ds'], 
                    y=df_indicators['BB_Upper'], 
                    name='Banda Superior',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                fig3.add_trace(go.Scatter(
                    x=df_indicators['ds'], 
                    y=df_indicators['BB_Lower'], 
                    name='Banda Inferior',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty'
                ))
                fig3.update_layout(
                    title="Bollinger Bands",
                    xaxis_title="Data",
                    yaxis_title="Preço",
                    template="plotly_white"
                )
                st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
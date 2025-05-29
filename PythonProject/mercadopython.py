import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from prophet.plot import plot_plotly
from sklearn.metrics import r2_score
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly
from datetime import timedelta


st.set_page_config(layout="wide", page_title="Análide de Ações na B3", initial_sidebar_state="expanded")

@st.cache_data
def carregar_dados():
    df = pd.read_excel(r"C:\Users\heito\OneDrive\Desktop\datestock.xlsx")
    df = df.rename(columns=lambda x: str(x).lower())
    df['ref.date'] = pd.to_datetime(df['ref.date'])
    df['setor'] = df['setor'].replace(False, 'sem setor')
    df['subsetor'] = df['subsetor'].replace(False, 'sem subsetor')
    price_cols = ['price.open', 'price.high', 'price.low', 'price.close', 'price.adjusted']
    return_cols = ['ret.adjusted.prices', 'ret.closing.prices']
    for col in price_cols + return_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

df = carregar_dados()

# Sidebar
st.sidebar.title("📈 Análise de Ações com Prophet")
ticker = st.sidebar.selectbox("Selecione o Ticker", df['ticker'].dropna().unique())
st.sidebar.markdown('> _\"Utilize esses resultados para auxiliar nas suas decisões, não como certeza.\"_')

# Filtragem
df_ticker = df[df['ticker'] == ticker][['ref.date', 'price.close']].rename(columns={'ref.date': 'ds', 'price.close': 'y'})
df_ticker.dropna(inplace=True)

# Modelagem
modelo = Prophet(daily_seasonality=True)
modelo.fit(df_ticker)
futuro = modelo.make_future_dataframe(periods=365)
previsao = modelo.predict(futuro)

# Gráfico interativo
st.markdown(f"## 📊 Previsão de Preço para {ticker}")
fig1 = plot_plotly(modelo, previsao)

# Renomear os eixos
fig1.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço Previsto (R$)",
    title=f"Previsão de Preço da Ação - {ticker}",
    template="plotly_dark"  # Garante o dark mode
)

st.plotly_chart(fig1, use_container_width=True)

# Componentes
st.markdown("## 🔍 Componentes da Previsão")
componentes = modelo.plot_components(previsao)
componentes.figure.set_size_inches(14, 8)
st.pyplot(componentes.figure)


# Métricas
st.markdown("## 📐 Métricas de Avaliação")
df_cv = cross_validation(modelo, initial='365 days', period='180 days', horizon='90 days')
df_p = performance_metrics(df_cv)
r2 = r2_score(df_cv['y'], df_cv['yhat'])

# HTML personalizado
html_metricas = f"""
<div style="background-color:#1c1c1c; padding:20px; border-radius:10px;">
  <h4 style="color:#00CED1;">Métricas de Avaliação do Modelo</h4>
  <ul style="color:white; font-size:16px;">
    <li><strong>RMSE:</strong> {df_p['rmse'].mean():.2f}</li>
    <li><strong>MAE:</strong> {df_p['mae'].mean():.2f}</li>
    <li><strong>MAPE:</strong> {df_p['mape'].mean() * 100:.2f}%</li>
    <li><strong>Coverage:</strong> {df_p['coverage'].mean() * 100:.2f}%</li>
    <li><strong>R²:</strong> {r2:.4f}</li>
  </ul>
</div>
"""

st.markdown(html_metricas, unsafe_allow_html=True)

# Datas-alvo baseadas na última data real
data_base = df_ticker['ds'].max()
datas_alvo = {
    '1 Semana': data_base + timedelta(days=7),
    '1 Mês': data_base + timedelta(days=30),
    '6 Meses': data_base + timedelta(days=180),
    '1 Ano': data_base + timedelta(days=365)
}

# Encontrar os preços previstos mais próximos das datas
precos_previstos = {}
for periodo, data_alvo in datas_alvo.items():
    linha = previsao.iloc[(previsao['ds'] - data_alvo).abs().argsort()[:1]]
    precos_previstos[periodo] = linha['yhat'].values[0]

# HTML estilizado para sidebar
html_sidebar = """
<div style="background-color:#262730; padding:15px; border-radius:10px;">
  <h4 style="color:#00CED1;">📊 Preços Previstos</h4>
  <ul style="color:white; font-size:15px; padding-left:20px;">
"""
for periodo, preco in precos_previstos.items():
    html_sidebar += f"<li><b>{periodo}:</b> R$ {preco:.2f}</li>"
html_sidebar += "</ul></div>"

# Exibir na sidebar
st.sidebar.markdown(html_sidebar, unsafe_allow_html=True)

# Download
st.download_button("📥 Baixar Previsão (Excel)", data=previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False),
                   file_name=f'previsao_{ticker}.csv', mime='text/csv')


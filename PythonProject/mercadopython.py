import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import r2_score
from datetime import timedelta


# Configuração da página
st.set_page_config(page_title="Previsão de Ações", layout="wide")

# Cabeçalho principal estilizado
st.markdown("""
<div style="background-color:#111111;padding:20px;border-radius:10px;text-align:center;">
  <h1 style="color:#00CED1;">📈 Aplicação de Previsão de Ações com Prophet</h1>
  <p style="color:#DDDDDD;font-size:16px;">
    Visualize projeções, componentes sazonais e indicadores técnicos.
  </p>
</div>
""", unsafe_allow_html=True)


# Carregamento e preprocessamento dos dados
@st.cache_data

def carregar_dados():
    df = pd.read_excel(r"C:\Users\heito\PycharmProjects\PythonProject\datestock.xlsx")
    df = df.rename(columns=lambda x: str(x).lower())
    df['ref.date'] = pd.to_datetime(df['ref.date'])
    df['setor'] = df['setor'].replace(False, 'sem setor')
    df['subsetor'] = df['subsetor'].replace(False, 'sem subsetor')
    for col in ['price.open', 'price.high', 'price.low', 'price.close', 'price.adjusted',
                'ret.adjusted.prices', 'ret.closing.prices']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

# Dados
df = carregar_dados()

# Sidebar
st.sidebar.title("🔢 Funções")
ticker = st.sidebar.selectbox("Selecione o Ticker", df['ticker'].dropna().unique())

# Filtro do ticker selecionado
df_ticker = df[df['ticker'] == ticker][['ref.date', 'price.close']].rename(columns={'ref.date': 'ds', 'price.close': 'y'})
df_ticker.dropna(inplace=True)

# Modelo Prophet
modelo = Prophet(daily_seasonality=True)
modelo.fit(df_ticker)
futuro = modelo.make_future_dataframe(periods=365)
previsao = modelo.predict(futuro)

# Gráfico de previsão com legenda e cores personalizadas
st.markdown("""
<div style="margin-top:30px;margin-bottom:10px;">
  <h2 style="color:#00CED1;">🔮 Previsão do Preço da Ação</h2>
</div>
""", unsafe_allow_html=True)

# Criar figura com plot_plotly
fig1 = plot_plotly(modelo, previsao)

# Personalizar legendas e cores
fig1.update_traces(
    selector=dict(name="yhat"),
    name="Previsão Central",
    line=dict(color="#00CED1", width=3)
)
fig1.update_traces(
    selector=dict(name="yhat_upper"),
    name="Limite Superior",
    line=dict(color="#98FB98", dash="dot")
)
fig1.update_traces(
    selector=dict(name="yhat_lower"),
    name="Limite Inferior",
    line=dict(color="#FF6F61", dash="dot")
)
fig1.update_traces(
    selector=dict(name="actual"),
    name="Preço Real",
    line=dict(color="#FFFFFF", width=2)
)

# Layout e tema escuro
fig1.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço (R$)",
    title=f"Previsão de Preço da Ação - {ticker}",
    template="plotly_dark",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

# Exibir
st.plotly_chart(fig1, use_container_width=True)


# Componentes da previsão
st.markdown("""
<div style="margin-top:30px;margin-bottom:10px;">
  <h2 style="color:#00CED1;">📊 Componentes da Previsão</h2>
</div>
""", unsafe_allow_html=True)
fig_comp = modelo.plot_components(previsao)
fig_comp.set_size_inches(12, 10)
st.pyplot(fig_comp.figure)

# Métricas
st.markdown("""
<div style="margin-top:30px;margin-bottom:10px;">
  <h2 style="color:#00CED1;">⚖️ Métricas de Avaliação</h2>
</div>
""", unsafe_allow_html=True)
df_cv = cross_validation(modelo, initial='365 days', period='180 days', horizon='90 days')
df_p = performance_metrics(df_cv)
r2 = r2_score(df_cv['y'], df_cv['yhat'])

html_metricas = f"""
<div style="background-color:#1c1c1c; padding:20px; border-radius:10px;">
  <ul style="color:#FFFFFF; font-size:16px;">
    <li><b>RMSE:</b> {df_p['rmse'].mean():.2f}</li>
    <li><b>MAE:</b> {df_p['mae'].mean():.2f}</li>
    <li><b>MAPE:</b> {df_p['mape'].mean() * 100:.2f}%</li>
    <li><b>Coverage:</b> {df_p['coverage'].mean() * 100:.2f}%</li>
    <li><b>R²:</b> {r2:.4f}</li>
  </ul>
</div>
"""
st.markdown(html_metricas, unsafe_allow_html=True)

# Preços previstos em horizontes específicos
data_base = df_ticker['ds'].max()
datas_alvo = {
    '1 Semana': data_base + timedelta(days=7),
    '1 Mês': data_base + timedelta(days=30),
    '6 Meses': data_base + timedelta(days=180),
    '1 Ano': data_base + timedelta(days=365)
}

precos_previstos = {}
for periodo, data_alvo in datas_alvo.items():
    linha = previsao.iloc[(previsao['ds'] - data_alvo).abs().argsort()[:1]]
    precos_previstos[periodo] = linha['yhat'].values[0]

html_sidebar = """
<div style="background-color:#262730; padding:15px; border-radius:10px;">
  <h4 style="color:#00CED1;">📊 Preços Previstos</h4>
  <ul style="color:white; font-size:15px;">
"""
for periodo, preco in precos_previstos.items():
    html_sidebar += f"<li><b>{periodo}:</b> R$ {preco:.2f}</li>"
html_sidebar += "</ul></div>"
st.sidebar.markdown(html_sidebar, unsafe_allow_html=True)

# Legenda da previsão na sidebar
st.sidebar.markdown("""
<div style="background-color:#262730; padding:15px; border-radius:10px; margin-top:15px;">
  <h4 style="color:#00CED1;">📘 Legenda - Previsão do Preço</h4>
  <ul style="color:white; font-size:14px;">
    <li><b style="color:#FFFFFF;">Preço Real:</b> pontos pretos</li>
    <li><b style="color:#00CED1;">Previsão dos preços:</b> linha azul</li>
    <li><b style="color:#98FB98;">Limite Superior e Inferior:</b> área opaca azul</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# Aviso
st.sidebar.markdown("""
<div style="background-color:#333;padding:15px;border-radius:10px;margin-top:20px;">
  <p style="color:#CCCCCC; font-size:14px; font-style:italic;">
    ⚠️ <strong>Atenção:</strong><br>
    Utilize essas previsões como <b>suporte</b> à decisão, <b>não como garantia</b> de resultado.
  </p>
</div>
""", unsafe_allow_html=True)

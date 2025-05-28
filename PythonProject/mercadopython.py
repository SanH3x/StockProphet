import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta

# 1. Carregar e preparar os dados
def load_and_prepare_data(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Formato de arquivo não suportado. Use .csv ou .xlsx")

    if 'ref.date' not in df.columns or 'price.close' not in df.columns:
        raise ValueError("Colunas obrigatórias 'ref.date' e 'price.close' não encontradas no arquivo")

    df['ref.date'] = pd.to_datetime(df['ref.date'], errors='coerce')
    df = df.rename(columns={'ref.date': 'ds', 'price.close': 'y'})
    df = df[['ds', 'y']].dropna()

    return df

# 2. Modelagem com Prophet
def create_prophet_model(df, periods=365, changepoint_prior_scale=0.05):
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale
    )
    model.add_country_holidays(country_name='BR')
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# 3. Criar gráficos interativos
def create_interactive_plots(df, forecast):
    # Gráfico de previsão
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico', line=dict(color='#1f77b4')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão', line=dict(color="#136413")))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line=dict(width=0), showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line=dict(width=0), name='Intervalo de Confiança'))

    fig_forecast.update_layout(
        title='Previsão de Preço de Fechamento', xaxis_title='Data', yaxis_title='Preço (R$)', hovermode='x unified',
        plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridcolor='#e0e0e0',
        zeroline=False,
        color='black'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='#e0e0e0',
        zeroline=False,
        color='black'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='#e0e0e0',
        zeroline=False,
        color='black'
    )
        )

    # Gráfico de componentes
    fig_components = go.Figure()
    fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Tendência', line=dict(color='#d62728')))
    fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'], name='Sazonalidade Semanal', line=dict(color='#9467bd')))
    fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Sazonalidade Anual', line=dict(color='#ff7f0e')))
    fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['daily'], name='Sazonalidade Diária', line=dict(color='#17becf')))

    fig_components.update_layout(title='Componentes da Previsão', xaxis_title='Data', hovermode='x unified')

    return fig_forecast, fig_components

# 4. Criar aplicação Dash
def create_dash_app(df):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Digite o Ticker da Ação:"),
                dcc.Input(id='ticker-input', type='text', value='', debounce=True, className="form-control")
            ], width=6)
        ], className="mb-4"),

        dbc.Row(dbc.Col(html.H1(id='titulo-analise', className="text-center my-4"))),

        dbc.Row([
            dbc.Col([
                html.Label("Período de Previsão (dias):"),
                dcc.Slider(
                    id='period-slider', min=30, max=730, step=30, value=365,
                    marks={i: str(i) for i in range(30, 731, 90)}
                )
            ], width=6),
            dbc.Col([
                html.Label("Sensibilidade a Mudanças:"),
                dcc.Slider(
                    id='changepoint-slider', min=0.01, max=0.5, step=0.01, value=0.05,
                    marks={0.01: '0.01', 0.1: '0.1', 0.2: '0.2', 0.3: '0.3', 0.4: '0.4', 0.5: '0.5'}
                )
            ], width=6)
        ], className="mb-4"),

        dbc.Row(dbc.Col(dcc.Graph(id='forecast-graph'))),
        dbc.Row(dbc.Col(dcc.Graph(id='components-graph'))),

        dbc.Row([
            dbc.Col(html.Div(id='metrics-output', className="mt-4 p-3 bg-light rounded")),
            dbc.Col(dcc.Markdown(id='ticker-description', className="mt-4 p-3 bg-light rounded"))
        ])
    ], fluid=True)

    @callback(
        Output('titulo-analise', 'children'),
        Input('ticker-input', 'value')
    )
    def update_titulo(ticker):
        return f"Análise de Previsão de Ações - {ticker.upper()}"

    @callback(
        [Output('forecast-graph', 'figure'),
         Output('components-graph', 'figure'),
         Output('metrics-output', 'children'),
         Output('ticker-description', 'children')],
        [Input('period-slider', 'value'),
         Input('changepoint-slider', 'value'),
         Input('ticker-input', 'value')]
    )
    def update_forecast(periods, changepoint_prior_scale, ticker):
        model, forecast = create_prophet_model(df, periods, changepoint_prior_scale)
        fig_forecast, fig_components = create_interactive_plots(df, forecast)

        last_date = df['ds'].max()
        forecast_start = last_date + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=periods-1)

        metrics = [
            html.H5(f"Previsão para {ticker.upper()}:"),
            html.P(f"Período: {forecast_start.strftime('%d/%m/%Y')} a {forecast_end.strftime('%d/%m/%Y')}"),
            html.P(f"Preço atual: R$ {df['y'].iloc[-1]:.2f}"),
            html.P(f"Preço previsto: R$ {forecast['yhat'].iloc[-1]:.2f}"),
            html.P(f"Variação: {((forecast['yhat'].iloc[-1] - df['y'].iloc[-1]) / df['y'].iloc[-1]) * 100:.2f}%")
        ]

        descricao = f"""
        ### Sobre o Modelo
        Esta aplicação utiliza o Facebook Prophet para prever os preços futuros das ações de {ticker.upper()}.

        - **Período de Previsão**: Define a quantidade de dias futuros incluídos na previsão.
        - **Sensibilidade a Mudanças**: Controla quão rapidamente o modelo reage a alterações nas tendências de preço.

        Os intervalos sombreados representam a incerteza da previsão com 80% de confiança.
        """

        return fig_forecast, fig_components, metrics, descricao

    return app

# 5. Função principal
def main():
    df = load_and_prepare_data(r"C:\Users\heito\PycharmProjects\PythonProject\datestock.xlsx")  # ou .xlsx conforme o caso
    app = create_dash_app(df)
    app.run(debug=True)

if __name__ == '__main__':
    main()

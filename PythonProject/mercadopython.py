from nicegui import ui
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
import openpyxl
import os
from datetime import datetime
import tempfile

# Configurações gerais
ui.page_title("Análise de Ações com Prophet")
ui.colors(primary="#026620")

# Dicionário de cores customizadas
cores = {
    "pontos_reais": "#8d0808",
    "linha_ma20": "#ff7f0e",
    "linha_ma50": "#1f77b4",
    "linha_ma200": "#2ca02c",
    "macd": "#1f77b4",
    "macd_sinal": "#ff9900",
    "macd_hist_acima": "green",
    "macd_hist_abaixo": "red",
    "bb_ma20": "#ff7f0e",
    "bb_superior": "#888888",
    "bb_inferior": "#888888",
    "preco": "#0000ff"
}

def calculate_technical_indicators(df):
    df = df.copy()
    df['y'] = df['y'].ffill()
    df['MA_20'] = df['y'].rolling(window=20, min_periods=1).mean()
    df['MA_50'] = df['y'].rolling(window=50, min_periods=1).mean()
    df['MA_200'] = df['y'].rolling(window=200, min_periods=1).mean()
    df['BB_MA20'] = df['y'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['y'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_MA20'] + (2 * rolling_std)
    df['BB_Lower'] = df['BB_MA20'] - (2 * rolling_std)
    ema12 = df['y'].ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = df['y'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

async def handle_upload(e):
    temp_path = None
    try:
        # 1. Obter o arquivo de forma universal
        upload_file = None
        if hasattr(e, 'files') and e.files:  # Para NiceGUI < 1.3.0
            upload_file = e.files[0]
        elif hasattr(e, 'content'):  # Para NiceGUI >= 1.3.0
            upload_file = e.content
        
        if not upload_file:
            ui.notify("Nenhum arquivo recebido ou formato inválido", type='negative')
            return

        # 2. Criar arquivo temporário de forma segura
        filename = upload_file.name.lower()
        file_ext = os.path.splitext(filename)[1]
        
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_path = temp_file.name
            if hasattr(upload_file, 'read'):
                temp_file.write(upload_file.read())
            else:
                temp_file.write(upload_file.getbuffer())

        # 3. Verificar extensão do arquivo
        if not (filename.endswith(('.csv', '.xls', '.xlsx'))):
            ui.notify("Formato de arquivo não suportado. Use CSV, XLS ou XLSX.", type='negative')
            return

        # 4. Tentar ler o arquivo conforme a extensão
        df_full = None
        try:
            if filename.endswith('.csv'):
                # Tentar ler CSV com diferentes encodings e delimitadores
                try:
                    df_full = pd.read_csv(temp_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df_full = pd.read_csv(temp_path, encoding='latin-1')
                    except Exception:
                        # Tentar detectar delimitador automaticamente
                        with open(temp_path, 'r') as f:
                            first_line = f.readline()
                        delimiter = ',' if ',' in first_line else ';' if ';' in first_line else '\t'
                        df_full = pd.read_csv(temp_path, sep=delimiter, encoding='latin-1')
            
            elif filename.endswith(('.xls', '.xlsx')):
                # Tentar ler Excel
                try:
                    df_full = pd.read_excel(temp_path, engine='openpyxl' if filename.endswith('.xlsx') else 'xlrd')
                except Exception as excel_error:
                    # Tentar ler todas as abas se houver erro
                    try:
                        xls = pd.ExcelFile(temp_path)
                        df_full = pd.concat([pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names])
                    except Exception:
                        raise Exception(f"Erro ao ler arquivo Excel: {str(excel_error)}")

        except Exception as read_error:
            ui.notify(f"Falha ao ler arquivo: {str(read_error)}", type='negative')
            return

        # 5. Verificar se o DataFrame foi criado com sucesso
        if df_full is None or df_full.empty:
            ui.notify("O arquivo está vazio ou não pôde ser interpretado", type='negative')
            return

        # 6. Normalizar nomes de colunas (case insensitive)
        df_full.columns = df_full.columns.str.lower().str.strip()
        
        # 7. Tentar identificar colunas automaticamente
        col_mapping = {}
        possible_date_cols = ['data', 'date', 'ref.date', 'datetime', 'time']
        possible_ticker_cols = ['ticker', 'ativo', 'symbol', 'codigo', 'papel', 'acao']
        possible_price_cols = ['preco', 'price', 'close', 'valor', 'fechamento', 'ultimo']
        
        # Mapear colunas automáticas
        for col in df_full.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in possible_date_cols):
                col_mapping['ds'] = col
            elif any(x in col_lower for x in possible_ticker_cols):
                col_mapping['ticker'] = col
            elif any(x in col_lower for x in possible_price_cols):
                col_mapping['y'] = col
        
        # Se não encontrou automaticamente, usar as primeiras colunas
        if not col_mapping:
            col_mapping = {
                'ds': df_full.columns[0],
                'ticker': df_full.columns[1] if len(df_full.columns) > 1 else None,
                'y': df_full.columns[2] if len(df_full.columns) > 2 else df_full.columns[1]
            }

        # 8. Processar os dados
        try:
            # Criar DataFrame padrão
            df_prophet = pd.DataFrame()
            df_prophet['ds'] = pd.to_datetime(df_full[col_mapping['ds']], errors='coerce')
            
            if 'ticker' in col_mapping and col_mapping['ticker'] in df_full.columns:
                df_prophet['ticker'] = df_full[col_mapping['ticker']]
            else:
                df_prophet['ticker'] = 'DEFAULT'
                
            df_prophet['y'] = pd.to_numeric(
                df_full[col_mapping['y']].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            
            # Remover linhas inválidas
            df_prophet = df_prophet.dropna(subset=['ds', 'y'])
            
        except Exception as processing_error:
            ui.notify(f"Erro no processamento: {str(processing_error)}", type='negative')
            return

        # 9. Atualizar os dados globais
        global tickers, df_prophet_base
        df_prophet_base = df_prophet
        tickers = df_prophet['ticker'].unique().tolist()
        
        # Atualizar a interface
        ticker_select.options = sorted(tickers)
        if tickers:
            ticker_select.value = tickers[0]
            ticker_select.update()
            update_charts()
            
        ui.notify(f"Arquivo carregado com sucesso! {len(tickers)} ativos encontrados.", type='positive')

    except Exception as e:
        ui.notify(f"Erro crítico: {str(e)}", type='negative')
    finally:
        # Garantir que o arquivo temporário seja removido
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

def update_charts():
    try:
        selected_ticker = ticker_select.value
        if not selected_ticker or 'df_prophet_base' not in globals():
            return
            
        df_prophet = df_prophet_base[df_prophet_base['ticker'] == selected_ticker][['ds', 'y']].copy()
        
        # Previsão
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=180)
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        fig.update_traces(
            selector=dict(name='y'),
            marker=dict(color=cores["pontos_reais"], size=4, opacity=0.8),
            line=dict(width=0)
        )
        fig.update_layout(
            title=f"Previsão de Preço da Ação ({selected_ticker})",
            xaxis_title="Data",
            yaxis_title="Preço (R$)",
            template="plotly_white",
            showlegend=True,
        )
        forecast_plot.update(fig)
        
        # Componentes
        fig_components = plot_components_plotly(model, forecast)
        fig_components.update_layout(
            title=f"Componentes da Previsão ({selected_ticker})",
            template="plotly_white"
        )
        components_plot.update(fig_components)
        
        # Indicadores Técnicos
        df_indicators = calculate_technical_indicators(df_prophet)
        
        # Médias Móveis
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['y'], name='Preço de Fechamento', line=dict(color=cores["preco"])))
        fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MA_20'], name='MA 20', line=dict(color=cores["linha_ma20"])))
        fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MA_50'], name='MA 50', line=dict(color=cores["linha_ma50"])))
        fig1.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MA_200'], name='MA 200', line=dict(color=cores["linha_ma200"])))
        fig1.update_layout(
            title="Preço e Médias Móveis",
            xaxis_title="Data",
            yaxis_title="Preço",
            template="plotly_white"
        )
        ma_plot.update(fig1)
        
        # MACD
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_indicators['ds'],
            y=df_indicators['MACD_Hist'],
            marker_color=np.where(df_indicators['MACD_Hist'] > 0, cores["macd_hist_acima"], cores["macd_hist_abaixo"]),
            name='Histograma MACD'
        ))
        fig2.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MACD'], name='MACD', line=dict(color=cores["macd"], width=2)))
        fig2.add_trace(go.Scatter(x=df_indicators['ds'], y=df_indicators['MACD_Signal'], name='Sinal', line=dict(color=cores["macd_sinal"], width=2)))
        fig2.update_layout(title="MACD", xaxis_title="Data", yaxis_title="Valor", template="plotly_white")
        macd_plot.update(fig2)
        
        # Bollinger Bands
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_indicators['ds'],
            y=df_indicators['y'],
            name='Preço',
            line=dict(color=cores["preco"], width=2)
        ))
        fig3.add_trace(go.Scatter(
            x=df_indicators['ds'],
            y=df_indicators['BB_MA20'],
            name='MA20',
            line=dict(color=cores["bb_ma20"], width=1)
        ))
        fig3.add_trace(go.Scatter(
            x=df_indicators['ds'],
            y=df_indicators['BB_Upper'],
            name='Banda Superior',
            line=dict(color=cores["bb_superior"], width=1, dash='dash')
        ))
        fig3.add_trace(go.Scatter(
            x=df_indicators['ds'],
            y=df_indicators['BB_Lower'],
            name='Banda Inferior',
            line=dict(color=cores["bb_inferior"], width=1, dash='dash'),
            fill='tonexty'
        ))
        fig3.update_layout(
            title="Bollinger Bands",
            xaxis_title="Data",
            yaxis_title="Preço",
            template="plotly_white"
        )
        bb_plot.update(fig3)
        
    except Exception as e:
        ui.notify(f"Erro ao gerar gráficos: {e}", type='negative')

# Variáveis globais
tickers = []
df_prophet_base = None

# Interface principal
with ui.header():
    ui.label("Análise de Ações com Prophet").classes("text-h4")

with ui.row().classes("w-full"):
    upload = ui.upload(label="Carregar arquivo de dados (CSV, XLS, XLSX)", 
                      on_upload=handle_upload,
                      auto_upload=True).classes("w-full")

with ui.row().classes("w-full"):
    ticker_select = ui.select(label="Selecione o ticker", 
                            options=[], 
                            on_change=update_charts).classes("w-full")

with ui.tabs().classes("w-full") as tabs:
    forecast_tab = ui.tab("Previsão")
    components_tab = ui.tab("Componentes")
    indicators_tab = ui.tab("Indicadores Técnicos")

with ui.tab_panels(tabs, value=forecast_tab).classes("w-full"):
    with ui.tab_panel(forecast_tab):
        forecast_plot = ui.plotly(figure=go.Figure()).classes("w-full h-96")
        
    with ui.tab_panel(components_tab):
        components_plot = ui.plotly(figure=go.Figure()).classes("w-full h-96")
        
    with ui.tab_panel(indicators_tab):
        with ui.column().classes("w-full"):
            ma_plot = ui.plotly(figure=go.Figure()).classes("w-full h-96")
            macd_plot = ui.plotly(figure=go.Figure()).classes("w-full h-96")
            bb_plot = ui.plotly(figure=go.Figure()).classes("w-full h-96")

ui.run(title="StockProphet", reload=False)

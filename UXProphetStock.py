import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import openpyxl
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.io as pio
import numpy as np
from tkhtmlview import HTMLLabel
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ProphetStockGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Ações com Prophet")
        self.root.geometry("1200x800")

        # Variáveis
        self.df = None
        self.ticker_var = tk.StringVar()
        self.file_path = tk.StringVar()

        # Criar widgets
        self.create_widgets()

    def create_widgets(self):
        # Frame externo para centralizar o conteúdo
        container_frame = ttk.Frame(self.root)
        container_frame.pack(expand=True, fill=tk.BOTH)

        # Frame central com controle
        control_frame = ttk.Frame(container_frame, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")

        # Configurar expansão para centralização
        container_frame.grid_rowconfigure(0, weight=1)
        container_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)

        # Linha 0 - Campo de entrada de arquivo
        ttk.Label(control_frame, text="Arquivo de dados:").grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Procurar", command=self.load_file).grid(row=0, column=2, padx=5, pady=5)

        # Linha 1 - Seleção de ticker
        ttk.Label(control_frame, text="Selecione o Ticker:").grid(row=1, column=0, sticky=tk.E, padx=5, pady=10)
        self.ticker_combobox = ttk.Combobox(control_frame, textvariable=self.ticker_var, state="readonly")
        self.ticker_combobox.grid(row=1, column=1, padx=5, pady=10, sticky="ew")
        ttk.Button(control_frame, text="Gerar Análise", command=self.generate_analysis).grid(row=1, column=2, padx=5, pady=10)

        # Notebook com abas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_forecast = ttk.Frame(self.notebook)
        self.tab_components = ttk.Frame(self.notebook)
        self.tab_indicators = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_forecast, text="Previsão")
        self.notebook.add(self.tab_components, text="Componentes")
        self.notebook.add(self.tab_indicators, text="Indicadores Técnicos")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo de dados",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )

        if file_path:
            self.file_path.set(file_path)
            try:
                self.df = pd.read_excel(file_path)
                self.preprocess_data()
                tickers = sorted(self.df['ticker'].unique())
                self.ticker_combobox['values'] = tickers
                if tickers:
                    self.ticker_var.set(tickers[0])
                messagebox.showinfo("Sucesso", "Dados carregados com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar arquivo:\n{str(e)}")

    def preprocess_data(self):
        self.df = self.df.rename(columns={
            'Empresa': 'empresa',
            'Setor': 'setor',
            'Subsetor': 'subsetor',
            'Tipo': 'tipo',
            False: 'valor.de.mercado'
        })
        self.df = self.df.rename(columns=lambda x: x.lower())

        price_columns = ['price.open', 'price.high', 'price.low', 'price.close', 'price.adjusted']
        for col in price_columns:
            self.df[col] = self.df[col].astype(str).str.replace(',', '.').astype(float)

        self.df['ref.date'] = pd.to_datetime(self.df['ref.date'])
        self.df['setor'] = self.df['setor'].replace(False, 'sem setor')
        self.df['subsetor'] = self.df['subsetor'].replace(False, 'sem subsetor')

    def generate_analysis(self):
        if self.df is None:
            messagebox.showerror("Erro", "Por favor, carregue um arquivo de dados primeiro.")
            return

        ticker = self.ticker_var.get()
        if not ticker:
            messagebox.showerror("Erro", "Por favor, selecione um ticker.")
            return

        try:
            df_ticker = self.df[self.df['ticker'] == ticker]
            if df_ticker.empty:
                messagebox.showerror("Erro", f"Nenhum dado encontrado para o ticker '{ticker}'.")
                return

            df_prophet = df_ticker[['ref.date', 'price.close']].rename(columns={'ref.date': 'ds', 'price.close': 'y'})
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=180)
            forecast = model.predict(future)

            self.plot_forecast(model, forecast, ticker)
            self.plot_components(model, forecast, ticker)
            self.plot_technical_indicators(df_prophet, ticker)

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao gerar a análise:\n{str(e)}")

    def plot_forecast(self, model, forecast, ticker):
        for widget in self.tab_forecast.winfo_children():
            widget.destroy()

        fig = plot_plotly(model, forecast)
        fig.update_layout(title=f"Previsão de Preço da Ação ({ticker})",
                          xaxis_title="Data",
                          yaxis_title="Preço (R$)",
                          template="plotly_white")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            fig.write_html(tmpfile.name)
            html_widget = HTMLLabel(self.tab_forecast, html=open(tmpfile.name, 'r', encoding='utf-8').read())
            html_widget.pack(fill=tk.BOTH, expand=True)
            self.temp_forecast_path = tmpfile.name

    def plot_components(self, model, forecast, ticker):
        for widget in self.tab_components.winfo_children():
            widget.destroy()

        fig = plot_components_plotly(model, forecast)
        fig.update_layout(title=f"Componentes da Previsão ({ticker})",
                          template="plotly_white")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            fig.write_html(tmpfile.name)
            html_widget = HTMLLabel(self.tab_components, html=open(tmpfile.name, 'r', encoding='utf-8').read())
            html_widget.pack(fill=tk.BOTH, expand=True)
            self.temp_components_path = tmpfile.name

    def plot_technical_indicators(self, df_ticker, ticker):
        for widget in self.tab_indicators.winfo_children():
            widget.destroy()

        df_ticker = self.calculate_technical_indicators(df_ticker)

        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'ggplot')
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df_ticker['ds'], df_ticker['y'], label='Preço de Fechamento')
        ax1.plot(df_ticker['ds'], df_ticker['MA_20'], label='MA 20')
        ax1.plot(df_ticker['ds'], df_ticker['MA_50'], label='MA 50')
        ax1.plot(df_ticker['ds'], df_ticker['MA_200'], label='MA 200')
        ax1.fill_between(df_ticker['ds'], df_ticker['BB_Upper'], df_ticker['BB_Lower'], alpha=0.1)
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(df_ticker['ds'], df_ticker['MACD_Hist'], color=np.where(df_ticker['MACD_Hist'] > 0, 'g', 'r'), alpha=0.3)
        ax2.plot(df_ticker['ds'], df_ticker['MACD'], label='MACD')
        ax2.plot(df_ticker['ds'], df_ticker['MACD_Signal'], label='Sinal')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(df_ticker['ds'], df_ticker['y'], label='Preço')
        ax3.plot(df_ticker['ds'], df_ticker['BB_MA20'], label='MA20')
        ax3.fill_between(df_ticker['ds'], df_ticker['BB_Upper'], df_ticker['BB_Lower'], alpha=0.2)
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_indicators)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def calculate_technical_indicators(self, df):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = ProphetStockGUI(root)
    root.mainloop()

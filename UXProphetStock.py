import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import openpyxl
from prophet import Prophet
import numpy as np
import os

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
        # Frame superior para controles
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Botão para carregar arquivo
        ttk.Label(control_frame, text="Arquivo de dados:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Procurar", command=self.load_file).grid(row=0, column=2)
        
        # Dropdown para selecionar ticker
        ttk.Label(control_frame, text="Selecione o Ticker:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.ticker_combobox = ttk.Combobox(control_frame, textvariable=self.ticker_var, state="readonly")
        self.ticker_combobox.grid(row=1, column=1, padx=5, pady=(10, 0), sticky=tk.W)
        
        # Botão para gerar análise
        ttk.Button(control_frame, text="Gerar Análise", command=self.generate_analysis).grid(row=1, column=2, pady=(10, 0))
        
        # Notebook para abas de gráficos
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Abas
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
        """Pré-processamento dos dados conforme o script original"""
        # Renomear colunas
        self.df = self.df.rename(columns={
            'Empresa': 'empresa',
            'Setor': 'setor',
            'Subsetor': 'subsetor',
            'Tipo': 'tipo',
            False: 'valor.de.mercado'
        })
        
        # Converter para minúsculas
        self.df = self.df.rename(columns=lambda x: x.lower())
        
        # Lista de colunas de preço que precisam da mesma conversão
        price_columns = [
            'price.open',
            'price.high',
            'price.low',
            'price.close',
            'price.adjusted'
        ]
        
        # Converter colunas de preço de string para float
        for col in price_columns:
            self.df[col] = self.df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Converter coluna de data
        self.df['ref.date'] = pd.to_datetime(self.df['ref.date'])
        
        # Substituir valores False
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
            # Filtrar dados para o ticker selecionado
            df_ticker = self.df[self.df['ticker'] == ticker]
            
            if df_ticker.empty:
                messagebox.showerror("Erro", f"Nenhum dado encontrado para o ticker '{ticker}'.")
                return
                
            # Preparar dados para o Prophet
            df_prophet = df_ticker[['ref.date', 'price.close']].rename(
                columns={'ref.date': 'ds', 'price.close': 'y'})
            
            # Treinar modelo
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            
            # Criar datas futuras para previsão
            future = model.make_future_dataframe(periods=180)
            forecast = model.predict(future)
            
            # Plotar gráficos
            self.plot_forecast(model, forecast, ticker)
            self.plot_components(model, forecast, ticker)
            self.plot_technical_indicators(df_prophet, ticker)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao gerar a análise:\n{str(e)}")
    
    def plot_forecast(self, model, forecast, ticker):
        """Plotar gráfico de previsão"""
        # Limpar aba anterior
        for widget in self.tab_forecast.winfo_children():
            widget.destroy()
            
        # Criar figura
        fig = model.plot(forecast)
        plt.title(f"Previsão de Preço da Ação ({ticker})", fontsize=16, pad=20)
        plt.xlabel("Data", fontsize=12)
        plt.ylabel("Preço (R$)", fontsize=12)
        plt.legend(['Dados Observados', 'Previsão', 'Intervalo de Confiança'], loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar figura ao Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.tab_forecast)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def plot_components(self, model, forecast, ticker):
        """Plotar componentes da previsão"""
        # Limpar aba anterior
        for widget in self.tab_components.winfo_children():
            widget.destroy()
            
        # Criar figura
        fig = model.plot_components(forecast)
        plt.suptitle(f"Componentes da Previsão para {ticker}", y=0.99, fontsize=16)
        plt.tight_layout()
        
        # Adicionar figura ao Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.tab_components)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_technical_indicators(self, df_ticker, ticker):
        """Plotar indicadores técnicos"""
        # Limpar aba anterior
        for widget in self.tab_indicators.winfo_children():
            widget.destroy()
            
        # Calcular indicadores
        df_ticker = self.calculate_technical_indicators(df_ticker)
        
        # Configurar estilo
        available_styles = plt.style.available
        modern_style = 'seaborn-v0_8' if 'seaborn-v0_8' in available_styles else 'ggplot'
        plt.style.use(modern_style)
        
        # Criar figura
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        
        # Gráfico 1: Preço com Médias Móveis e Bandas de Bollinger
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title('1. Preço com Médias Móveis e Bandas de Bollinger', fontsize=12, pad=15, loc='left', fontweight='bold')
        ax1.plot(df_ticker['ds'], df_ticker['y'], label='Preço de Fechamento', color='#1f77b4', linewidth=2, alpha=0.9)
        ax1.plot(df_ticker['ds'], df_ticker['MA_20'], label='Média Móvel 20 dias', color='#ff7f0e', linestyle='--', linewidth=1.5)
        ax1.plot(df_ticker['ds'], df_ticker['MA_50'], label='Média Móvel 50 dias', color='#2ca02c', linestyle='-.', linewidth=1.5)
        ax1.plot(df_ticker['ds'], df_ticker['MA_200'], label='Média Móvel 200 dias', color='#d62728', linestyle=':', linewidth=1.5)
        
        # Preencher Bandas de Bollinger
        ax1.fill_between(df_ticker['ds'], df_ticker['BB_Upper'], df_ticker['BB_Lower'],
                        color='#9467bd', alpha=0.1, label='Bandas Bollinger')
        
        ax1.set_title(f'Análise Técnica: {ticker}', fontsize=14, pad=20, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_facecolor('whitesmoke')
        
        # Gráfico 2: MACD
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_title('2. MACD (12,26,9) com Linha de Sinal e Histograma', fontsize=12, pad=15, loc='left', fontweight='bold')
        
        # Plotar o histograma
        ax2.bar(df_ticker['ds'], df_ticker['MACD_Hist'],
              color=np.where(df_ticker['MACD_Hist'] > 0, '#2ca02c', '#d62728'),
              alpha=0.3, width=0.8, label='Histograma MACD')
        
        # Plotar as linhas
        ax2.plot(df_ticker['ds'], df_ticker['MACD'], label='MACD (12,26)', color='#17becf', linewidth=1.5)
        ax2.plot(df_ticker['ds'], df_ticker['MACD_Signal'], label='Linha de Sinal (9)', color='#f14cc1', linewidth=1.5)
        
        # Linha zero
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.7)
        
        ax2.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), frameon=True, framealpha=0.9, edgecolor='black')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_facecolor('whitesmoke')
        ax2.set_ylabel('Valor MACD', fontsize=10)
        
        # Gráfico 3: Bandas de Bollinger em Destaque
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_title('3. Bandas de Bollinger (20,2) com Preço e Média Central', fontsize=12, pad=15, loc='left', fontweight='bold')
        ax3.plot(df_ticker['ds'], df_ticker['y'], label='Preço de Fechamento', color='#1f77b4', linewidth=1.5, alpha=0.9)
        ax3.plot(df_ticker['ds'], df_ticker['BB_MA20'], label='Média Central (20 dias)', color='#9467bd', linestyle='--', linewidth=1.5)
        ax3.fill_between(df_ticker['ds'], df_ticker['BB_Upper'], df_ticker['BB_Lower'],
                        color='#9467bd', alpha=0.2, label='Faixa das Bandas (2σ)')
        ax3.legend(loc='upper left', framealpha=0.9)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.set_facecolor('whitesmoke')
        ax3.set_ylabel('Preço (R$)', fontsize=10)
        ax3.set_xlabel('Data', fontsize=10)
        
        plt.tight_layout()
        
        # Adicionar figura ao Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.tab_indicators)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def calculate_technical_indicators(self, df):
        """Calcular indicadores técnicos"""
        df = df.copy()
        df['y'] = df['y'].ffill()  # Preencher valores faltantes
        
        # 1. Médias Móveis
        df['MA_20'] = df['y'].rolling(window=20, min_periods=1).mean()
        df['MA_50'] = df['y'].rolling(window=50, min_periods=1).mean()
        df['MA_200'] = df['y'].rolling(window=200, min_periods=1).mean()
        
        # 2. Bandas de Bollinger
        df['BB_MA20'] = df['y'].rolling(window=20, min_periods=1).mean()
        rolling_std = df['y'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_MA20'] + (2 * rolling_std)
        df['BB_Lower'] = df['BB_MA20'] - (2 * rolling_std)
        
        # 3. MACD
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
import json
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
import csv
import pytz
import time as time_module
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
from gnews import GNews
import pandas_market_calendars as mcal
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gym
from gym import spaces
import backtrader as bt
import optuna
import traceback
import warnings
import logging
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
import random
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataFeed:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_live_prices(self):
        """Fetch live prices for specified tickers using yfinance."""
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="1d", interval="1m")
                if not df.empty:
                    latest = df.iloc[-1]
                    data[ticker] = {
                        "price": latest["Close"],
                        "volume": latest["Volume"]
                    }
                else:
                    logger.warning(f"No data returned for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }

class VirtualPortfolio:
    def __init__(self, config):
        self.starting_balance = config["starting_balance"]
        self.cash = self.starting_balance
        self.holdings = {}  # {asset: {qty, avg_price}}
        self.trade_log = []
        self.api = REST(
            key_id=config["alpaca_api_key"],
            secret_key=config["alpaca_api_secret"],
            base_url=config["base_url"]
        )
        self.config = config
        self.portfolio_file = "data/portfolio.json"
        self.trade_log_file = "data/trade_log.json"
        self.initialize_files()

    def initialize_files(self):
        """Initialize portfolio and trade log JSON files if they don't exist."""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, "w") as f:
                json.dump({"cash": self.cash, "holdings": self.holdings}, f, indent=4)
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, "w") as f:
                json.dump([], f, indent=4)

    def initialize_portfolio(self, balance=None):
        """Reset or initialize portfolio with a given balance."""
        if balance is not None:
            self.starting_balance = balance
        self.cash = self.starting_balance
        self.holdings = {}
        self.trade_log = []
        self.save_portfolio()
        self.save_trade_log()

    def buy(self, asset, qty, price):
        """Execute a buy order in paper trading mode."""
        cost = qty * price
        if cost > self.cash:
            logger.warning(f"Insufficient cash for buy order: {asset}, qty: {qty}, price: {price}")
            return False
        try:
            self.api.submit_order(
                symbol=asset,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            self.cash -= cost
            if asset in self.holdings:
                current_qty = self.holdings[asset]["qty"]
                current_avg_price = self.holdings[asset]["avg_price"]
                new_qty = current_qty + qty
                new_avg_price = ((current_avg_price * current_qty) + (price * qty)) / new_qty
                self.holdings[asset] = {"qty": new_qty, "avg_price": new_avg_price}
            else:
                self.holdings[asset] = {"qty": qty, "avg_price": price}
            self.log_trade({"asset": asset, "action": "buy", "qty": qty, "price": price, "timestamp": str(datetime.now())})
            self.save_portfolio()
            return True
        except Exception as e:
            logger.error(f"Error executing buy order for {asset}: {e}")
            return False

    def sell(self, asset, qty, price):
        """Execute a sell order in paper trading mode."""
        if asset not in self.holdings or self.holdings[asset]["qty"] < qty:
            logger.warning(f"Insufficient holdings for sell order: {asset}, qty: {qty}")
            return False
        try:
            self.api.submit_order(
                symbol=asset,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
            revenue = qty * price
            self.cash += revenue
            current_qty = self.holdings[asset]["qty"]
            if current_qty == qty:
                del self.holdings[asset]
            else:
                self.holdings[asset]["qty"] -= qty
            self.log_trade({"asset": asset, "action": "sell", "qty": qty, "price": price, "timestamp": str(datetime.now())})
            self.save_portfolio()
            return True
        except Exception as e:
            logger.error(f"Error executing sell order for {asset}: {e}")
            return False

    def get_value(self, current_prices):
        """Calculate total portfolio value based on current prices."""
        total_value = self.cash
        for asset, data in self.holdings.items():
            price = current_prices.get(asset, {}).get("price", 0)
            total_value += data["qty"] * price
        return total_value

    def get_metrics(self):
        """Return portfolio metrics including PnL and exposure."""
        current_prices = self.get_current_prices()
        metrics = {
            "cash": self.cash,
            "holdings": self.holdings,
            "total_value": self.get_value(current_prices),
            "current_portfolio_value": self.config.get("current_portfolio_value", 0),  # Add from config
            "current_pnl": self.config.get("current_pnl", 0),  # Add from config
            "realized_pnl": sum(
                (t["price"] - self.holdings.get(t["asset"], {}).get("avg_price", t["price"])) * t["qty"]
                for t in self.trade_log if t["action"] == "sell"
            ),
            "unrealized_pnl": sum(
                (current_prices.get(asset, {}).get("price", 0) - data["avg_price"]) * data["qty"]
                for asset, data in self.holdings.items()
            ),
            "total_exposure": sum(
                data["qty"] * current_prices.get(asset, {}).get("price", 0)
                for asset, data in self.holdings.items()
            )
        }
        return metrics
    
    def log_trade(self, trade):
        """Log a trade to the trade log file."""
        self.trade_log.append(trade)
        self.save_trade_log()

    def save_portfolio(self):
        """Save portfolio state to JSON file."""
        try:
            with open(self.portfolio_file, "w") as f:
                json.dump({"cash": self.cash, "holdings": self.holdings}, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def save_trade_log(self):
        """Save trade log to JSON file."""
        try:
            with open(self.trade_log_file, "w") as f:
                json.dump(self.trade_log, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")

    def get_current_prices(self):
        """Fetch current prices from Alpaca API."""
        prices = {}
        for asset in self.holdings:
            try:
                bar = self.api.get_bars(asset, timeframe="1Min", limit=1).df
                if not bar.empty:
                    prices[asset] = {"price": bar["close"].iloc[-1], "volume": bar["volume"].iloc[-1]}
            except Exception as e:
                logger.error(f"Error fetching price for {asset}: {e}")
        return prices

class PaperExecutor:
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.mode = config["mode"]

    def execute_trade(self, asset, action, qty, price):
        """Execute a simulated trade in paper mode."""
        if self.mode != "paper":
            raise ValueError("Executor is not in paper trading mode")
        if action == "buy":
            return self.portfolio.buy(asset, qty, price)
        elif action == "sell":
            return self.portfolio.sell(asset, qty, price)
        else:
            logger.error(f"Invalid action: {action}")
            return False

class PerformanceReport:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_report(self):
        """Generate a daily performance report."""
        metrics = self.portfolio.get_metrics()
        total_value = metrics["total_value"]
        starting_value = self.portfolio.starting_balance
        daily_roi = ((total_value / starting_value) - 1) * 100
        cumulative_roi = daily_roi  # Simplified for daily report

        # Calculate Sharpe Ratio (simplified)
        returns = [t["price"] for t in self.portfolio.trade_log if t["action"] == "sell"]
        if len(returns) > 1:
            returns = np.array(returns)
            sharpe_ratio = (np.mean(returns) - 0.02) / np.std(returns) if np.std(returns) != 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate Max Drawdown
        values = [starting_value] + [metrics["total_value"]]
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "roi_today": daily_roi,
            "cumulative_roi": cumulative_roi,
            "sharpe": sharpe_ratio,
            "drawdown": max_drawdown,
            "trades_executed": len(self.portfolio.trade_log)
        }

        report_file = os.path.join(self.report_dir, f"report_{datetime.now().strftime('%Y%m%d')}.json")
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Saved report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        return report

class PortfolioTracker:
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.config = config

    def log_metrics(self):
        """Log portfolio metrics once."""
        try:
            metrics = self.portfolio.get_metrics()
            logger.info(f"Portfolio Metrics:")
            logger.info(f"Cash: ${metrics['cash']:.2f}")
            logger.info(f"Holdings: {metrics['holdings']}")
            logger.info(f"Total Value: ${metrics['total_value']:.2f}")
            logger.info(f"Current Portfolio Value (Alpaca): ${self.config.get('current_portfolio_value', 0):.2f}")
            logger.info(f"Current PnL (Alpaca): ${self.config.get('current_pnl', 0):.2f}")
            logger.info(f"Realized PnL: ${metrics['realized_pnl']:.2f}")
            logger.info(f"Unrealized PnL: ${metrics['unrealized_pnl']:.2f}")
            logger.info(f"Total Exposure: ${metrics['total_exposure']:.2f}")
        except Exception as e:
            logger.error(f"Error logging portfolio metrics: {e}")
            
            
class Stock:
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"
    FMPC_API_KEY = os.getenv("FMPC_API_KEY")
    MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
    SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
    SANTIMENT_API = "https://api.santiment.net"
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    def __init__(self, reddit_client_id=None, reddit_client_secret=None, reddit_user_agent=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        if reddit_client_id and reddit_client_secret and reddit_user_agent:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
        else:
            self.reddit = None
        self.COINGECKO_API = "https://api.coingecko.com/api/v3"
        self.last_rate_fetch = None
        self.rate_cache = None
        self.cache_duration = 300
        self.google_news = GNews()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warnings.filterwarnings('ignore')

    def fetch_exchange_rates(self):
        """Fetch real-time exchange rates for INR, EUR, BTC, ETH"""
        if (self.last_rate_fetch and 
            self.rate_cache and 
            (datetime.now() - self.last_rate_fetch).total_seconds() < self.cache_duration):
            return self.rate_cache

        try:
            url = f"{self.COINGECKO_API}/simple/price?ids=bitcoin,ethereum&vs_currencies=usd,inr,eur"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            btc_usd = data.get("bitcoin", {}).get("usd", 1)
            rates = {
                "bitcoin": {"usd": btc_usd},
                "ethereum": {"usd": data.get("ethereum", {}).get("usd", 1)},
                "inr": {"usd": data.get("bitcoin", {}).get("inr", 1) / btc_usd if btc_usd != 0 else 83},
                "eur": {"usd": data.get("bitcoin", {}).get("eur", 1) / btc_usd if btc_usd != 0 else 0.95}
            }

            self.rate_cache = rates
            self.last_rate_fetch = datetime.now()
            return rates
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {e}")
            fallback = {
                "bitcoin": {"usd": 85000},
                "ethereum": {"usd": 1633},
                "inr": {"usd": 86},
                "eur": {"usd": 0.95}
            }
            self.rate_cache = fallback
            self.last_rate_fetch = datetime.now()
            return fallback

    def convert_price(self, price, exchange_rates):
        """Convert price to different currencies including crypto and fiat"""
        if not isinstance(price, (int, float)) or price == "N/A":
            return {"USD": price}
        try:
            return {
                "USD": round(float(price), 2),
                "INR": round(float(price) * exchange_rates.get("inr", {}).get("usd", 1), 2),
                "EUR": round(float(price) * exchange_rates.get("eur", {}).get("usd", 1), 2),
                "BTC": round(float(price) / exchange_rates.get("bitcoin", {}).get("usd", 1), 8),
                "ETH": round(float(price) / exchange_rates.get("ethereum", {}).get("usd", 1), 8)
            }
        except Exception as e:
            logger.error(f"Error converting price: {e}")
            return {
                "USD": round(float(price), 2),
                "INR": price,
                "EUR": price,
                "BTC": price,
                "ETH": price
            }

    def convert_np_types(self, data):
        if isinstance(data, dict):
            return {k: self.convert_np_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_np_types(item) for item in data]
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif pd.isna(data):
            return None
        else:
            return data

    def alpha_vantage_sentiment(self, ticker):
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            if "feed" in data:
                for article in data["feed"][:10]:
                    sentiment_score = float(article.get("overall_sentiment_score", 0))
                    if sentiment_score > 0.1:
                        sentiments["positive"] += 1
                    elif sentiment_score < -0.1:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def reddit_sentiment(self, ticker):
        if not self.reddit:
            return {"positive": 0, "negative": 0, "neutral": 0}
        try:
            subreddit = self.reddit.subreddit("all")
            query = f"${ticker}"
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            for submission in subreddit.search(query, limit=10):
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]:
                    sentiment = self.sentiment_analyzer.polarity_scores(comment.body)
                    if sentiment["compound"] > 0.1:
                        sentiments["positive"] += 1
                    elif sentiment["compound"] < -0.1:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def google_news_sentiment(self, ticker):
        try:
            news = self.google_news.get_news(ticker)
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            for article in news[:10]:
                description = article.get("description", "")
                sentiment = self.sentiment_analyzer.polarity_scores(description)
                if sentiment["compound"] > 0.1:
                    sentiments["positive"] += 1
                elif sentiment["compound"] < -0.1:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Google News sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

 
    def fetch_combined_sentiment(self, ticker):
        try:
            alpha_sentiment = self.alpha_vantage_sentiment(ticker)
            reddit_sentiment = self.reddit_sentiment(ticker)
            google_sentiment = self.google_news_sentiment(ticker)

            aggregated = {
                "positive": (alpha_sentiment["positive"] + reddit_sentiment["positive"] +
                            google_sentiment["positive"] ),
                "negative": (alpha_sentiment["negative"] + reddit_sentiment["negative"] +
                            google_sentiment["negative"]  ),
                "neutral": (alpha_sentiment["neutral"] + reddit_sentiment["neutral"] +
                           google_sentiment["neutral"]  )
            }

            return {
                "alpha_vantage": alpha_sentiment,
                "reddit": reddit_sentiment,
                "google_news": google_sentiment,
                "aggregated": aggregated
            }
        except Exception as e:
            logger.error(f"Error fetching combined sentiment: {e}")
            return {
                "alpha_vantage": {"positive": 0, "negative": 0, "neutral": 0},
                "reddit": {"positive": 0, "negative": 0, "neutral": 0},
                "google_news": {"positive": 0, "negative": 0, "neutral": 0},
                "fmp": {"positive": 0, "negative": 0, "neutral": 0},
                "aggregated": {"positive": 0, "negative": 0, "neutral": 0}
            }

    def _generate_detailed_recommendation(self, stock_data, recommendation, buy_score, sell_score,
                                        price_to_sma200, trend_direction, sentiment_score,
                                        volatility, sharpe_ratio):
        explanation = f"Recommendation for {stock_data['name']} ({stock_data['symbol']}): {recommendation}\n"
        explanation += f"Current Price: ${stock_data['current_price']['USD']:.2f}\n\n"
        explanation += f"Buy Score: {buy_score:.3f}, Sell Score: {sell_score:.3f}\n\n"

        if recommendation in ["STRONG BUY", "BUY"]:
            explanation += "Bullish Factors:\n"
            if price_to_sma200 < 1:
                explanation += "- Price is below the 200-day SMA, indicating potential undervaluation.\n"
            if trend_direction == "UPTREND":
                explanation += "- Stock is in an uptrend (50-day SMA > 200-day SMA).\n"
            if sentiment_score > 0.6:
                explanation += "- Positive market sentiment from news and social media.\n"
            if sharpe_ratio > 0:
                explanation += "- Positive risk-adjusted return (Sharpe Ratio).\n"
        elif recommendation in ["STRONG SELL", "SELL"]:
            explanation += "Bearish Factors:\n"
            if price_to_sma200 > 1:
                explanation += "- Price is above the 200-day SMA, suggesting potential overvaluation.\n"
            if trend_direction == "DOWNTREND":
                explanation += "- Stock is in a downtrend (50-day SMA < 200-day SMA).\n"
            if sentiment_score < 0.4:
                explanation += "- Negative market sentiment from news and social media.\n"
            if sharpe_ratio < 0:
                explanation += "- Negative risk-adjusted return (Sharpe Ratio).\n"
        else:
            explanation += "Neutral Factors:\n"
            explanation += "- Price is stable relative to moving averages.\n"
            explanation += "- Sentiment is balanced, with no strong bullish or bearish signals.\n"

        explanation += f"\nRisk Assessment:\n"
        explanation += f"- Volatility: {volatility:.4f} (higher values indicate higher risk)\n"
        explanation += f"- Sector: {stock_data.get('sector', 'N/A')}\n"
        explanation += f"- Industry: {stock_data.get('industry', 'N/A')}\n"

        return explanation

    def convert_df_to_dict(self, df):
        if df is None or df.empty:
            return {}
        try:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.where(pd.notnull(df), None)
            result = df.to_dict(orient='index')
            return {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in result.items()}
        except Exception as e:
            logger.error(f"Error converting DataFrame to dict: {e}")
            return {}

    def income_statement(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.financials
            if income_stmt.empty:
                return {"success": False, "message": f"No income statement data for {ticker}"}
            income_dict = self.convert_df_to_dict(income_stmt)
            return {
                "success": True,
                "income_statement": income_dict
            }
        except Exception as e:
            logger.error(f"Error fetching income statement: {e}")
            return {"success": False, "message": f"Error fetching income statement: {str(e)}"}

    def balance_sheet(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            balance = stock.balance_sheet
            if balance.empty:
                return {"success": False, "message": f"No balance sheet data for {ticker}"}
            balance_dict = self.convert_df_to_dict(balance)
            return {
                "success": True,
                "balance_sheet": balance_dict
            }
        except Exception as e:
            logger.error(f"Error fetching balance sheet: {e}")
            return {"success": False, "message": f"Error fetching balance sheet: {str(e)}"}

    def cash_flow(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            cashflow = stock.cashflow
            if cashflow.empty:
                return {"success": False, "message": f"No cash flow data for {ticker}"}
            cashflow_dict = self.convert_df_to_dict(cashflow)
            return {
                "success": True,
                "cash_flow": cashflow_dict
            }
        except Exception as e:
            logger.error(f"Error fetching cash flow: {e}")
            return {"success": False, "message": f"Error fetching cash flow: {str(e)}"}

    def calculate_mpt_metrics(self, stock_history, benchmark_tickers):
        try:
            stock_returns = stock_history["Close"].pct_change().dropna()
            if stock_returns.empty:
                return {
                    "annual_return": "N/A",
                    "annual_volatility": "N/A",
                    "sharpe_ratio": "N/A",
                    "beta": "N/A",
                    "alpha": "N/A"
                }

            annual_return = stock_returns.mean() * 252
            annual_volatility = stock_returns.std() * np.sqrt(252)
            risk_free_rate = 0.02
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else "N/A"

            beta = "N/A"
            alpha = "N/A"
            for benchmark_ticker in benchmark_tickers:
                benchmark = yf.Ticker(benchmark_ticker)
                benchmark_history = benchmark.history(period="2y")
                benchmark_returns = benchmark_history["Close"].pct_change().dropna()

                if not benchmark_returns.empty:
                    aligned_returns = pd.concat([stock_returns, benchmark_returns], axis=1, join='inner')
                    if not aligned_returns.empty:
                        stock_ret = aligned_returns.iloc[:, 0]
                        bench_ret = aligned_returns.iloc[:, 1]
                        covariance = stock_ret.cov(bench_ret)
                        benchmark_variance = bench_ret.var()
                        beta = covariance / benchmark_variance if benchmark_variance != 0 else "N/A"
                        market_return = bench_ret.mean() * 252
                        alpha = annual_return - (risk_free_rate + beta * (market_return - risk_free_rate)) if beta != "N/A" else "N/A"
                        break

            return {
                "annual_return": float(annual_return) if not pd.isna(annual_return) else "N/A",
                "annual_volatility": float(annual_volatility) if not pd.isna(annual_volatility) else "N/A",
                "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio != "N/A" else "N/A",
                "beta": float(beta) if beta != "N/A" else "N/A",
                "alpha": float(alpha) if alpha != "N/A" else "N/A"
            }
        except Exception as e:
            logger.error(f"Error calculating MPT metrics: {e}")
            return {
                "annual_return": "N/A",
                "annual_volatility": "N/A",
                "sharpe_ratio": "N/A",
                "beta": "N/A",
                "alpha": "N/A"
            }

    def generate_adversarial_financial_data(self, history, epsilon=0.05, noise_factor=0.1, event_prob=0.1):
        try:
            adv_history = history.copy()
            price_std = history['Close'].std()
            volume_std = history['Volume'].std()
            
            for i in range(len(adv_history)):
                perturbation = np.random.uniform(-epsilon, epsilon) * price_std
                noise = np.random.normal(0, noise_factor * price_std)
                
                adv_history['Close'].iloc[i] += perturbation + noise
                adv_history['Open'].iloc[i] += perturbation + noise
                adv_history['High'].iloc[i] = max(adv_history['High'].iloc[i] + perturbation + noise, 
                                                adv_history['Open'].iloc[i], 
                                                adv_history['Close'].iloc[i])
                adv_history['Low'].iloc[i] = min(adv_history['Low'].iloc[i] + perturbation + noise, 
                                               adv_history['Open'].iloc[i], 
                                               adv_history['Close'].iloc[i])
                
                volume_perturbation = np.random.uniform(-epsilon, epsilon) * volume_std
                adv_history['Volume'].iloc[i] = max(0, adv_history['Volume'].iloc[i] + volume_perturbation)
                
                if np.random.random() < event_prob:
                    event_type = np.random.choice(['crash', 'spike'])
                    if event_type == 'crash':
                        drop_factor = np.random.uniform(0.85, 0.95)
                        adv_history['Close'].iloc[i] *= drop_factor
                        adv_history['Open'].iloc[i] *= drop_factor
                        adv_history['High'].iloc[i] *= drop_factor
                        adv_history['Low'].iloc[i] *= drop_factor
                        adv_history['Volume'].iloc[i] *= 1.5
                    else:
                        spike_factor = np.random.uniform(1.05, 1.15)
                        adv_history['Close'].iloc[i] *= spike_factor
                        adv_history['Open'].iloc[i] *= spike_factor
                        adv_history['High'].iloc[i] *= spike_factor
                        adv_history['Low'].iloc[i] *= spike_factor
                        adv_history['Volume'].iloc[i] *= 1.3
            
            adv_history['Close'] = adv_history['Close'].clip(lower=0.01)
            adv_history['Open'] = adv_history['Open'].clip(lower=0.01)
            adv_history['High'] = adv_history['High'].clip(lower=0.01)
            adv_history['Low'] = adv_history['Low'].clip(lower=0.01)
            adv_history['Volume'] = adv_history['Volume'].clip(lower=0)
            
            return adv_history
        
        except Exception as e:
            logger.error(f"Error generating adversarial data: {e}")
            return history

    def train_rl_with_adversarial_events(self, history, ml_predicted_price, current_price, 
                                       num_episodes=100, adversarial_freq=0.2, max_event_magnitude=0.1):
        try:
            #Ensure technical indicators are computed in the history DataFrame
            history = history.copy()  # Avoid modifying the original DataFrame
            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()

            # Calculate RSI
            def calculate_rsi(data, periods=14):
                delta = data.diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(com=periods-1, adjust=False).mean()
                roll_down = down.ewm(com=periods-1, adjust=False).mean()
                rs = roll_up / roll_down
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return rsi

            history["RSI"] = calculate_rsi(history["Close"])

            # Calculate MACD
            exp1 = history["Close"].ewm(span=12, adjust=False).mean()
            exp2 = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = exp1 - exp2

            # Calculate Volatility
            history["Daily_Return"] = history["Close"].pct_change()
            history["Volatility"] = history["Daily_Return"].rolling(window=30).std()

            # Fill NaN values with reasonable defaults
            history.fillna({
                "SMA_50": history["Close"],
                "SMA_200": history["Close"],
                "RSI": 50,
                "MACD": 0,
                "Volatility": 0
            }, inplace=True)
            class AdversarialStockTradingEnv(gym.Env):
                def __init__(self, history, current_price, ml_predicted_price, 
                           adversarial_freq, max_event_magnitude):
                    super(AdversarialStockTradingEnv, self).__init__()
                    self.history = history
                    self.current_price = current_price
                    self.ml_predicted_price = ml_predicted_price
                    self.adversarial_freq = adversarial_freq
                    self.max_event_magnitude = max_event_magnitude
                    self.max_steps = len(history) - 1
                    self.current_step = 0
                    self.initial_balance = 10000
                    self.balance = self.initial_balance
                    self.shares_held = 0
                    self.net_worth = self.initial_balance
                    self.max_shares = 100
                    
                    self.action_space = spaces.Discrete(3)
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
                    )
                    
                def reset(self):
                    self.current_step = 0
                    self.balance = self.initial_balance
                    self.shares_held = 0
                    self.net_worth = self.initial_balance
                    self.event_occurred = 0
                    return self._get_observation()
                
                def _get_observation(self):
                    price = float(self.history["Close"].iloc[self.current_step])
                    sma_50 = float(self.history["SMA_50"].iloc[self.current_step] or 0)
                    sma_200 = float(self.history["SMA_200"].iloc[self.current_step] or 0)
                    rsi = float(self.history["RSI"].iloc[self.current_step] or 50)
                    macd = float(self.history["MACD"].iloc[self.current_step] or 0)
                    volatility = float(self.history["Volatility"].iloc[self.current_step] or 0)
                    ml_pred = self.ml_predicted_price if self.current_step == self.max_steps else price
                    return np.array([
                        price, sma_50, sma_200, rsi, macd, volatility,
                        self.balance, self.shares_held, self.net_worth, ml_pred,
                        self.event_occurred
                    ], dtype=np.float32)
                
                def step(self, action):
                    current_price = float(self.history["Close"].iloc[self.current_step])
                    reward = 0
                    
                    if np.random.random() < self.adversarial_freq:
                        event_magnitude = np.random.uniform(-self.max_event_magnitude, 
                                                          self.max_event_magnitude)
                        current_price *= (1 + event_magnitude)
                        self.event_occurred = abs(event_magnitude)
                    else:
                        self.event_occurred = 0
                    
                    if action == 1:
                        shares_to_buy = min(self.max_shares - self.shares_held, 
                                          int(self.balance / current_price))
                        cost = shares_to_buy * current_price
                        if cost <= self.balance:
                            self.balance -= cost
                            self.shares_held += shares_to_buy
                    elif action == 2:
                        shares_to_sell = self.shares_held
                        if shares_to_sell > 0:
                            revenue = shares_to_sell * current_price
                            self.balance += revenue
                            self.shares_held = 0
                            
                    self.net_worth = self.balance + self.shares_held * current_price
                    reward = self.net_worth - self.initial_balance
                    
                    self.current_step += 1
                    done = self.current_step >= self.max_steps
                    
                    if done:
                        reward += (self.ml_predicted_price - current_price) * self.shares_held
                        
                    return self._get_observation(), reward, done, {}
            
            env = AdversarialStockTradingEnv(
                history=history,
                current_price=current_price,
                ml_predicted_price=ml_predicted_price,
                adversarial_freq=adversarial_freq,
                max_event_magnitude=max_event_magnitude
            )
            
            class AdversarialQLearningAgent:
                def __init__(self, state_size, action_size):
                    self.state_size = state_size
                    self.action_size = action_size
                    self.q_table = {}
                    self.alpha = 0.1
                    self.gamma = 0.95
                    self.epsilon = 0.1
                    self.event_threshold = 0.05
                    
                def discretize_state(self, state):
                    rounded_state = np.round(state[:-1], 2)
                    event = 1 if state[-1] > self.event_threshold else 0
                    return tuple(np.append(rounded_state, event))
                
                def get_action(self, state):
                    state_key = self.discretize_state(state)
                    if np.random.random() < self.epsilon:
                        return np.random.randint(self.action_size)
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    return np.argmax(self.q_table[state_key])
                
                def update(self, state, action, reward, next_state):
                    state_key = self.discretize_state(state)
                    next_state_key = self.discretize_state(next_state)
                    
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    if next_state_key not in self.q_table:
                        self.q_table[next_state_key] = np.zeros(self.action_size)
                        
                    q_value = self.q_table[state_key][action]
                    next_max = np.max(self.q_table[next_state_key])
                    new_q_value = q_value + self.alpha * (reward + self.gamma * next_max - q_value)
                    self.q_table[state_key][action] = new_q_value
            
            agent = AdversarialQLearningAgent(state_size=11, action_size=3)
            
            total_rewards = []
            event_counts = []
            epoch_logs = []
            
            logger.info(f"Training adversarial RL agent...")
            for episode in range(num_episodes):
                state = env.reset()
                total_reward = 0
                done = False
                episode_events = 0
                
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if state[-1] > 0:
                        episode_events += 1
                        
                total_rewards.append(total_reward)
                event_counts.append(episode_events)
                if (episode + 1) % 10 == 0:
                    logger.info(f"Adversarial Episode {episode + 1}/{num_episodes}, "
                                f"Total Reward: {total_reward:.2f}, Events: {episode_events}")
                    epoch_logs.append({
                        "episode": episode + 1,
                        "total_reward": total_reward,
                        "average_reward": np.mean(total_rewards[-10:]),
                        "events_triggered": episode_events
                    })
            
            state = env.reset()
            done = False
            actions_taken = []
            net_worth_history = []
            
            while not done:
                action = agent.get_action(state)
                actions_taken.append(action)
                next_state, reward, done, _ = env.step(action)
                net_worth_history.append(env.net_worth)
                state = next_state
                
            final_net_worth = net_worth_history[-1]
            performance = (final_net_worth - env.initial_balance) / env.initial_balance * 100
            
            recommendation = "BUY" if performance > 10 else "SELL" if performance < -5 else "HOLD"
            
            return {
                "success": True,
                "recommendation": recommendation,
                "performance_pct": float(performance),
                "final_net_worth": float(final_net_worth),
                "average_reward": float(np.mean(total_rewards)),
                "average_events_per_episode": float(np.mean(event_counts)),
                "actions_distribution": {
                    "hold": actions_taken.count(0) / len(actions_taken),
                    "buy": actions_taken.count(1) / len(actions_taken),
                    "sell": actions_taken.count(2) / len(actions_taken)
                },
                "epoch_logs": epoch_logs
            }
            
        except Exception as e:
            logger.error(f"Error in adversarial RL training: {e}")
            return {
                "success": False,
                "message": f"Error in adversarial RL training: {str(e)}"
            }

    def adversarial_training_loop(self, X_train, y_train, X_test, y_test, input_size, 
                                 seq_length=20, num_epochs=50, adv_lambda=0.1):
        try:
            logger.info("Cleaning input data...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

            logger.info("Converting data to tensors...")
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device, requires_grad=True)
            y_train_tensor = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, 
                                        dtype=torch.float32, device=self.device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device, requires_grad=True)
            y_test_tensor = torch.tensor(y_test.values if isinstance(y_test, pd.Series) else y_test, 
                                        dtype=torch.float32, device=self.device)
            
            def create_sequences(x_data, y_data, seq_length):
                logger.info(f"Creating sequences with length {seq_length}...")
                xs, ys = [], []
                for i in range(len(x_data) - seq_length):
                    seq = x_data[i:i+seq_length].detach().clone()
                    seq.requires_grad_(True)
                    xs.append(seq)
                    ys.append(y_data[i+seq_length])
                xs_tensor = torch.stack(xs).to(self.device)
                ys_tensor = torch.stack(ys).to(self.device)
                xs_tensor.requires_grad_(True)
                return xs_tensor, ys_tensor
            
            logger.info("Generating training sequences...")
            X_train_seq, y_train_seq = create_sequences(X_train_tensor, y_train_tensor, seq_length)
            logger.info("Generating test sequences...")
            X_test_seq, y_test_seq = create_sequences(X_test_tensor, y_test_tensor, seq_length)
            
            train_dataset = TensorDataset(X_train_seq, y_train_seq)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_dataset = TensorDataset(X_test_seq, y_test_seq)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_size, output_size)
                    
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            class TransformerModel(nn.Module):
                def __init__(self, input_size, seq_length, num_heads=4, dim_feedforward=128, num_layers=2, output_size=1):
                    super(TransformerModel, self).__init__()
                    adjusted_input_size = ((input_size + num_heads - 1) // num_heads) * num_heads
                    self.encoder_layer = nn.TransformerEncoderLayer(
                        d_model=adjusted_input_size,
                        nhead=num_heads,
                        dim_feedforward=dim_feedforward,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
                    self.fc = nn.Linear(adjusted_input_size * seq_length, output_size)
                    self.input_size = input_size
                    self.seq_length = seq_length
                    self.input_proj = nn.Linear(input_size, adjusted_input_size)
                    
                def forward(self, x):
                    x = self.input_proj(x)
                    out = self.transformer_encoder(x)
                    out = out.reshape(out.shape[0], -1)
                    out = self.fc(out)
                    return out
            
            logger.info("Initializing LSTM and Transformer models...")
            lstm_model = LSTMModel(input_size=input_size).to(self.device)
            transformer_model = TransformerModel(input_size=input_size, seq_length=seq_length).to(self.device)
            
            criterion = nn.MSELoss()
            lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
            transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
            
            def fgsm_attack(data, epsilon, data_grad):
                if data_grad is None:
                    logger.warning("data_grad is None, skipping perturbation")
                    return data
                sign_data_grad = data_grad.sign()
                perturbed_data = data + epsilon * sign_data_grad
                perturbed_data = perturbed_data.detach().requires_grad_(True)
                return perturbed_data
            
            lstm_logs = {}
            transformer_logs = {}
            
            logger.info("Starting adversarial training for LSTM and Transformer...")
            for epoch in range(num_epochs):
                lstm_model.train()
                transformer_model.train()
                lstm_running_loss = 0.0
                transformer_running_loss = 0.0
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = inputs.clone().detach().requires_grad_(True)
                    
                    lstm_optimizer.zero_grad()
                    with torch.enable_grad():
                        lstm_outputs = lstm_model(inputs)
                        lstm_loss = criterion(lstm_outputs.squeeze(), labels)
                        
                        lstm_loss.backward(retain_graph=True)
                        data_grad = inputs.grad
                        if data_grad is None:
                            logger.warning(f"LSTM data_grad is None in batch {batch_idx}, skipping adversarial step")
                            perturbed_inputs = inputs.clone().detach().requires_grad_(True)
                        else:
                            perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)
                        
                        lstm_optimizer.zero_grad()
                        lstm_adv_outputs = lstm_model(perturbed_inputs)
                        lstm_adv_loss = criterion(lstm_adv_outputs.squeeze(), labels)
                        
                        lstm_total_loss = lstm_loss + adv_lambda * lstm_adv_loss
                        lstm_total_loss.backward()
                    
                    lstm_optimizer.step()
                    lstm_running_loss += lstm_total_loss.item()
                    
                    transformer_optimizer.zero_grad()
                    inputs = inputs.clone().detach().requires_grad_(True)
                    with torch.enable_grad():
                        transformer_outputs = transformer_model(inputs)
                        transformer_loss = criterion(transformer_outputs.squeeze(), labels)
                        
                        transformer_loss.backward(retain_graph=True)
                        data_grad = inputs.grad
                        if data_grad is None:
                            logger.warning(f"Transformer data_grad is None in batch {batch_idx}, skipping adversarial step")
                            perturbed_inputs = inputs.clone().detach().requires_grad_(True)
                        else:
                            perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)
                        
                        transformer_optimizer.zero_grad()
                        transformer_adv_outputs = transformer_model(perturbed_inputs)
                        transformer_adv_loss = criterion(transformer_adv_outputs.squeeze(), labels)
                        
                        transformer_total_loss = transformer_loss + adv_lambda * transformer_adv_loss
                        transformer_total_loss.backward()
                    
                    transformer_optimizer.step()
                    transformer_running_loss += transformer_total_loss.item()
                
                if (epoch + 1) % 10 == 0:
                    lstm_epoch_loss = lstm_running_loss / len(train_loader)
                    transformer_epoch_loss = transformer_running_loss / len(train_loader)
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                                f'LSTM Loss: {lstm_epoch_loss:.4f}, '
                                f'Transformer Loss: {transformer_epoch_loss:.4f}')
                    lstm_logs[f"Epoch_{epoch+1}"] = lstm_epoch_loss
                    transformer_logs[f"Epoch_{epoch+1}"] = transformer_epoch_loss
            
            lstm_model.eval()
            transformer_model.eval()
            lstm_preds = []
            transformer_preds = []
            
            logger.info("Evaluating models...")
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    lstm_outputs = lstm_model(inputs)
                    lstm_preds.extend(lstm_outputs.squeeze().cpu().tolist())
                    transformer_outputs = transformer_model(inputs)
                    transformer_preds.extend(transformer_outputs.squeeze().cpu().tolist())
            
            lstm_mse = mean_squared_error(y_test_seq.cpu().numpy(), lstm_preds)
            lstm_r2 = r2_score(y_test_seq.cpu().numpy(), lstm_preds)
            transformer_mse = mean_squared_error(y_test_seq.cpu().numpy(), transformer_preds)
            transformer_r2 = r2_score(y_test_seq.cpu().numpy(), transformer_preds)
            
            lstm_adv_preds = []
            transformer_adv_preds = []
            
            logger.info("Evaluating adversarial robustness...")
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device).clone().detach().requires_grad_(True)
                lstm_outputs = lstm_model(inputs)
                lstm_loss = criterion(lstm_outputs.squeeze(), labels.to(self.device))
                lstm_loss.backward(retain_graph=True)
                data_grad = inputs.grad
                perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)
                
                lstm_adv_outputs = lstm_model(perturbed_inputs)
                lstm_adv_preds.extend(lstm_adv_outputs.squeeze().cpu().tolist())
                
                inputs = inputs.clone().detach().requires_grad_(True)
                transformer_outputs = transformer_model(inputs)
                transformer_loss = criterion(transformer_outputs.squeeze(), labels.to(self.device))
                transformer_loss.backward(retain_graph=True)
                data_grad = inputs.grad
                perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)
                
                transformer_adv_outputs = transformer_model(perturbed_inputs)
                transformer_adv_preds.extend(transformer_adv_outputs.squeeze().cpu().tolist())
            
            lstm_adv_mse = mean_squared_error(y_test_seq.cpu().numpy(), lstm_adv_preds)
            lstm_adv_r2 = r2_score(y_test_seq.cpu().numpy(), lstm_adv_preds)
            transformer_adv_mse = mean_squared_error(y_test_seq.cpu().numpy(), transformer_adv_preds)
            transformer_adv_r2 = r2_score(y_test_seq.cpu().numpy(), transformer_adv_preds)
            
            return {
                "success": True,
                "lstm_metrics": {
                    "mse": float(lstm_mse),
                    "r2": float(lstm_r2),
                    "adv_mse": float(lstm_adv_mse),
                    "adv_r2": float(lstm_adv_r2)
                },
                "transformer_metrics": {
                    "mse": float(transformer_mse),
                    "r2": float(transformer_r2),
                    "adv_mse": float(transformer_adv_mse),
                    "adv_r2": float(transformer_adv_r2)
                },
                "lstm_model": lstm_model,
                "transformer_model": transformer_model,
                "lstm_epoch_logs": lstm_logs,
                "transformer_epoch_logs": transformer_logs
            }
            
        except Exception as e:
            logger.error(f"Error in adversarial training loop: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error in adversarial training: {str(e)}",
                "lstm_epoch_logs": {},
                "transformer_epoch_logs": {}
            }

    def analyze_stock(self, ticker, benchmark_tickers=None, prediction_days=30, training_period="7y"):
        try:
            ticker = ticker.strip().upper()
            logger.info(f"Fetching and analyzing data for {ticker}...")

            stock = yf.Ticker(ticker)
            history = stock.history(period="2y")

            if history.empty:
                logger.error(f"No price data found for {ticker}.")
                return {
                    "success": False,
                    "message": f"Unable to fetch data for {ticker}: No price data found"
                }

            stock_info = stock.info
            current_price = float(history["Close"].iloc[-1])
            exchange_rates = self.fetch_exchange_rates()
            converted_prices = self.convert_price(current_price, exchange_rates)
            market_cap = stock_info.get("marketCap", "N/A")
            volume = stock_info.get("volume", "N/A")
            pe_ratio = stock_info.get("trailingPE", "N/A")
            dividends = stock_info.get("dividendYield", "N/A")
            dividend_yield = float(dividends) * 100 if isinstance(dividends, (int, float)) else "N/A"
            high_52w = stock_info.get("fiftyTwoWeekHigh", "N/A")
            low_52w = stock_info.get("fiftyTwoWeekLow", "N/A")
            sector = stock_info.get("sector", "N/A")
            industry = stock_info.get("industry", "N/A")

            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()
            history["EMA_50"] = history["Close"].ewm(span=50, adjust=False).mean()

            def calculate_rsi(data, periods=14):
                delta = data.diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                roll_up = up.ewm(com=periods-1, adjust=False).mean()
                roll_down = down.ewm(com=periods-1, adjust=False).mean()
                rs = roll_up / roll_down
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return rsi

            history["RSI"] = calculate_rsi(history["Close"])

            history["BB_Middle"] = history["Close"].rolling(window=20).mean()
            history["BB_Upper"] = history["BB_Middle"] + 2 * history["Close"].rolling(window=20).std()
            history["BB_Lower"] = history["BB_Middle"] - 2 * history["Close"].rolling(window=20).std()

            exp1 = history["Close"].ewm(span=12, adjust=False).mean()
            exp2 = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = exp1 - exp2
            history["Signal_Line"] = history["MACD"].ewm(span=9, adjust=False).mean()
            history["MACD_Histogram"] = history["MACD"] - history["Signal_Line"]

            history["Daily_Return"] = history["Close"].pct_change()
            history["Volatility"] = history["Daily_Return"].rolling(window=30).std()

            mpt_metrics = self.calculate_mpt_metrics(history, benchmark_tickers or ['^GSPC'])

            risk_free_rate = 0.02
            sharpe_ratio = (history["Daily_Return"].mean() - risk_free_rate) / history["Daily_Return"].std() if history["Daily_Return"].std() != 0 else 0

            sma_50 = float(history["SMA_50"].iloc[-1]) if not pd.isna(history["SMA_50"].iloc[-1]) else current_price
            sma_200 = float(history["SMA_200"].iloc[-1]) if not pd.isna(history["SMA_200"].iloc[-1]) else current_price
            ema_50 = float(history["EMA_50"].iloc[-1]) if not pd.isna(history["EMA_50"].iloc[-1]) else current_price
            volatility = float(history["Volatility"].iloc[-1]) if not pd.isna(history["Volatility"].iloc[-1]) else 0
            rsi = float(history["RSI"].iloc[-1]) if not pd.isna(history["RSI"].iloc[-1]) else 50
            bb_upper = float(history["BB_Upper"].iloc[-1]) if not pd.isna(history["BB_Upper"].iloc[-1]) else current_price * 1.1
            bb_lower = float(history["BB_Lower"].iloc[-1]) if not pd.isna(history["BB_Lower"].iloc[-1]) else current_price * 0.9
            macd = float(history["MACD"].iloc[-1]) if not pd.isna(history["MACD"].iloc[-1]) else 0
            signal_line = float(history["Signal_Line"].iloc[-1]) if not pd.isna(history["Signal_Line"].iloc[-1]) else 0
            macd_histogram = float(history["MACD_Histogram"].iloc[-1]) if not pd.isna(history["MACD_Histogram"].iloc[-1]) else 0

            momentum = 0
            if len(history) >= 30:
                momentum = (current_price - history["Close"].iloc[-30]) / history["Close"].iloc[-30]

            logger.info(f"Fetching sentiment for {ticker}...")
            sentiment_data = self.fetch_combined_sentiment(ticker)

            sentiment = sentiment_data["aggregated"]
            total_sentiment = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]
            sentiment_score = sentiment["positive"] / total_sentiment if total_sentiment > 0 else 0.5

            price_to_sma200 = current_price / sma_200 if sma_200 > 0 else 1
            price_to_sma50 = current_price / sma_50 if sma_50 > 0 else 1
            trend_direction = "UPTREND" if sma_50 > sma_200 else "DOWNTREND"
            volume_trend = "HIGH" if isinstance(volume, (int, float)) and volume > 1000000 else "MODERATE"

            logger.info(f"Fetching institutional investments data for {ticker}...")
            institutional_holders = stock.institutional_holders
            major_holders = stock.major_holders

            institutional_data = {}
            if institutional_holders is not None and not institutional_holders.empty:
                top_institutional = institutional_holders.head(5)
                institutional_data["top_holders"] = []
                for _, row in top_institutional.iterrows():
                    holder_data = {
                        "name": row["Holder"] if "Holder" in row else "Unknown",
                        "shares": row["Shares"] if "Shares" in row else 0,
                        "date_reported": str(row["Date Reported"]) if "Date Reported" in row else "Unknown",
                        "pct_out": round(float(row["% Out"]) * 100, 2) if "% Out" in row else 0,
                        "value": row["Value"] if "Value" in row else 0
                    }
                    institutional_data["top_holders"].append(holder_data)
                institutional_data["total_shares_held"] = institutional_holders["Shares"].sum() if "Shares" in institutional_holders else 0
                institutional_data["total_value"] = institutional_holders["Value"].sum() if "Value" in institutional_holders else 0

            if major_holders is not None and not major_holders.empty:
                try:
                    inst_value = major_holders.iloc[0, 0]
                    if isinstance(inst_value, str) and '%' in inst_value:
                        institutional_data["institutional_ownership_pct"] = float(inst_value.strip('%'))
                    else:
                        institutional_data["institutional_ownership_pct"] = float(inst_value)

                    insider_value = major_holders.iloc[1, 0]
                    if isinstance(insider_value, str) and '%' in insider_value:
                        institutional_data["insider_ownership_pct"] = float(insider_value.strip('%'))
                    else:
                        institutional_data["insider_ownership_pct"] = float(insider_value)
                except (IndexError, ValueError, AttributeError) as e:
                    logger.error(f"Error processing major holders data: {e}")
                    institutional_data["institutional_ownership_pct"] = 0
                    institutional_data["insider_ownership_pct"] = 0

            mutual_fund_holders = stock.mutualfund_holders
            mutual_fund_data = {}
            if mutual_fund_holders is not None and not mutual_fund_holders.empty:
                top_mutual_funds = mutual_fund_holders.head(5)
                mutual_fund_data["top_holders"] = []
                for _, row in top_mutual_funds.iterrows():
                    holder_data = {
                        "name": row["Holder"] if "Holder" in row else "Unknown",
                        "shares": row["Shares"] if "Shares" in row else 0,
                        "date_reported": str(row["Date Reported"]) if "Date Reported" in row else "Unknown",
                        "pct_out": round(float(row["% Out"]) * 100, 2) if "% Out" in row else 0,
                        "value": row["Value"] if "Value" in row else 0
                    }
                    mutual_fund_data["top_holders"].append(holder_data)
                mutual_fund_data["total_shares_held"] = mutual_fund_holders["Shares"].sum() if "Shares" in mutual_fund_holders else 0
                mutual_fund_data["total_value"] = mutual_fund_holders["Value"].sum() if "Value" in mutual_fund_holders else 0

            institutional_confidence = 0
            if institutional_data.get("institutional_ownership_pct", 0) > 70:
                institutional_confidence = 0.3
            elif institutional_data.get("institutional_ownership_pct", 0) > 50:
                institutional_confidence = 0.2
            elif institutional_data.get("institutional_ownership_pct", 0) > 30:
                institutional_confidence = 0.1
            if institutional_data.get("insider_ownership_pct", 0) > 20:
                institutional_confidence += 0.1

            buy_score = (
                (1 - price_to_sma200) * 0.2 +
                (1 if trend_direction == "UPTREND" else -0.5) * 0.2 +
                (sentiment_score - 0.5) * 0.25 +
                (sharpe_ratio * 0.15) +
                (0.5 - volatility) * 0.1 +
                institutional_confidence * 0.1
            )
            sell_score = (
                (price_to_sma200 - 1) * 0.2 +
                (1 if trend_direction == "DOWNTREND" else -0.5) * 0.2 +
                ((1 - sentiment_score) - 0.5) * 0.25 +
                (-sharpe_ratio * 0.15) +
                (volatility - 0.5) * 0.1 +
                (-institutional_confidence * 0.1)
            )

            if buy_score > 0.2:
                recommendation = "STRONG BUY"
            elif buy_score > 0:
                recommendation = "BUY"
            elif sell_score > 0.2:
                recommendation = "STRONG SELL"
            elif sell_score > 0:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"

            support_level = min(sma_200, sma_50) * 0.95
            resistance_level = max(current_price * 1.05, sma_50 * 1.05)

            if volatility > 0.03:
                risk_level = "HIGH"
            elif volatility > 0.015:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            timeframe = "LONG_TERM" if trend_direction == "UPTREND" and sentiment_score > 0.6 else "MEDIUM_TERM" if trend_direction == "UPTREND" else "SHORT_TERM"

            stock_data = {
                "symbol": ticker,
                "name": stock_info.get("shortName", ticker),
                "current_price": converted_prices,
                "sector": sector,
                "industry": industry,
                "pe_ratio": pe_ratio,
                "dividends": dividends
            }

            balance_sheet = self.balance_sheet(ticker)
            income_statement = self.income_statement(ticker)
            cash_flow = self.cash_flow(ticker)

            def filter_non_nan(data):
                return {k: v for k, v in data.items() if v not in ["N/A", "nan", None, float('nan'), "null", ""]}

            balance_sheet_filtered = filter_non_nan(balance_sheet.get("balance_sheet", {}))
            income_statement_filtered = filter_non_nan(income_statement.get("income_statement", {}))
            cash_flow_filtered = filter_non_nan(cash_flow.get("cash_flow", {}))

            explanation = self._generate_detailed_recommendation(
                stock_data, recommendation, buy_score, sell_score,
                price_to_sma200, trend_direction, sentiment_score,
                volatility, sharpe_ratio
            )

            logger.info(f"Fetching extended data for {ticker} for ML analysis...")
            extended_history = stock.history(period=training_period)

            if extended_history.empty:
                logger.error(f"Unable to fetch sufficient extended historical data for {ticker}")
                ml_analysis = {
                    "success": False,
                    "message": f"Unable to fetch sufficient historical data for ML analysis of {ticker}"
                }
            else:
                logger.info(f"Generating adversarial financial data for {ticker}...")
                adv_history = self.generate_adversarial_financial_data(extended_history)

                combined_history = pd.concat([extended_history, adv_history]).reset_index(drop=True)

                logger.info(f"Engineering features for ML pattern recognition...")
                data = combined_history[['Close']].copy()

                data['SMA_5'] = combined_history['Close'].rolling(window=5).mean()
                data['SMA_20'] = combined_history['Close'].rolling(window=20).mean()
                data['SMA_50'] = combined_history['Close'].rolling(window=50).mean()
                data['SMA_200'] = combined_history['Close'].rolling(window=200).mean()

                data['EMA_12'] = combined_history['Close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = combined_history['Close'].ewm(span=26, adjust=False).mean()

                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

                delta = combined_history['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))

                data['BB_Middle'] = data['SMA_20']
                stddev = combined_history['Close'].rolling(window=20).std()
                data['BB_Upper'] = data['BB_Middle'] + 2 * stddev
                data['BB_Lower'] = data['BB_Middle'] - 2 * stddev
                data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']

                data['Volume_Change'] = combined_history['Volume'].pct_change()
                data['Volume_SMA_5'] = combined_history['Volume'].rolling(window=5).mean()
                data['Volume_SMA_20'] = combined_history['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = combined_history['Volume'] / data['Volume_SMA_5']

                data['Price_Change'] = combined_history['Close'].pct_change()
                data['Price_Change_5d'] = combined_history['Close'].pct_change(periods=5)
                data['Price_Change_20d'] = combined_history['Close'].pct_change(periods=20)

                data['Volatility_5d'] = data['Price_Change'].rolling(window=5).std()
                data['Volatility_20d'] = data['Price_Change'].rolling(window=20).std()

                data['Price_to_SMA50'] = combined_history['Close'] / data['SMA_50'] - 1
                data['Price_to_SMA200'] = combined_history['Close'] / data['SMA_200'] - 1

                data['ROC_5'] = (combined_history['Close'] / combined_history['Close'].shift(5) - 1) * 100
                data['ROC_10'] = (combined_history['Close'] / combined_history['Close'].shift(10) - 1) * 100

                obv = pd.Series(index=combined_history.index)
                obv.iloc[0] = 0
                for i in range(1, len(combined_history)):
                    if combined_history['Close'].iloc[i] > combined_history['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + combined_history['Volume'].iloc[i]
                    elif combined_history['Close'].iloc[i] < combined_history['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - combined_history['Volume'].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                data['OBV'] = obv
                data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()

                low_14 = combined_history['Low'].rolling(window=14).min()
                high_14 = combined_history['High'].rolling(window=14).max()
                data['%K'] = (combined_history['Close'] - low_14) / (high_14 - low_14) * 100
                data['%D'] = data['%K'].rolling(window=3).mean()

                tr1 = abs(combined_history['High'] - combined_history['Low'])
                tr2 = abs(combined_history['High'] - combined_history['Close'].shift())
                tr3 = abs(combined_history['Low'] - combined_history['Close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()

                plus_dm = combined_history['High'].diff()
                minus_dm = combined_history['Low'].diff().mul(-1)
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

                smoothed_plus_dm = plus_dm.rolling(window=14).sum()
                smoothed_minus_dm = minus_dm.rolling(window=14).sum()
                smoothed_atr = atr.rolling(window=14).sum()

                plus_di = 100 * smoothed_plus_dm / smoothed_atr
                minus_di = 100 * smoothed_minus_dm / smoothed_atr
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                data['ADX'] = dx.rolling(window=14).mean()
                data['Plus_DI'] = plus_di
                data['Minus_DI'] = minus_di

                # Add sentiment score as a feature
                sentiment_score = sentiment["positive"] / (sentiment["positive"] + sentiment["negative"] + sentiment["neutral"] + 1e-10)
                data['Sentiment_Score'] = sentiment_score

                # Add trend direction as a feature
                data['Trend_Direction'] = (data['SMA_50'] > data['SMA_200']).astype(int)

                # Add momentum as a feature
                data['Momentum_10d'] = (combined_history['Close'] / combined_history['Close'].shift(10) - 1) * 100

                data['Target'] = data['Close'].shift(-prediction_days)

                data = data.replace([np.inf, -np.inf], np.nan).dropna()

                if len(data) < 100:
                    ml_analysis = {
                        "success": False,
                        "message": f"Insufficient data points for {ticker} after feature engineering"
                    }
                else:
                    X = data.drop(['Target', 'Close'], axis=1)
                    y = data['Target']

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

                    logger.info("Running adversarial training for deep learning models...")
                    adv_training_results = self.adversarial_training_loop(
                        X_train, y_train, X_test, y_test,
                        input_size=X_train.shape[1],
                        seq_length=20,
                        num_epochs=50,
                        adv_lambda=0.1
                    )

                    logger.info("Training traditional machine learning models...")
                    models = {
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
                    }

                    ml_predictions = {}
                    ml_metrics = {}
                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            ml_predictions[name] = float(y_pred[-1]) if len(y_pred) > 0 else y_test.iloc[-1]
                            ml_metrics[name] = {
                                "mse": float(mean_squared_error(y_test, y_pred)),
                                "mae": float(mean_absolute_error(y_test, y_pred)),
                                "r2": float(r2_score(y_test, y_pred))
                            }
                        except Exception as e:
                            logger.error(f"Error training {name} model: {e}")
                            ml_predictions[name] = y_test.iloc[-1]
                            ml_metrics[name] = {"mse": "N/A", "mae": "N/A", "r2": "N/A"}

                    logger.info("Creating ensemble model with stacking...")
                    estimators = [
                        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
                        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
                        ("xgb", XGBRegressor(n_estimators=100, random_state=42))
                    ]
                    stacking_model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=LinearRegression()
                    )

                    try:
                        stacking_model.fit(X_train, y_train)
                        y_pred_stack = stacking_model.predict(X_test)
                        ensemble_prediction = float(y_pred_stack[-1]) if len(y_pred_stack) > 0 else y_test.iloc[-1]
                        ensemble_metrics = {
                            "mse": float(mean_squared_error(y_test, y_pred_stack)),
                            "mae": float(mean_absolute_error(y_test, y_pred_stack)),
                            "r2": float(r2_score(y_test, y_pred_stack))
                        }
                    except Exception as e:
                        logger.error(f"Error training stacking model: {e}")
                        ensemble_prediction = y_test.iloc[-1]
                        ensemble_metrics = {"mse": "N/A", "mae": "N/A", "r2": "N/A"}

                    logger.info("Optimizing hyperparameters with Optuna...")
                    def objective(trial):
                        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
                        rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30)
                        model = RandomForestRegressor(
                            n_estimators=rf_n_estimators,
                            max_depth=rf_max_depth,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        return mean_squared_error(y_test, y_pred)

                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=10)
                    best_params = study.best_params
                    optimized_rf = RandomForestRegressor(
                        n_estimators=best_params["rf_n_estimators"],
                        max_depth=best_params["rf_max_depth"],
                        random_state=42
                    )
                    optimized_rf.fit(X_train, y_train)
                    optimized_prediction = float(optimized_rf.predict(X_test)[-1])

                    logger.info("Training RL model with adversarial events...")
                    rl_results = self.train_rl_with_adversarial_events(
                        history=combined_history,
                        ml_predicted_price=ensemble_prediction,
                        current_price=current_price
                    )

                    final_ml_prediction = ensemble_prediction
                    if adv_training_results["success"]:
                        lstm_model = adv_training_results.get("lstm_model")
                        transformer_model = adv_training_results.get("transformer_model")
                        if lstm_model and transformer_model:
                            X_latest = X_scaled[-20:]  # Use last 20 timesteps
                            X_latest_tensor = torch.tensor(X_latest, dtype=torch.float32, device=self.device).unsqueeze(0)
                            lstm_model.eval()
                            transformer_model.eval()
                            with torch.no_grad():
                                lstm_pred = lstm_model(X_latest_tensor).cpu().numpy().flatten()[0]
                                transformer_pred = transformer_model(X_latest_tensor).cpu().numpy().flatten()[0]
                            final_ml_prediction = (ensemble_prediction + lstm_pred + transformer_pred) / 3

                    ml_analysis = {
                        "success": True,
                        "predicted_price": float(final_ml_prediction),
                        "prediction_confidence": float(ensemble_metrics["r2"]) if ensemble_metrics["r2"] != "N/A" else 0.5,
                        "ml_metrics": {
                            **ml_metrics,
                            "Stacking Ensemble": ensemble_metrics
                        },
                        "optimized_rf_prediction": float(optimized_prediction),
                        "rl_results": rl_results,
                        "adversarial_training_results": {
                            "lstm_metrics": adv_training_results.get("lstm_metrics", {}),
                            "transformer_metrics": adv_training_results.get("transformer_metrics", {}),
                            "lstm_epoch_logs": adv_training_results.get("lstm_epoch_logs", {}),
                            "transformer_epoch_logs": adv_training_results.get("transformer_epoch_logs", {})
                        }
                    }

            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "stock_data": {
                    "symbol": ticker,
                    "name": stock_info.get("shortName", ticker),
                    "current_price": converted_prices,
                    "market_cap": market_cap,
                    "volume": volume,
                    "pe_ratio": pe_ratio,
                    "dividend_yield": dividend_yield,
                    "52_week_high": high_52w,
                    "52_week_low": low_52w,
                    "sector": sector,
                    "industry": industry
                },
                "technical_indicators": {
                    "sma_50": float(sma_50),
                    "sma_200": float(sma_200),
                    "ema_50": float(ema_50),
                    "rsi": float(rsi),
                    "bollinger_upper": float(bb_upper),
                    "bollinger_lower": float(bb_lower),
                    "macd": float(macd),
                    "signal_line": float(signal_line),
                    "macd_histogram": float(macd_histogram),
                    "volatility": float(volatility),
                    "momentum": float(momentum)
                },
                "sentiment_analysis": sentiment_data,
                "recommendation": {
                    "action": recommendation,
                    "buy_score": float(buy_score),
                    "sell_score": float(sell_score),
                    "explanation": explanation,
                    "support_level": float(support_level),
                    "resistance_level": float(resistance_level),
                    "risk_level": risk_level,
                    "timeframe": timeframe
                },
                "mpt_metrics": mpt_metrics,
                "institutional_investments": institutional_data,
                "mutual_fund_investments": mutual_fund_data,
                "financial_statements": {
                    "balance_sheet": balance_sheet_filtered,
                    "income_statement": income_statement_filtered,
                    "cash_flow": cash_flow_filtered
                },
                "ml_analysis": ml_analysis
            }

            logger.info(f"Analysis completed for {ticker}")
            return self.convert_np_types(result)

        except Exception as e:
            logger.error(f"Error analyzing stock {ticker}: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error analyzing stock {ticker}: {str(e)}"
            }
            
    def save_analysis_to_files(self, analysis, output_dir="stock_analysis"):
        try:
            if not analysis.get("success", False):
                print(f"Cannot save analysis: {analysis.get('message', 'Unknown error')}")
                return {"success": False, "message": analysis.get('message', 'Unknown error')}

            os.makedirs(output_dir, exist_ok=True)
            ticker = analysis.get("stock_data", {}).get("symbol", "UNKNOWN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_ticker = ticker.replace(".", "_")

            json_filename = os.path.join(output_dir, f"{sanitized_ticker}_analysis_{timestamp}.json")
            csv_filename = os.path.join(output_dir, f"{sanitized_ticker}_summary_{timestamp}.csv")
            log_filename = os.path.join(output_dir, "ml_logs.txt")

            # Save full analysis to JSON
            json_data = self.convert_np_types(analysis)
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            print(f"Saved full analysis to {json_filename}")

            # Prepare CSV summary
            stock_data = analysis.get("stock_data", {})
            ml_analysis = analysis.get("ml_analysis", {})
            current_price = stock_data.get("current_price", {})
            predicted_price = ml_analysis.get("predicted_price", "N/A")
            
            # Handle current_price and predicted_price
            current_price_usd = current_price.get("USD", "N/A") if isinstance(current_price, dict) else current_price
            predicted_price_usd = predicted_price if isinstance(predicted_price, (int, float)) else "N/A"

            # Calculate predicted change percentage if possible
            predicted_change_pct = "N/A"
            if isinstance(current_price_usd, (int, float)) and isinstance(predicted_price_usd, (int, float)) and current_price_usd != 0:
                predicted_change_pct = ((predicted_price_usd - current_price_usd) / current_price_usd) * 100

            csv_data = {
                "Symbol": ticker,
                "Name": stock_data.get("name", "N/A"),
                "Current_Price_USD": current_price_usd,
                "Predicted_Price": predicted_price_usd,
                "Recommendation": analysis.get("recommendation", {}).get("action", "N/A"),
                "Buy_Score": analysis.get("recommendation", {}).get("buy_score", "N/A"),
                "Sell_Score": analysis.get("recommendation", {}).get("sell_score", "N/A"),
                "Risk_Level": analysis.get("recommendation", {}).get("risk_level", "N/A"),
                "Sector": stock_data.get("sector", "N/A"),
                "Industry": stock_data.get("industry", "N/A"),
                "Timestamp": timestamp
            }

            with open(csv_filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
            print(f"Saved summary to {csv_filename}")

            # Log ML, RL, and Stock Analysis Report details
            if ml_analysis.get("success", False):
                log_entry = f"\n[{timestamp}] Analysis for {ticker}\n"
                log_entry += "=" * 50 + "\n"
                log_entry += "Stock Analysis Report\n"
                log_entry += "=" * 50 + "\n"
                recommendation = analysis.get("recommendation", {})
                log_entry += analysis.get("recommendation", {}).get("explanation", "No recommendation explanation available")
                log_entry += "\n"  # Add spacing after the report
                log_entry += f"Prediction Days: 30\n"
                log_entry += f"Current Price (USD): {current_price_usd}\n"
                log_entry += f"Predicted Price (USD): {predicted_price_usd}\n"
                log_entry += f"Predicted Change (%): {predicted_change_pct:.2f}\n" if predicted_change_pct != "N/A" else "Predicted Change (%): N/A\n"
                log_entry += f"Confidence Score: {ml_analysis.get('prediction_confidence', 'N/A')}\n"
                log_entry += f"Pattern: Technical and Sentiment Analysis\n"
                log_entry += "Model Scores:\n"
                for model, scores in ml_analysis.get("ml_metrics", {}).items():
                    log_entry += f"  {model}:\n"
                    for metric, value in scores.items():
                        log_entry += f"    {metric}: {value}\n"
                log_entry += f"Best Model: Stacking Ensemble\n"

                # Log LSTM and Transformer epoch logs
                adv_results = ml_analysis.get("adversarial_training_results", {})
                for model, logs in [
                    ("LSTM", adv_results.get("lstm_epoch_logs", {})),
                    ("Transformer", adv_results.get("transformer_epoch_logs", {}))
                ]:
                    if logs:
                        log_entry += f"{model} Epoch Logs:\n"
                        for epoch, loss in logs.items():
                            log_entry += f"  {epoch}: Loss = {loss:.4f}\n"

                # Log RL results
                rl_results = ml_analysis.get("rl_results", {})
                if rl_results.get("success", False):
                    log_entry += "Reinforcement Learning Results:\n"
                    log_entry += f"  Recommendation: {rl_results.get('recommendation', 'N/A')}\n"
                    log_entry += f"  Performance (%): {rl_results.get('performance_pct', 'N/A')}\n"
                    log_entry += f"  Average Reward: {rl_results.get('average_reward', 'N/A')}\n"
                    log_entry += f"  Average Events/Episode: {rl_results.get('average_events_per_episode', 'N/A')}\n"
                    log_entry += "  Actions Distribution:\n"
                    for action, prob in rl_results.get("actions_distribution", {}).items():
                        log_entry += f"    {action}: {prob:.4f}\n"
                    log_entry += "  RL Epoch Logs:\n"
                    for log in rl_results.get("epoch_logs", []):
                        log_entry += (f"    Episode {log.get('episode', 'N/A')}: "
                                      f"Reward = {log.get('total_reward', 'N/A'):.2f}, "
                                      f"Avg Reward = {log.get('average_reward', 'N/A'):.2f}, "
                                      f"Events = {log.get('events_triggered', 'N/A')}\n")

                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(log_entry)
                print(f"Appended ML, RL, and Stock Analysis Report to {log_filename}")

            return {
                "success": True,
                "json_file": json_filename,
                "csv_file": csv_filename,
                "log_file": log_filename
            }

        except Exception as e:
            print(f"Error saving analysis: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error saving analysis: {str(e)}"
            }
            
class StockTradingBot:
    def __init__(self, config):
        self.config = config
        self.timezone = pytz.timezone("America/New_York")
        self.data_feed = DataFeed(config["tickers"])
        self.portfolio = VirtualPortfolio(config)
        self.executor = PaperExecutor(self.portfolio, config)
        self.tracker = PortfolioTracker(self.portfolio, config)
        self.reporter = PerformanceReport(self.portfolio)
        self.stock_analyzer = Stock(
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.initialize()

    def initialize(self):
        """Initialize the bot and its components."""
        logger.info("Initializing Stock Trading Bot...")
        self.portfolio.initialize_portfolio()

    def run_analysis(self, ticker):
        """Run analysis for a given ticker and return the result."""
        return self.stock_analyzer.analyze_stock(
            ticker,
            benchmark_tickers=self.config.get("benchmark_tickers", ["^GSPC"]),
            prediction_days=self.config.get("prediction_days", 30),
            training_period=self.config.get("period", "3y")
        )
    def is_market_open(self):
        """Check if the NYSE is open."""
        try:
            nyse = mcal.get_calendar("NYSE")
            now = datetime.now(self.timezone)  # Use NYSE timezone
            schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
            if schedule.empty:
                logger.info(f"Market is closed on {now.date()} (no schedule found).")
                return False
            market_open = schedule.iloc[0]["market_open"]
            market_close = schedule.iloc[0]["market_close"]
            is_open = market_open <= now <= market_close
            logger.debug(f"Market check: Now={now}, Open={market_open}, Close={market_close}, Is Open={is_open}")
            return is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def make_trading_decision(self, analysis):
        if not analysis.get("success"):
            logger.warning(f"Skipping trading decision for {analysis.get('stock_data', {}).get('symbol')} due to failed analysis")
            return None

        ticker = analysis["stock_data"]["symbol"]
        recommendation = analysis["recommendation"]["action"]
        current_price = analysis["stock_data"]["current_price"]["USD"]
        ml_analysis = analysis["ml_analysis"]
        technical_indicators = analysis["technical_indicators"]
        sentiment_data = analysis["sentiment_analysis"]
        risk_level = analysis["recommendation"]["risk_level"]
        support_level = analysis["recommendation"]["support_level"]
        resistance_level = analysis["recommendation"]["resistance_level"]
        volatility = technical_indicators["volatility"]
        metrics = self.portfolio.get_metrics()
        available_cash = metrics["cash"]
        total_value = metrics["total_value"]
        history = yf.Ticker(ticker).history(period="2y")

        # Signal weights
        weights = {
            "technical": 0.5,
            "sentiment": 0.25,
            "ml": 0.25,
            "dl": 0.0,
            "rl": 0.0
        }
        scores = {"buy": 0.0, "sell": 0.0}

        # Technical Score
        technical_score = 0.0
        if recommendation in ["BUY", "STRONG BUY"]:
            technical_score += 0.5
            if technical_indicators["rsi"] < 40:
                technical_score += 0.3
            if technical_indicators["macd"] > technical_indicators["signal_line"]:
                technical_score += 0.2
            if current_price < technical_indicators["bollinger_lower"]:
                technical_score += 0.2
            if technical_indicators["sma_50"] > technical_indicators["sma_200"]:
                technical_score += 0.2
            scores["buy"] += technical_score * weights["technical"]
        elif recommendation in ["SELL", "STRONG SELL"]:
            technical_score -= 0.5
            if technical_indicators["rsi"] > 60:
                technical_score -= 0.3
            if technical_indicators["macd"] < technical_indicators["signal_line"]:
                technical_score -= 0.2
            if current_price > technical_indicators["bollinger_upper"]:
                technical_score -= 0.2
            if technical_indicators["sma_50"] < technical_indicators["sma_200"]:
                technical_score -= 0.2
            scores["sell"] += abs(technical_score) * weights["technical"]

        # Sentiment Score
        sentiment_score = 0.0
        aggregated = sentiment_data["aggregated"]
        total_sentiment = aggregated["positive"] + aggregated["negative"] + aggregated["neutral"]
        sentiment_ratio = aggregated["positive"] / total_sentiment if total_sentiment > 0 else 0.5
        sentiment_score = (sentiment_ratio - 0.5) * 2
        if sentiment_score > 0:
            scores["buy"] += sentiment_score * weights["sentiment"]
        else:
            scores["sell"] += abs(sentiment_score) * weights["sentiment"]

        # ML Score
        ml_score = 0.0
        if ml_analysis.get("success"):
            ensemble_r2 = ml_analysis["ml_metrics"]["Stacking Ensemble"]["r2"]
            predicted_price = ml_analysis["predicted_price"]
            price_change_pct = ((predicted_price - current_price) / current_price) if current_price > 0 else 0
            if price_change_pct > 0.05 and ensemble_r2 > 0.3:
                ml_score += min(price_change_pct * 2, 1.0) * ensemble_r2
            elif price_change_pct < -0.05 and ensemble_r2 > 0.3:
                ml_score -= min(abs(price_change_pct) * 2, 1.0) * ensemble_r2
            if ml_score > 0:
                scores["buy"] += ml_score * weights["ml"]
            else:
                scores["sell"] += abs(ml_score) * weights["ml"]

        # RL Score
        if ml_analysis.get("rl_results", {}).get("success"):
            rl_recommendation = ml_analysis["rl_results"]["recommendation"]
            if rl_recommendation == "SELL":
                scores["sell"] += 0.2 * weights["rl"]
            elif rl_recommendation == "BUY":
                scores["buy"] += 0.2 * weights["rl"]

        final_buy_score = scores["buy"]
        final_sell_score = scores["sell"]

        # Confidence Thresholds
        base_confidence_threshold = 0.1
        confidence_threshold = min(base_confidence_threshold + (volatility * 0.5), 0.15)
        if risk_level == "HIGH":
            confidence_threshold += 0.05
        elif risk_level == "LOW":
            confidence_threshold -= 0.05
        sell_confidence_threshold = confidence_threshold * 0.8

        # Calculate unrealized PnL
        current_prices = self.portfolio.get_current_prices()
        current_ticker_price = current_prices.get(ticker, {"price": current_price})["price"]
        unrealized_pnl = (
            (current_ticker_price - self.portfolio.holdings[ticker]["avg_price"])
            * self.portfolio.holdings[ticker]["qty"]
            if ticker in self.portfolio.holdings else 0
        )

        # Signals
        buy_signals = sum([
            technical_score > 0,
            sentiment_score > 0,
            ml_score > 0,
            ml_analysis.get("rl_results", {}).get("recommendation") == "BUY"
        ])
        sell_signals = sum([
            technical_score < 0,
            sentiment_score < 0,
            ml_score < 0,
            current_ticker_price < support_level * 0.98,
            current_ticker_price > resistance_level * 1.02,
            unrealized_pnl < -0.05 * total_value,
            ml_analysis.get("rl_results", {}).get("recommendation") == "SELL"
        ])

        # Portfolio Constraints
        max_exposure_per_stock = total_value * 0.2
        current_stock_exposure = self.portfolio.holdings.get(ticker, {"qty": 0})["qty"] * current_ticker_price
        sector = analysis["stock_data"]["sector"]
        sector_exposure = sum(
            data["qty"] * price["price"]
            for asset, data in self.portfolio.holdings.items()
            for price in [self.portfolio.get_current_prices().get(asset, {"price": 0})]
            if analysis["stock_data"].get("sector") == sector
        )
        max_sector_exposure = total_value * 0.5

        # Calculate ATR for volatility-based sizing
        high_low = history['High'] - history['Low']
        high_close = abs(history['High'] - history['Close'].shift())
        low_close = abs(history['Low'] - history['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        atr = float(atr) if not pd.isna(atr) else volatility * current_ticker_price

        # Kelly Criterion for position sizing
        win_prob = min(final_buy_score / 0.3, 1.0) if final_buy_score > 0 else 0.5
        expected_return = (ml_analysis.get("predicted_price", current_ticker_price) / current_ticker_price - 1) if current_ticker_price > 0 else 0
        loss_prob = 1 - win_prob
        kelly_fraction = (win_prob * (expected_return + 1) - 1) / expected_return if expected_return > 0 else 0.1
        kelly_fraction = max(min(kelly_fraction, 0.25), 0.05)  # Constrain between 5% and 25%

        # Liquidity Check
        avg_daily_volume = history["Volume"].rolling(window=20).mean().iloc[-1]
        max_trade_volume = avg_daily_volume * 0.01  # Max 1% of average daily volume
        max_qty_by_volume = max_trade_volume if not pd.isna(max_trade_volume) else float('inf')

        # Stop-Loss and Take-Profit
        stop_loss = support_level * 0.98 if support_level > 0 else current_ticker_price * 0.95
        take_profit = resistance_level * 1.02 if resistance_level > 0 else current_ticker_price * 1.1

        # Trailing Stop-Loss
        if ticker in self.portfolio.holdings:
            avg_price = self.portfolio.holdings[ticker]["avg_price"]
            trailing_stop_pct = 0.05
            highest_price = max(history["Close"].iloc[-30:]) if not history.empty else current_ticker_price
            trailing_stop = highest_price * (1 - trailing_stop_pct)
            if current_ticker_price < trailing_stop:
                logger.info(f"Trailing Stop-Loss triggered for {ticker}: Price ${current_ticker_price:.2f} < Trailing Stop ${trailing_stop:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": trailing_stop,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "trailing_stop_loss"
                }

        # Stop-Loss and Take-Profit
        if ticker in self.portfolio.holdings:
            if current_ticker_price < stop_loss:
                logger.info(f"Stop-Loss triggered for {ticker}: Price ${current_ticker_price:.2f} < Stop-Loss ${stop_loss:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "stop_loss"
                }
            elif current_ticker_price > take_profit:
                logger.info(f"Take-Profit triggered for {ticker}: Price ${current_ticker_price:.2f} > Take-Profit ${take_profit:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "take_profit"
                }

        # Max Holding Period
        max_holding_days = 90
        if ticker in self.portfolio.holdings:
            first_trade = min(
                (datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S.%f") for t in self.portfolio.trade_log if t["asset"] == ticker and t["action"] == "buy"),
                default=datetime.now()
            )
            holding_days = (datetime.now() - first_trade).days
            if holding_days > max_holding_days and unrealized_pnl < 0:
                logger.info(f"Max holding period exceeded for {ticker}: {holding_days} days, Unrealized PnL: ${unrealized_pnl:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "max_holding_period"
                }

        # Buy Quantity Calculation
        buy_qty = 0
        if final_buy_score > confidence_threshold and buy_signals >= 1:
            if available_cash < 100:  # Minimum trade value
                logger.info(f"Skipping BUY for {ticker}: Insufficient cash (${available_cash:.2f} < $100)")
            else:
                # Base position size based on Kelly Criterion and portfolio percentage
                position_size_pct = kelly_fraction * 0.5  # Scale down Kelly for conservatism
                target_position_value = total_value * position_size_pct

                # Adjust for volatility using ATR
                volatility_factor = max(0.5, min(1.5, 1.0 / (1 + atr / current_ticker_price))) if current_ticker_price > 0 else 1.0
                target_position_value *= volatility_factor

                # Confidence scaling
                confidence_factor = 0.5 + (final_buy_score / 0.3) * 0.5  # Scale between 0.5 and 1.0
                confidence_factor = min(max(confidence_factor, 0.5), 1.0)
                target_position_value *= confidence_factor

                # Apply exposure limits
                target_position_value = max(100, min(target_position_value, max_exposure_per_stock - current_stock_exposure))
                target_position_value = min(target_position_value, max_sector_exposure - sector_exposure)

                # Liquidity constraint
                buy_qty = target_position_value / current_ticker_price if current_ticker_price > 0 else 0
                buy_qty = min(buy_qty, max_qty_by_volume, available_cash / current_ticker_price)
                buy_qty = max(0, int(buy_qty))  # Round down to whole shares

        # Sell Quantity Calculation
        sell_qty = 0
        if ticker in self.portfolio.holdings and final_sell_score > sell_confidence_threshold and sell_signals >= 1:
            holding_qty = self.portfolio.holdings[ticker]["qty"]
            # Scale sell quantity based on proximity to take-profit or stop-loss
            price_to_take_profit = (current_ticker_price - take_profit) / take_profit if take_profit > 0 else 0
            price_to_stop_loss = (stop_loss - current_ticker_price) / stop_loss if stop_loss > 0 else 0
            sell_factor = min(max(final_sell_score * 2, abs(price_to_take_profit), abs(price_to_stop_loss)), 1.0)
            
            if final_sell_score > 0.3 or current_ticker_price < stop_loss or current_ticker_price > take_profit:
                sell_qty = holding_qty
            else:
                # Partial sell based on confidence and risk
                sell_qty = holding_qty * sell_factor
                # Adjust for volatility
                volatility_factor = max(0.5, min(1.5, 1.0 / (1 + atr / current_ticker_price))) if current_ticker_price > 0 else 1.0
                sell_qty *= volatility_factor
            sell_qty = max(0, int(min(sell_qty, holding_qty)))  # Round down to whole shares

        # Validate Exposure Limits
        if buy_qty > 0:
            new_stock_exposure = current_stock_exposure + buy_qty * current_ticker_price
            new_sector_exposure = sector_exposure + buy_qty * current_ticker_price
            if new_stock_exposure > max_exposure_per_stock or new_sector_exposure > max_sector_exposure:
                buy_qty = min(
                    (max_exposure_per_stock - current_stock_exposure) / current_ticker_price,
                    (max_sector_exposure - sector_exposure) / current_ticker_price
                ) if current_ticker_price > 0 else 0
                buy_qty = max(0, int(buy_qty))

        # Backoff Logic
        recent_trades = [t for t in self.portfolio.trade_log if datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S.%f") > datetime.now() - timedelta(hours=24)]
        recent_pnl = sum(
            (t["price"] - self.portfolio.holdings.get(t["asset"], {"avg_price": t["price"]})["avg_price"]) * t["qty"]
            for t in recent_trades if t["action"] == "sell"
        )
        trade_frequency = len(recent_trades)
        backoff = (recent_pnl < -0.01 * total_value or trade_frequency > 10) and not (
            current_ticker_price < stop_loss or unrealized_pnl < -0.05 * total_value
        )

        # Execute Trades
        trade = None
        if (
            buy_qty > 0
            and final_buy_score > confidence_threshold
            and buy_signals >= 1
            and buy_qty * current_ticker_price <= available_cash
            and not backoff
        ):
            logger.info(f"Executing BUY for {ticker}: {buy_qty:.0f} units at ${current_ticker_price:.2f}, "
                        f"Position Value: ${buy_qty * current_ticker_price:.2f} ({(buy_qty * current_ticker_price / total_value * 100):.2f}% of portfolio), "
                        f"Stop-Loss: ${stop_loss:.2f}, Take-Profit: ${take_profit:.2f}, ATR: ${atr:.2f}, Kelly Fraction: {kelly_fraction:.2f}")
            success = self.executor.execute_trade(ticker, "buy", buy_qty, current_ticker_price)
            trade = {
                "action": "buy",
                "ticker": ticker,
                "qty": buy_qty,
                "price": current_ticker_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "success": success,
                "confidence_score": final_buy_score,
                "signals": buy_signals,
                "reason": "signal_based"
            }
        elif (
            sell_qty > 0
            and final_sell_score > sell_confidence_threshold
            and sell_signals >= 1
            and ticker in self.portfolio.holdings
            and not backoff
        ):
            logger.info(f"Executing SELL for {ticker}: {sell_qty:.0f} units at ${current_ticker_price:.2f}, "
                        f"Position Value: ${sell_qty * current_ticker_price:.2f} ({(sell_qty * current_ticker_price / total_value * 100):.2f}% of portfolio), "
                        f"Stop-Loss: ${stop_loss:.2f}, Take-Profit: ${take_profit:.2f}, ATR: ${atr:.2f}")
            success = self.executor.execute_trade(ticker, "sell", sell_qty, current_ticker_price)
            trade = {
                "action": "sell",
                "ticker": ticker,
                "qty": sell_qty,
                "price": current_ticker_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "success": success,
                "confidence_score": final_sell_score,
                "signals": sell_signals,
                "reason": "signal_based"
            }
        else:
            # Hold Logic
            hold_conditions = (
                abs(final_buy_score - final_sell_score) < 0.05
                or (support_level * 0.99 < current_ticker_price < resistance_level * 1.01)
                or (buy_signals == 0 and sell_signals == 0)
            )
            if hold_conditions:
                logger.info(f"HOLD {ticker}: Neutral market conditions, "
                            f"Price ${current_ticker_price:.2f} within Support ${support_level:.2f} "
                            f"and Resistance ${resistance_level:.2f}, "
                            f"Buy Score={final_buy_score:.2f}, Sell Score={final_sell_score:.2f}")
                trade = {
                    "action": "hold",
                    "ticker": ticker,
                    "qty": 0,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": True,
                    "confidence_score": max(final_buy_score, final_sell_score),
                    "signals": max(buy_signals, sell_signals),
                    "reason": "neutral_conditions"
                }

        if trade is None:
            logger.info(f"No trade executed for {ticker}: Buy Score={final_buy_score:.2f}, "
                        f"Sell Score={final_sell_score:.2f}, Buy Signals={buy_signals}, "
                        f"Sell Signals={sell_signals}, Buy Qty={buy_qty:.0f}, Sell Qty={sell_qty:.0f}, "
                        f"Backoff={backoff}, Cash=${available_cash:.2f}, ATR=${atr:.2f}")

        return trade


    def run(self):
        """Main bot loop to run analysis, make trades, and generate reports."""
        logger.info("Starting Stock Trading Bot...")
        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market is closed, waiting...")
                    time_module.sleep(60) 
                    continue
                # Log metrics at the start of the loop
                logger.info("Logging portfolio metrics at start of trading cycle...")
                self.tracker.log_metrics()

                for ticker in self.config["tickers"]:
                    logger.info(f"Processing {ticker}...")
                    analysis = self.run_analysis(ticker)
                    if analysis.get("success"):
                        save_result = self.stock_analyzer.save_analysis_to_files(analysis)
                        if save_result.get("success"):
                            logger.info(f"Saved analysis files: {save_result}")
                        else:
                            logger.warning(f"Failed to save analysis: {save_result.get('message')}")
                        trade = self.make_trading_decision(analysis)
                        if trade and trade["success"]:
                            logger.info(f"Trade executed: {trade}")
                    else:
                        logger.warning(f"Analysis failed for {ticker}: {analysis.get('message')}")

                # Generate daily report
                report = self.reporter.generate_report()
                logger.info(f"Daily Report: {report}")

                # Log metrics at the end of the loop
                logger.info("Logging portfolio metrics at end of trading cycle...")
                self.tracker.log_metrics()

                # Sleep until next trading cycle
                time.sleep(self.config.get("sleep_interval", 60))  
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying

def main():
    """Entry point for the trading bot."""
    from alpaca_trade_api.rest import REST
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api = REST(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        base_url="https://paper-api.alpaca.markets"
    )

    account = api.get_account()
    current_cash = float(account.cash)
    portfolio_value = float(account.portfolio_value)
    last_equity = float(account.last_equity)
    current_pnl = portfolio_value - last_equity

    config = {
        "tickers" : [ "META", "UBER", "FUBO", "XPEV", "AGX", "NFLX"],
        "starting_balance": current_cash,
        "current_portfolio_value": portfolio_value,
        "current_pnl": current_pnl,
        "mode": "paper",
        "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_api_secret": os.getenv("ALPACA_API_SECRET"),
        "base_url": "https://paper-api.alpaca.markets",
        "period": "3y",
        "prediction_days": 30,
        "benchmark_tickers": ["^GSPC"],
        "sleep_interval": 60
    }

    bot = StockTradingBot(config)
    bot.run()
if __name__ == "__main__":
    main() 
    
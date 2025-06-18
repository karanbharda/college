import json
import os
import time
import csv
import random
import traceback
import logging
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import backtrader as bt
import torch
import torch.nn as nn
import gym
from gym import spaces  # Fix: Import spaces from gym
import praw
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import optuna
from optuna.trial import Trial
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv() 

# Symbol mapping for cryptocurrency names to Alpaca-compatible symbols
SYMBOL_MAP = {
    "Bitcoin": "BTCUSD",
    "Ethereum": "ETHUSD",
    "Ripple": "XRPUSD",
    "Cardano": "ADAUSD",
    "Solana": "SOLUSD",
    "Polkadot": "DOTUSD",
    "Dogecoin": "DOGEUSD"
}

# --- Start of paper_executor.py ---
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
            print(f"Invalid action: {action}")
            return False
# --- End of paper_executor.py ---

# --- Start of generate_report.py ---
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
        except Exception as e:
            logger.error(f"Error saving report to {report_file}: {e}")
            return report

        return report
# --- End of generate_report.py ---

# --- Start of data_feed.py ---
class DataFeed:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_live_prices(self):
        """Fetch live prices for specified cryptocurrency tickers using yfinance."""
        data = {}
        for ticker in self.tickers:
            try:
                crypto = yf.Ticker(ticker)
                df = crypto.history(period="1d", interval="1m")
                if not df.empty:
                    latest = df.iloc[-1]
                    data[ticker] = {
                        "price": latest["Close"],
                        "volume": latest["Volume"]
                    }
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }
# --- End of data_feed.py ---

# --- Start of tracker.py ---
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
# --- End of tracker.py ---

# --- Start of virtual_portfolio.py ---
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
            try:
                with open(self.portfolio_file, "w") as f:
                    json.dump({"cash": self.cash, "holdings": self.holdings}, f, indent=4)
            except Exception as e:
                logger.error(f"Error initializing portfolio file: {e}")
        if not os.path.exists(self.trade_log_file):
            try:
                with open(self.trade_log_file, "w") as f:
                    json.dump([], f, indent=4)
            except Exception as e:
                logger.error(f"Error initializing trade log file: {e}")

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
            return False
        try:
            alpaca_symbol = SYMBOL_MAP.get(asset, asset)  # Get Alpaca symbol or use original
            self.api.submit_order(
                symbol=alpaca_symbol,
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
            logger.error(f"Error executing buy order: {e}")
            return False

    def sell(self, asset, qty, price):
        """Execute a sell order in paper trading mode."""
        if asset not in self.holdings or self.holdings[asset]["qty"] < qty:
            return False
        try:
            alpaca_symbol = SYMBOL_MAP.get(asset, asset)  # Get Alpaca symbol or use original
            self.api.submit_order(
                symbol=alpaca_symbol,
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
            logger.error(f"Error executing sell order: {e}")
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
                alpaca_symbol = SYMBOL_MAP.get(asset, asset)  # Use mapped symbol for API call
                bar = self.api.get_bars(alpaca_symbol, timeframe="1Min", limit=1).df
                if not bar.empty:
                    prices[asset] = {"price": bar["close"].iloc[-1], "volume": bar["volume"].iloc[-1]}
            except Exception as e:
                logger.error(f"Error fetching price for {asset}: {e}")
        return prices
# --- End of virtual_portfolio.py ---

# --- Start of crypto.py ---
class Crypto:
    # API Constants loaded from environment variables
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

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    @staticmethod
    def convert_np_types(obj):
        """Convert NumPy int64 and float64 to native Python types"""
        if isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [Crypto.convert_np_types(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: Crypto.convert_np_types(v) for k, v in obj.items()}
        else:
            return obj.item() if hasattr(obj, "item") else obj

    def fetch_exchange_rates(self):
        """Fetch cryptocurrency exchange rates for price conversion"""
        try:
            url = f"{self.COINGECKO_API}/simple/price"
            params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {e}")
            return {"bitcoin": {"usd": 1}, "ethereum": {"usd": 1}}

    def convert_price(self, price, exchange_rates):
        """Convert price to different currencies including crypto"""
        return {
            "USD": round(price, 2),
            "BTC": round(price / exchange_rates.get("bitcoin", {}).get("usd", 1), 8),
            "ETH": round(price / exchange_rates.get("ethereum", {}).get("usd", 1), 8)
        }

    def fetch_combined_crypto_sentiment(self, crypto_name):
        """Fetch sentiment from multiple crypto platforms with unified output structure."""
        def cryptopanic_sentiment(crypto_name, max_pages=20):
            try:
                positive, negative, neutral = 0, 0, 0
                for page in range(1, max_pages + 1):
                    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.CRYPTOPANIC_API_KEY}¤cies={crypto_name}&page={page}"
                    response = requests.get(url)
                    if response.status_code != 200:
                        return {"positive": 0, "negative": 0, "neutral": 0, "error": f"Invalid response: {response.status_code}"}
                    data = response.json()
                    articles = data.get("results", [])
                    if not articles:
                        break
                    for post in articles:
                        votes = post.get("votes", {})
                        positive += votes.get("positive", 0)
                        negative += votes.get("negative", 0)
                        neutral += votes.get("neutral", 0)
                        tags = post.get("tags", [])
                        if "bullish" in tags:
                            positive += 1
                        if "bearish" in tags:
                            negative += 1
                        if "news" in tags:
                            neutral += 1
                return {"positive": positive, "negative": negative, "neutral": neutral}
            except Exception as e:
                return {"error": str(e), "positive": 0, "negative": 0, "neutral": 0}
        
        def google_news_sentiment(crypto_name):
            try:
                class GNews:
                    def get_news(self, query):
                        return [{"title": f"News about {query}", "description": f"Description about {query}"}] * 5
                news = GNews()
                articles = news.get_news(crypto_name)
                positive, negative, neutral = 0, 0, 0
                for article in articles:
                    content = f"{article['title']} {article['description']}".lower()
                    sentiment = self.vader.polarity_scores(content)
                    if sentiment['compound'] > 0.05:
                        positive += 1
                    elif sentiment['compound'] < -0.05:
                        negative += 1
                    else:
                        neutral += 1
                return {"positive": positive, "negative": negative, "neutral": neutral}
            except Exception as e:
                logger.error(f"Error fetching Google News sentiment: {e}")
                return {"error": str(e), "positive": 0, "negative": 0, "neutral": 0}

        def fetch_slugs():
            try:
                url = "https://api.santiment.net/graphql"
                headers = {"Content-Type": "application/json"}
                params = {"apikey": self.SANTIMENT_API_KEY}
                query = """
                {
                    allProjects {
                        slug
                        name
                    }
                }
                """
                response = requests.post(url, json={"query": query}, headers=headers, params=params)
                response.raise_for_status()
                data = response.json().get("data", {}).get("allProjects", [])
                return {proj['name'].lower(): proj['slug'] for proj in data if 'slug' in proj and 'name' in proj}
            except Exception as e:
                logger.error(f"Error fetching Santiment slugs: {e}")
                return {"error": str(e)}

        def santiment_sentiment(crypto_name, days=180):
            try:
                from datetime import datetime, timedelta, UTC
                SANTIMENT_SLUGS = fetch_slugs()
                if not SANTIMENT_SLUGS or "error" in SANTIMENT_SLUGS:
                    return {"error": "Failed to fetch Santiment slugs", "positive": 0, "negative": 0, "neutral": 0}
                slug = SANTIMENT_SLUGS.get(crypto_name.lower())
                if not slug:
                    return {"positive": 0, "negative": 0, "neutral": 0, "error": f"No Santiment slug found for {crypto_name}"}
                today = datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                past_date = (datetime.now(UTC) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
                query = f"""
                {{
                    positive: getMetric(metric: "sentiment_positive_total") {{
                        timeseriesData(
                            slug: "{slug}"
                            from: "{past_date}"
                            to: "{today}"
                            interval: "1d"
                        ) {{
                            datetime
                            value
                        }}
                    }}
                    negative: getMetric(metric: "sentiment_negative_total") {{
                        timeseriesData(
                            slug: "{slug}"
                            from: "{past_date}"
                            to: "{today}"
                            interval: "1d"
                        ) {{
                            datetime
                            value
                        }}
                    }}
                }}
                """
                url = "https://api.santiment.net/graphql"
                headers = {"Content-Type": "application/json"}
                params = {"apikey": self.SANTIMENT_API_KEY}
                response = requests.post(url, json={"query": query}, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                positive_data = data.get("data", {}).get("positive", {}).get("timeseriesData", [])
                negative_data = data.get("data", {}).get("negative", {}).get("timeseriesData", [])
                if not positive_data and not negative_data:
                    return {"positive": 0, "negative": 0, "neutral": 100, "error": "No sentiment data found."}
                positive = sum(entry.get('value', 0) for entry in positive_data)
                negative = sum(entry.get('value', 0) for entry in negative_data)
                total = positive + negative
                if total > 0:
                    positive_pct = int(round((positive / total) * 100))
                    negative_pct = int(round((negative / total) * 100))
                    neutral_pct = int(round(100 - (positive_pct + negative_pct)))
                else:
                    positive_pct, negative_pct, neutral_pct = 0, 0, 100
                return {"positive": positive_pct, "negative": negative_pct, "neutral": neutral_pct, "total_articles": len(positive_data) + len(negative_data)}
            except Exception as e:
                logger.error(f"Error fetching Santiment sentiment: {e}")
                return {"error": str(e), "positive": 0, "negative": 0, "neutral": 0}

        def reddit_sentiment(crypto_name):
            try:
                reddit = praw.Reddit(client_id=self.REDDIT_CLIENT_ID, client_secret=self.REDDIT_CLIENT_SECRET, user_agent=self.REDDIT_USER_AGENT)
                subreddit = reddit.subreddit("CryptoCurrency")
                posts = subreddit.search(crypto_name, limit=100)
                positive, negative, neutral = 0, 0, 0
                for post in posts:
                    sentiment = self.vader.polarity_scores(post.title)
                    if sentiment['compound'] > 0.05:
                        positive += 1
                    elif sentiment['compound'] < -0.05:
                        negative += 1
                    else:
                        neutral += 1
                return {"positive": positive, "negative": negative, "neutral": neutral}
            except Exception as e:
                logger.error(f"Error fetching Reddit sentiment: {e}")
                return {"error": str(e), "positive": 0, "negative": 0, "neutral": 0}

        sources = {
            "cryptopanic": cryptopanic_sentiment(crypto_name),
            "google_news": google_news_sentiment(crypto_name),
            "santiment": santiment_sentiment(crypto_name),
            "reddit": reddit_sentiment(crypto_name)
        }
        aggregated = {"positive": 0, "negative": 0, "neutral": 0}
        for src, sentiment in sources.items():
            for key in aggregated:
                aggregated[key] += sentiment.get(key, 0)
        return {"sources": sources, "aggregated": aggregated}

    def get_santiment_slug(self, coin_id):
        coin_map = {
            "bitcoin": "bitcoin", "ethereum": "ethereum", "ripple": "xrp", "cardano": "cardano",
            "solana": "solana", "polkadot": "polkadot", "dogecoin": "dogecoin"
        }
        return coin_map.get(coin_id.lower(), coin_id)

    def generate_adversarial_financial_data(self, original_data, num_samples=100, perturbation_strength=0.05):
        try:
            if not isinstance(original_data, pd.DataFrame) or 'Close' not in original_data.columns:
                raise ValueError("Input must be a DataFrame with a 'Close' column")
            synthetic_datasets = []
            original_prices = original_data['Close'].values
            n = len(original_prices)
            returns = pd.Series(original_prices).pct_change().dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            for _ in range(num_samples):
                synth_data = original_data.copy()
                noise = np.random.normal(0, std_return * perturbation_strength, n)
                shock_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
                shocks = np.zeros(n)
                shocks[shock_indices] = np.random.choice([-1, 1], size=len(shock_indices)) * std_return * 2
                perturbed_returns = returns + noise + shocks
                perturbed_prices = original_prices[0] * (1 + perturbed_returns).cumprod()
                perturbed_prices = np.maximum(perturbed_prices, 0.01)
                synth_data['Close'] = perturbed_prices
                synth_data['SMA_5'] = synth_data['Close'].rolling(window=5).mean()
                synth_data['SMA_20'] = synth_data['Close'].rolling(window=20).mean()
                synth_data['SMA_50'] = synth_data['Close'].rolling(window=50).mean()
                synth_data['SMA_200'] = synth_data['Close'].rolling(window=200).mean()
                synth_data['EMA_12'] = synth_data['Close'].ewm(span=12, adjust=False).mean()
                synth_data['EMA_26'] = synth_data['Close'].ewm(span=26, adjust=False).mean()
                synth_data['MACD'] = synth_data['EMA_12'] - synth_data['EMA_26']
                synth_data['MACD_Signal'] = synth_data['MACD'].ewm(span=9, adjust=False).mean()
                synthetic_datasets.append(synth_data)
            return synthetic_datasets
        except Exception as e:
            logger.error(f"Error generating adversarial financial data: {e}")
            return []

    def rl_adversarial_training_wrapper(self, env, agent, num_episodes=100, adversarial_events=0.1):
        try:
            total_rewards = []
            event_logs = []
            for episode in range(num_episodes):
                state = env.reset()
                total_reward = 0
                done = False
                episode_events = []
                while not done:
                    if np.random.random() < adversarial_events:
                        current_price = float(env.history["Close"].iloc[env.current_step])
                        event_type = np.random.choice(["flash_crash", "liquidity_squeeze"])
                        if event_type == "flash_crash":
                            env.history["Close"].iloc[env.current_step] *= 0.7
                            episode_events.append({"step": env.current_step, "event": "flash_crash", "price_change": -0.3})
                        elif event_type == "liquidity_squeeze":
                            env.history["Volatility"].iloc[env.current_step] *= 1.5
                            episode_events.append({"step": env.current_step, "event": "liquidity_squeeze", "volatility_change": 1.5})
                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if episode_events and episode_events[-1]["step"] == env.current_step:
                        if episode_events[-1]["event"] == "flash_crash":
                            env.history["Close"].iloc[env.current_step] = current_price
                        elif episode_events[-1]["event"] == "liquidity_squeeze":
                            env.history["Volatility"].iloc[env.current_step] /= 1.5
                total_rewards.append(total_reward)
                agent.decay_epsilon()
                event_logs.append({
                    "episode": episode + 1,
                    "total_reward": total_reward,
                    "adversarial_events": episode_events,
                    "average_reward": np.mean(total_rewards[-10:]) if total_rewards else 0
                })
                if (episode + 1) % 10 == 0:
                    logger.info(f"Adversarial RL Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.4f}, Events: {len(episode_events)}")
            return {"total_rewards": total_rewards, "event_logs": event_logs}
        except Exception as e:
            logger.error(f"Error in adversarial RL training: {e}")
            return {"total_rewards": [], "event_logs": []}

    def adversarial_training_loop(self, model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, epochs=50, adv_lambda=0.1):
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            metrics = {"train_loss": [], "test_loss": [], "adv_loss": []}
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_seq)
                loss = criterion(outputs, y_train_seq)
                X_adv = X_train_seq.clone().detach().requires_grad_(True)
                outputs_adv = model(X_adv)
                loss_adv = criterion(outputs_adv, y_train_seq)
                grad = torch.autograd.grad(loss_adv, X_adv, retain_graph=True)[0]
                perturbation = adv_lambda * grad.sign()
                X_adv = X_adv + perturbation
                X_adv = X_adv.detach()
                outputs_adv = model(X_adv)
                loss_adv = criterion(outputs_adv, y_train_seq)
                total_loss = loss + adv_lambda * loss_adv
                total_loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_seq)
                    test_loss = criterion(test_outputs, y_test_seq)
                metrics["train_loss"].append(loss.item())
                metrics["test_loss"].append(test_loss.item())
                metrics["adv_loss"].append(loss_adv.item())
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Adversarial Training Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Adv Loss: {loss_adv.item():.4f}, Test Loss: {test_loss.item():.4f}")
            return metrics
        except Exception as e:
            logger.error(f"Error in adversarial training loop: {e}")
            return {"train_loss": [], "test_loss": [], "adv_loss": []}

    def analyze_crypto(self, ticker, period="2y", prediction_days=30):
        logger.info(f"Analyzing {ticker}...")
        try:
            ticker = ticker.strip()
            is_name_format = not ("-" in ticker)
            if is_name_format:
                crypto_name = ticker.lower()
                market_url = f"https://api.coingecko.com/api/v3/coins/markets"
                params = {"vs_currency": "usd", "ids": crypto_name, "order": "market_cap_desc", "per_page": 1, "page": 1, "sparkline": False}
                market_response = requests.get(market_url, params=params)
                market_response.raise_for_status()
                market_data = market_response.json()
                if not market_data:
                    return {"success": False, "message": f"Could not find data for cryptocurrency '{crypto_name}'"}
                coin_info = market_data[0]
                coin_id = coin_info["id"]
                symbol = coin_info["symbol"].upper()
                history_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                history_params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
                history_response = requests.get(history_url, params=history_params)
                history_response.raise_for_status()
                history_data = history_response.json()
                if "prices" not in history_data:
                    return {"success": False, "message": f"Historical data not available for cryptocurrency '{crypto_name}'"}
                prices = [entry[1] for entry in history_data["prices"]]
                dates = [datetime.fromtimestamp(entry[0]/1000) for entry in history_data["prices"]]
                history = pd.DataFrame({'Date': dates, 'Close': prices})
                history.set_index('Date', inplace=True)
                if "total_volumes" in history_data:
                    history['Volume'] = [entry[1] for entry in history_data["total_volumes"]]
                current_price = float(coin_info["current_price"])
                market_cap = coin_info["market_cap"]
                volume = coin_info["total_volume"]
                circulating_supply = coin_info.get("circulating_supply", "N/A")
                price_change_24h = coin_info.get("price_change_percentage_24h", "N/A")
                high_24h = coin_info.get("high_24h", "N/A")
                low_24h = coin_info.get("low_24h", "N/A")
                liquidity_score = coin_info.get("liquidity_score", "N/A")
            else:
                crypto = yf.Ticker(ticker)
                history = crypto.history(period=period)
                if history.empty:
                    return {"success": False, "message": f"Unable to fetch data for {ticker}"}
                symbol = ticker.split('-')[0]
                current_price = history['Close'].iloc[-1]
                market_cap = None
                volume = history['Volume'].iloc[-1] if 'Volume' in history.columns else None
                circulating_supply = None
                price_change_24h = ((history['Close'].iloc[-1] / history['Close'].iloc[-2]) - 1) * 100 if len(history) > 1 else "N/A"
                high_24h = history['High'].iloc[-1] if 'High' in history.columns else None
                low_24h = history['Low'].iloc[-1] if 'Low' in history.columns else None
                liquidity_score = None

            logger.info(f"Performing technical analysis for {ticker}...")
            data = history[['Close']].copy()
            data['SMA_5'] = history['Close'].rolling(window=5).mean()
            data['SMA_20'] = history['Close'].rolling(window=20).mean()
            data['SMA_50'] = history['Close'].rolling(window=50).mean()
            data['SMA_200'] = history['Close'].rolling(window=200).mean()
            data['EMA_12'] = history['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = history['Close'].ewm(span=26, adjust=False).mean()
            data['EMA_50'] = history['Close'].ewm(span=50, adjust=False).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            delta = history['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['BB_Middle'] = data['SMA_20']
            stddev = history['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + 2 * stddev
            data['BB_Lower'] = data['BB_Middle'] - 2 * stddev
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            data['Volatility_5d'] = data['Close'].pct_change().rolling(window=5).std()
            data['Volatility_20d'] = data['Close'].pct_change().rolling(window=20).std()
            data['Volatility_30d'] = data['Close'].pct_change().rolling(window=30).std()
            data['Volatility'] = data['Volatility_20d']
            if 'Volume' in history.columns:
                data['Volume_Change'] = history['Volume'].pct_change()
                data['Volume_SMA_5'] = history['Volume'].rolling(window=5).mean()
                data['Volume_SMA_20'] = history['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = history['Volume'] / data['Volume_SMA_5']
            data['Price_Change'] = history['Close'].pct_change()
            data['Price_Change_5d'] = history['Close'].pct_change(periods=5)
            data['Price_Change_20d'] = history['Close'].pct_change(periods=20)
            low_14 = history['Low'].rolling(window=14).min() if 'Low' in history else history['Close'].rolling(window=14).min()
            high_14 = history['High'].rolling(window=14).max() if 'High' in history else history['Close'].rolling(window=14).max()
            data['%K'] = (history['Close'] - low_14) / (high_14 - low_14) * 100
            data['%D'] = data['%K'].rolling(window=3).mean()

            def calculate_support_level(history_data, lookback=30):
                recent_data = history_data.iloc[-lookback:]
                return recent_data['Low'].min() if 'Low' in recent_data else recent_data['Close'].min()

            def calculate_resistance_level(history_data, lookback=30):
                recent_data = history_data.iloc[-lookback:]
                return recent_data['High'].max() if 'High' in recent_data else recent_data['Close'].max()

            support_level = calculate_support_level(history)
            resistance_level = calculate_resistance_level(history)

            logger.info(f"Calculating risk metrics for {ticker}...")
            def calculate_advanced_risk_metrics(prices):
                returns = pd.Series(prices).pct_change().dropna()
                risk_free_rate = 0.02
                trading_days = 365
                annual_return = (1 + returns.mean()) ** trading_days - 1 if not returns.empty else 0
                annual_volatility = returns.std() * (trading_days ** 0.5) if not returns.empty else 0
                excess_returns = returns - (risk_free_rate / trading_days)
                sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (trading_days ** 0.5) if not excess_returns.empty and excess_returns.std() != 0 else 0
                downside_returns = returns[returns < 0]
                downside_volatility = downside_returns.std() * (trading_days ** 0.5) if not downside_returns.empty else 0
                sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min() if not drawdown.empty else 0
                calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
                var_95 = np.percentile(returns, 5) if not returns.empty else 0
                def classify_risk_level(volatility):
                    if volatility < 0.1:
                        return "Low Risk"
                    elif volatility < 0.2:
                        return "Moderate Risk"
                    else:
                        return "High Risk"
                risk_level = classify_risk_level(annual_volatility)
                return {
                    "annual_return": annual_return, "annual_volatility": annual_volatility, "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio, "max_drawdown": max_drawdown, "calmar_ratio": calmar_ratio,
                    "var_95": var_95, "risk_level": risk_level
                }
            risk_metrics = calculate_advanced_risk_metrics(data["Close"])

            logger.info(f"Detecting patterns for {ticker}...")
            patterns = []
            if len(data) > 200:
                if data['SMA_50'].iloc[-2] <= data['SMA_200'].iloc[-2] and data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
                    patterns.append({"name": "Golden Cross", "description": "50-day MA crossed above 200-day MA", "significance": "High", "impact": "Bullish"})
                elif data['SMA_50'].iloc[-2] >= data['SMA_200'].iloc[-2] and data['SMA_50'].iloc[-1] < data['SMA_200'].iloc[-1]:
                    patterns.append({"name": "Death Cross", "description": "50-day MA crossed below 200-day MA", "significance": "High", "impact": "Bearish"})
            if len(data) > 50:
                if data['SMA_20'].iloc[-1] > data['SMA_20'].iloc[-20] and data['SMA_50'].iloc[-1] > data['SMA_50'].iloc[-20]:
                    patterns.append({"name": "Moving Average Uptrend", "description": "Short and medium-term MAs trending upward", "significance": "Medium", "impact": "Bullish"})
                elif data['SMA_20'].iloc[-1] < data['SMA_20'].iloc[-20] and data['SMA_50'].iloc[-1] < data['SMA_50'].iloc[-20]:
                    patterns.append({"name": "Moving Average Downtrend", "description": "Short and medium-term MAs trending downward", "significance": "Medium", "impact": "Bearish"})
            if not pd.isna(data['RSI'].iloc[-1]):
                if data['RSI'].iloc[-1] > 70:
                    patterns.append({"name": "Overbought (RSI)", "description": "RSI above 70", "significance": "Medium", "impact": "Bearish"})
                elif data['RSI'].iloc[-1] < 30:
                    patterns.append({"name": "Oversold (RSI)", "description": "RSI below 30", "significance": "Medium", "impact": "Bullish"})
            if not pd.isna(data['MACD'].iloc[-1]):
                if data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2] and data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
                    patterns.append({"name": "MACD Bullish Crossover", "description": "MACD crossed above signal line", "significance": "Medium", "impact": "Bullish"})
                elif data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2] and data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1]:
                    patterns.append({"name": "MACD Bearish Crossover", "description": "MACD crossed below signal line", "significance": "Medium", "impact": "Bearish"})
            if not pd.isna(data['BB_Width'].iloc[-1]):
                recent_bb_width = data['BB_Width'].iloc[-20:].mean()
                if data['BB_Width'].iloc[-1] < recent_bb_width * 0.7:
                    patterns.append({"name": "Bollinger Band Squeeze", "description": "Narrowing BBs indicate low volatility", "significance": "High", "impact": "Neutral"})
                if history['Close'].iloc[-1] > data['BB_Upper'].iloc[-1]:
                    patterns.append({"name": "Upper BB Breakout", "description": "Price broke above upper BB", "significance": "High", "impact": "Bullish"})
                elif history['Close'].iloc[-1] < data['BB_Lower'].iloc[-1]:
                    patterns.append({"name": "Lower BB Breakout", "description": "Price broke below lower BB", "significance": "High", "impact": "Bearish"})
            if 'Volume' in history.columns:
                price_trend_up = history['Close'].iloc[-5:].mean() > history['Close'].iloc[-10:-5].mean()
                volume_trend_up = history['Volume'].iloc[-5:].mean() > history['Volume'].iloc[-10:-5].mean()
                if price_trend_up and not volume_trend_up:
                    patterns.append({"name": "Bearish Volume Divergence", "description": "Price rising, volume decreasing", "significance": "Medium", "impact": "Bearish"})
                elif not price_trend_up and volume_trend_up:
                    patterns.append({"name": "Bullish Volume Divergence", "description": "Price falling, volume increasing", "significance": "Medium", "impact": "Bullish"})

            logger.info(f"Training machine learning models for {ticker}...")
            ml_analysis = None
            try:
                data['Target'] = data['Close'].shift(-prediction_days)
                ml_data = data.dropna()
                if len(ml_data) < 100:
                    logger.warning(f"Insufficient data points for {ticker} after feature engineering")
                else:
                    features_to_exclude = ['Target', 'Close']
                    X = ml_data.drop(features_to_exclude, axis=1)
                    y = ml_data['Target']
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    X_scaled = scaler.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
                    models = {
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                        "Linear Regression": LinearRegression()
                    }
                    model_results = {}
                    predictions = {}
                    feature_importance = {}
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        predictions[name] = y_pred
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        model_results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
                        if hasattr(model, 'feature_importances_'):
                            importance = model.feature_importances_
                            feature_importance[name] = {X.columns[i]: importance[i] for i in range(len(X.columns))}
                    ensemble_pred = np.mean([predictions[name] for name in models.keys()], axis=0)
                    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
                    ensemble_rmse = np.sqrt(ensemble_mse)
                    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    model_results["Ensemble"] = {"MSE": ensemble_mse, "RMSE": ensemble_rmse, "MAE": ensemble_mae, "R2": ensemble_r2}
                    latest_data = X.iloc[-1].values.reshape(1, -1)
                    latest_data_scaled = scaler.transform(latest_data)
                    future_predictions = {name: model.predict(latest_data_scaled)[0] for name, model in models.items()}
                    future_predictions["Ensemble"] = np.mean(list(future_predictions.values()))

                    class LSTMModel(nn.Module):
                        def __init__(self, input_size, hidden_size, num_layers):
                            super(LSTMModel, self).__init__()
                            self.hidden_size = hidden_size
                            self.num_layers = num_layers
                            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                            self.fc = nn.Linear(hidden_size, 1)
                        def forward(self, x):
                            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                            out, _ = self.lstm(x, (h0, c0))
                            out = self.fc(out[:, -1, :])
                            return out

                    class TransformerModel(nn.Module):
                        def __init__(self, input_size, d_model, n_heads, num_layers):
                            super(TransformerModel, self).__init__()
                            self.embedding = nn.Linear(input_size, d_model)
                            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True), num_layers)
                            self.fc = nn.Linear(d_model, 1)
                        def forward(self, x):
                            x = self.embedding(x)
                            x = self.transformer(x)
                            x = self.fc(x[:, -1, :])
                            return x

                    sequence_length = 10
                    if len(X_scaled) < sequence_length + prediction_days:
                        raise ValueError("Not enough data for sequence modeling with given prediction_days")
                    X_sequences = [X_scaled[i:i + sequence_length] for i in range(len(X_scaled) - sequence_length - prediction_days + 1)]
                    y_sequences_short = [y.iloc[i + sequence_length + 7 - 1] for i in range(len(X_scaled) - sequence_length - prediction_days + 1)]
                    y_sequences_long = [y.iloc[i + sequence_length + prediction_days - 1] for i in range(len(X_scaled) - sequence_length - prediction_days + 1)]
                    X_sequences = np.array(X_sequences)
                    y_sequences_short = np.array(y_sequences_short)
                    y_sequences_long = np.array(y_sequences_long)
                    train_size = int(len(X_sequences) * 0.8)
                    test_size = len(y_test)
                    train_size = len(X_sequences) - test_size
                    X_train_seq = torch.FloatTensor(X_sequences[:train_size])
                    X_test_seq = torch.FloatTensor(X_sequences[train_size:train_size + test_size])
                    y_train_seq_short = torch.FloatTensor(y_sequences_short[:train_size]).reshape(-1, 1)
                    y_test_seq_short = torch.FloatTensor(y_sequences_short[train_size:train_size + test_size]).reshape(-1, 1)
                    y_train_seq_long = torch.FloatTensor(y_sequences_long[:train_size]).reshape(-1, 1)
                    y_test_seq_long = torch.FloatTensor(y_sequences_long[train_size:train_size + test_size]).reshape(-1, 1)
                    lstm_model = LSTMModel(input_size=X.shape[1], hidden_size=64, num_layers=2)
                    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                    criterion = nn.MSELoss()
                    for _ in range(50):
                        lstm_model.train()
                        optimizer.zero_grad()
                        outputs = lstm_model(X_train_seq)
                        loss = criterion(outputs, y_train_seq_short)
                        loss.backward()
                        optimizer.step()
                    lstm_model.eval()
                    with torch.no_grad():
                        lstm_pred = lstm_model(X_test_seq).numpy().flatten()
                        lstm_mse = mean_squared_error(y_test_seq_short.numpy(), lstm_pred)
                        lstm_rmse = np.sqrt(lstm_mse)
                        lstm_mae = mean_absolute_error(y_test_seq_short.numpy(), lstm_pred)
                        lstm_r2 = r2_score(y_test_seq_short.numpy(), lstm_pred)
                    model_results["LSTM"] = {"MSE": lstm_mse, "RMSE": lstm_rmse, "MAE": lstm_mae, "R2": lstm_r2}
                    latest_sequence = torch.FloatTensor(X_scaled[-sequence_length:]).unsqueeze(0)
                    with torch.no_grad():
                        lstm_future_pred = lstm_model(latest_sequence).item()
                    future_predictions["LSTM"] = lstm_future_pred
                    transformer_model = TransformerModel(input_size=X.shape[1], d_model=64, n_heads=4, num_layers=2)
                    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
                    criterion = nn.MSELoss()
                    for _ in range(50):
                        transformer_model.train()
                        optimizer.zero_grad()
                        outputs = transformer_model(X_train_seq)
                        loss = criterion(outputs, y_train_seq_long)
                        loss.backward()
                        optimizer.step()
                    transformer_model.eval()
                    with torch.no_grad():
                        transformer_pred = transformer_model(X_test_seq).numpy().flatten()
                        transformer_mse = mean_squared_error(y_test_seq_long.numpy(), transformer_pred)
                        transformer_rmse = np.sqrt(transformer_mse)
                        transformer_mae = mean_absolute_error(y_test_seq_long.numpy(), transformer_pred)
                        transformer_r2 = r2_score(y_test_seq_long.numpy(), transformer_pred)
                    model_results["Transformer"] = {"MSE": transformer_mse, "RMSE": transformer_rmse, "MAE": transformer_mae, "R2": transformer_r2}
                    with torch.no_grad():
                        transformer_future_pred = transformer_model(latest_sequence).item()
                    future_predictions["Transformer"] = transformer_future_pred
                    all_predictions = [predictions["Random Forest"][:len(lstm_pred)], predictions["Gradient Boosting"][:len(lstm_pred)],
                                       predictions["Linear Regression"][:len(lstm_pred)], lstm_pred, transformer_pred]
                    ensemble_pred_with_dl = np.mean(all_predictions, axis=0)
                    ensemble_mse_dl = mean_squared_error(y_test[:len(ensemble_pred_with_dl)], ensemble_pred_with_dl)
                    ensemble_rmse_dl = np.sqrt(ensemble_mse_dl)
                    ensemble_mae_dl = mean_absolute_error(y_test[:len(ensemble_pred_with_dl)], ensemble_pred_with_dl)
                    ensemble_r2_dl = r2_score(y_test[:len(ensemble_pred_with_dl)], ensemble_pred_with_dl)
                    model_results["Enhanced Ensemble"] = {"MSE": ensemble_mse_dl, "RMSE": ensemble_rmse_dl, "MAE": ensemble_mae_dl, "R2": ensemble_r2_dl}
                    future_predictions["Enhanced Ensemble"] = np.mean(list(future_predictions.values()))
                    price_volatility = data['Volatility_20d'].iloc[-1] if not pd.isna(data['Volatility_20d'].iloc[-1]) else 0
                    confidence_score = max((1 - price_volatility) * max(ensemble_r2_dl, 0), 0.1)
                    predicted_price = future_predictions["Enhanced Ensemble"]
                    predicted_change = (predicted_price - current_price) / current_price if current_price != 0 else 0
                    trend_prediction = "UPTREND" if predicted_change > 0 else "DOWNTREND"
                    predicted_change_pct = predicted_change * 100
                    ml_analysis = {
                        "success": True, "model_performance": model_results, "prediction": {
                            "days_ahead": prediction_days, "predicted_price": predicted_price, "current_price": current_price,
                            "predicted_change_pct": predicted_change_pct, "trend_prediction": trend_prediction, "confidence_score": confidence_score
                        }, "feature_importance": feature_importance
                    }
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
                ml_analysis = {"success": False, "message": f"ML prediction failed: {str(e)}"}

            logger.info(f"Starting RL optimization for {ticker}...")
            class StockTradingEnv(gym.Env):
                def __init__(self, history, current_price, ml_predicted_price):
                    super(StockTradingEnv, self).__init__()
                    self.history = history
                    self.current_price = current_price
                    self.ml_predicted_price = ml_predicted_price
                    self.max_steps = len(history) - 1
                    self.current_step = 0
                    self.initial_balance = 1000000
                    self.balance = self.initial_balance
                    self.shares_held = 0
                    self.net_worth = self.initial_balance
                    self.max_shares = 1000
                    self.action_space = spaces.Discrete(3)
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

                def reset(self):
                    self.current_step = 0
                    self.balance = self.initial_balance
                    self.shares_held = 0
                    self.net_worth = self.initial_balance
                    return self._get_observation()

                def _get_observation(self):
                    price = float(self.history["Close"].iloc[self.current_step])
                    sma_50 = float(self.history["SMA_50"].iloc[self.current_step] if pd.notna(self.history["SMA_50"].iloc[self.current_step]) else 0)
                    sma_200 = float(self.history["SMA_200"].iloc[self.current_step] if pd.notna(self.history["SMA_200"].iloc[self.current_step]) else 0)
                    rsi = float(self.history["RSI"].iloc[self.current_step] if pd.notna(self.history["RSI"].iloc[self.current_step]) else 50)
                    macd = float(self.history["MACD"].iloc[self.current_step] if pd.notna(self.history["MACD"].iloc[self.current_step]) else 0)
                    volatility = float(self.history["Volatility"].iloc[self.current_step] if pd.notna(self.history["Volatility"].iloc[self.current_step]) else 0)
                    ml_pred = self.ml_predicted_price if self.current_step == self.max_steps else price
                    return np.array([price, sma_50, sma_200, rsi, macd, volatility, self.balance, self.shares_held, self.net_worth, ml_pred], dtype=np.float32)

                def step(self, action):
                    current_price = float(self.history["Close"].iloc[self.current_step])
                    reward = 0
                    if action == 1:
                        shares_to_buy = min(self.max_shares - self.shares_held, self.balance / current_price)
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
                    reward = (self.net_worth - self.initial_balance) / self.initial_balance
                    if action == 0:
                        reward -= 0.01
                    self.current_step += 1
                    done = self.current_step >= self.max_steps
                    if done:
                        reward += (self.ml_predicted_price - current_price) * self.shares_held / self.initial_balance
                    return self._get_observation(), reward, done, {}

            class QLearningAgent:
                def __init__(self, state_size, action_size):
                    self.state_size = state_size
                    self.action_size = action_size
                    self.q_table = {}
                    self.alpha = 0.1
                    self.gamma = 0.95
                    self.epsilon = 0.5
                    self.epsilon_min = 0.01
                    self.epsilon_decay = 0.995

                def discretize_state(self, state):
                    return tuple(np.round(state, 4))

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

                def decay_epsilon(self):
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            env = StockTradingEnv(history=data, current_price=current_price,
                                  ml_predicted_price=ml_analysis["prediction"]["predicted_price"] if ml_analysis and ml_analysis["success"] else current_price)
            agent = QLearningAgent(state_size=10, action_size=3)
            num_episodes = 100
            total_rewards = []
            RL_epoch_logs = []
            logger.info(f"Training RL agent for {ticker} ...")
            for episode in range(num_episodes):
                state = env.reset()
                total_reward = 0
                done = False
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                total_rewards.append(total_reward)
                agent.decay_epsilon()
                if (episode + 1) % 10 == 0:
                    logger.info(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
                    RL_epoch_logs.append({"episode": episode + 1, "total_reward": total_reward, "average_reward": np.mean(total_rewards[-10:])})

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
            performance = (final_net_worth - env.initial_balance) / env.initial_balance * 100 if env.initial_balance != 0 else 0
            rl_recommendation = "BUY" if performance > 10 else "SELL" if performance < -5 else "HOLD"
            rl_optimization = {
                "success": True, "recommendation": rl_recommendation, "performance_pct": float(performance),
                "final_net_worth": float(final_net_worth), "average_reward": float(np.mean(total_rewards)),
                "actions_distribution": {
                    "hold": actions_taken.count(0) / len(actions_taken) if actions_taken else 0,
                    "buy": actions_taken.count(1) / len(actions_taken) if actions_taken else 0,
                    "sell": actions_taken.count(2) / len(actions_taken) if actions_taken else 0
                }, "RL_epoch_logs": RL_epoch_logs
            }

            logger.info(f"Running backtrader simulations for {symbol}...")
            sim_history = history.copy()
            if sim_history.empty or len(sim_history) < 200:
                logger.warning(f"Unable to fetch sufficient data for backtrader simulation of {symbol}")
                backtest_results = {"success": False, "message": "Insufficient data for backtest"}
                optimization_results = {"success": False, "message": "Insufficient data for optimization"}
            else:
                # Clean sim_history to remove NaN values
                sim_history = sim_history[['Close']].dropna()
                if 'Open' not in sim_history:
                    sim_history['Open'] = sim_history['Close']
                if 'High' not in sim_history:
                    sim_history['High'] = sim_history['Close']
                if 'Low' not in sim_history:
                    sim_history['Low'] = sim_history['Close']
                if 'Volume' not in sim_history:
                    sim_history['Volume'] = 0
                # Ensure numeric data
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    sim_history[col] = pd.to_numeric(sim_history[col], errors='coerce')
                sim_history = sim_history.dropna()

                if len(sim_history) < 200:
                    logger.warning(f"Insufficient valid data points after cleaning for backtrader simulation of {symbol}")
                    backtest_results = {"success": False, "message": "Insufficient valid data after cleaning"}
                    optimization_results = {"success": False, "message": "Insufficient valid data after cleaning"}
                else:
                    class CryptoStrategy(bt.Strategy):
                        params = (
                            ('sma_short', 50),
                            ('sma_long', 200),
                            ('rsi_low', 30),
                            ('rsi_high', 70),
                            ('size', 0.1),
                            ('ml_predicted_price', None),
                            ('commission', 0.001),
                        )

                        def __init__(self):
                            self.sma_short = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.sma_short)
                            self.sma_long = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.sma_long)
                            self.rsi = bt.indicators.RSI(self.datas[0], period=14)
                            self.order = None
                            self.tradehistory = []
                            self.ml_predicted_price = self.params.ml_predicted_price

                        def notify_order(self, order):
                            if order.status in [order.Completed]:
                                if order.isbuy():
                                    logger.info(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}, Cost: {order.executed.value:.2f}")
                                elif order.issell():
                                    logger.info(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}, Value: {order.executed.value:.2f}")
                                self.order = None

                        def notify_trade(self, trade):
                            if trade.isclosed:
                                trade_datetime = trade.data.datetime.datetime()
                                self.tradehistory.append({
                                    "date": trade_datetime,
                                    "pnl": trade.pnl,
                                    "pnlcomm": trade.pnlcomm,
                                    "price": trade.price,
                                    "size": trade.size,
                                    "value": trade.value
                                })
                                logger.info(f"TRADE CLOSED, PNL: {trade.pnl:.2f}, Net PNL: {trade.pnlcomm:.2f}")

                        def next(self):
                            if self.order:
                                return
                            current_price = self.datas[0].close[0]
                            position_size = self.params.size * self.broker.getcash() / current_price
                            bullish_ml_signal = self.ml_predicted_price > current_price * 1.05 if self.ml_predicted_price else False
                            bearish_ml_signal = self.ml_predicted_price < current_price * 0.95 if self.ml_predicted_price else False
                            if not self.position:
                                if (self.sma_short[0] > self.sma_long[0] and self.rsi[0] < self.params.rsi_low) or bullish_ml_signal:
                                    self.order = self.buy(size=position_size)
                                    logger.info(f"BUY ORDER PLACED at {current_price:.2f}, Size: {position_size:.4f}")
                            else:
                                if (self.sma_short[0] < self.sma_long[0] and self.rsi[0] > self.params.rsi_high) or bearish_ml_signal:
                                    self.order = self.sell(size=self.position.size)
                                    logger.info(f"SELL ORDER PLACED at {current_price:.2f}, Size: {self.position.size:.4f}")

                    # Run initial backtest
                    ml_pred_price = ml_analysis["prediction"]["predicted_price"] if ml_analysis and ml_analysis["success"] else current_price
                    if not isinstance(ml_pred_price, (int, float)) or ml_pred_price <= 0:
                        logger.warning(f"Invalid ML predicted price for {symbol}: {ml_pred_price}. Using current price.")
                        ml_pred_price = current_price

                    cerebro = bt.Cerebro()
                    cerebro.addstrategy(CryptoStrategy, ml_predicted_price=ml_pred_price)
                    data_feed = bt.feeds.PandasData(dataname=sim_history)
                    cerebro.adddata(data_feed)
                    cerebro.broker.setcash(1000000)
                    cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
                    cerebro.broker.setcommission(commission=0.001)
                    logger.info("Starting Backtrader simulation...")
                    cerebro.run()
                    final_value = cerebro.broker.getvalue()
                    initial_value = 1000000
                    total_return = ((final_value / initial_value) - 1) * 100
                    strategy = cerebro.runstrats[0][0]
                    trades = strategy.tradehistory
                    number_of_trades = len(trades)
                    wins = sum(1 for trade in trades if trade["pnl"] > 0)
                    win_rate = (wins / number_of_trades * 100) if number_of_trades > 0 else 0
                    total_pnl = sum(trade["pnl"] for trade in trades)
                    backtest_results = {
                        "success": True,
                        "initial_value": initial_value,
                        "final_value": final_value,
                        "total_return_pct": total_return,
                        "number_of_trades": number_of_trades,
                        "win_rate_pct": win_rate,
                        "total_pnl": total_pnl
                    }

                    # Run optimization
                    logger.info(f"Optimizing strategy parameters for {symbol}...")
                    def objective(trial: Trial):
                        params = {
                            'sma_short': trial.suggest_int('sma_short', 10, min(100, len(sim_history) - 1)),
                            'sma_long': trial.suggest_int('sma_long', 50, min(200, len(sim_history) - 1)),
                            'rsi_low': trial.suggest_float('rsi_low', 20, 40),
                            'rsi_high': trial.suggest_float('rsi_high', 60, 80),
                            'size': trial.suggest_float('size', 0.05, 0.5)
                        }
                        try:
                            cerebro_opt = bt.Cerebro()
                            cerebro_opt.addstrategy(CryptoStrategy, **params, ml_predicted_price=ml_pred_price)
                            cerebro_opt.adddata(bt.feeds.PandasData(dataname=sim_history))
                            cerebro_opt.broker.setcash(1000000)
                            cerebro_opt.addsizer(bt.sizers.FixedSize, stake=1000)
                            cerebro_opt.broker.setcommission(commission=0.001)
                            cerebro_opt.run()
                            final_value = cerebro_opt.broker.getvalue()
                            if not isinstance(final_value, (int, float)) or np.isnan(final_value):
                                logger.error(f"Trial {trial.number} returned invalid final value: {final_value}")
                                raise optuna.TrialPruned()
                            return final_value
                        except Exception as e:
                            logger.error(f"Error in optimization trial {trial.number}: {str(e)}")
                            raise optuna.TrialPruned()

                    try:
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=50)
                        if not study.trials:
                            logger.error("No optimization trials completed successfully")
                            optimization_results = {"success": False, "message": "No trials completed successfully"}
                        else:
                            best_params = study.best_params
                            cerebro_opt = bt.Cerebro()
                            cerebro_opt.addstrategy(CryptoStrategy, **best_params, ml_predicted_price=ml_pred_price)
                            cerebro_opt.adddata(bt.feeds.PandasData(dataname=sim_history))
                            cerebro_opt.broker.setcash(1000000)
                            cerebro_opt.addsizer(bt.sizers.FixedSize, stake=1000)
                            cerebro_opt.broker.setcommission(commission=0.001)
                            cerebro_opt.run()
                            optimized_value = cerebro_opt.broker.getvalue()
                            optimized_return = ((optimized_value / initial_value) - 1) * 100
                            optimization_results = {
                                "success": True,
                                "best_parameters": best_params,
                                "optimized_value": optimized_value,
                                "optimized_return_pct": optimized_return
                            }
                    except Exception as e:
                        logger.error(f"Error in strategy optimization: {e}")
                        optimization_results = {"success": False, "message": f"Optimization failed: {str(e)}"}

            logger.info(f"Fetching sentiment analysis for {ticker}...")
            sentiment = self.fetch_combined_crypto_sentiment(symbol.lower() if is_name_format else ticker.split('-')[0].lower())

            exchange_rates = self.fetch_exchange_rates()
            converted_prices = self.convert_price(current_price, exchange_rates)

            recommendation = "NEUTRAL"
            confidence = 0.5
            if ml_analysis and ml_analysis["success"]:
                confidence = ml_analysis["prediction"]["confidence_score"]
                if ml_analysis["prediction"]["trend_prediction"] == "UPTREND" and confidence > 0.6:
                    recommendation = "BUY"
                elif ml_analysis["prediction"]["trend_prediction"] == "DOWNTREND" and confidence > 0.6:
                    recommendation = "SELL"
            if rl_optimization["success"]:
                if (rl_optimization["recommendation"] == "BUY" and recommendation != "SELL"):
                    recommendation = "BUY"
                    confidence = max(confidence, 0.7)
                elif (rl_optimization["recommendation"] == "SELL" and recommendation != "BUY"):
                    recommendation = "SELL"
                    confidence = max(confidence, 0.7)

            analysis_result = {
                "success": True,
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_price": current_price,
                "converted_prices": converted_prices,
                "market_cap": market_cap,
                "volume": volume,
                "circulating_supply": circulating_supply,
                "price_change_24h_pct": price_change_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "liquidity_score": liquidity_score,
                "support_level": support_level,
                "resistance_level": resistance_level,
                "technical_indicators": {
                    "SMA_5": float(data['SMA_5'].iloc[-1]) if not pd.isna(data['SMA_5'].iloc[-1]) else None,
                    "SMA_20": float(data['SMA_20'].iloc[-1]) if not pd.isna(data['SMA_20'].iloc[-1]) else None,
                    "SMA_50": float(data['SMA_50'].iloc[-1]) if not pd.isna(data['SMA_50'].iloc[-1]) else None,
                    "SMA_200": float(data['SMA_200'].iloc[-1]) if not pd.isna(data['SMA_200'].iloc[-1]) else None,
                    "RSI": float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else None,
                    "MACD": float(data['MACD'].iloc[-1]) if not pd.isna(data['MACD'].iloc[-1]) else None,
                    "MACD_Signal": float(data['MACD_Signal'].iloc[-1]) if not pd.isna(data['MACD_Signal'].iloc[-1]) else None,
                    "BB_Upper": float(data['BB_Upper'].iloc[-1]) if not pd.isna(data['BB_Upper'].iloc[-1]) else None,
                    "BB_Lower": float(data['BB_Lower'].iloc[-1]) if not pd.isna(data['BB_Lower'].iloc[-1]) else None,
                    "BB_Width": float(data['BB_Width'].iloc[-1]) if not pd.isna(data['BB_Width'].iloc[-1]) else None,
                    "Volatility_20d": float(data['Volatility_20d'].iloc[-1]) if not pd.isna(data['Volatility_20d'].iloc[-1]) else None
                },
                "patterns_detected": patterns,
                "risk_metrics": risk_metrics,
                "ml_analysis": ml_analysis,
                "rl_optimization": rl_optimization,
                "backtest_results": backtest_results,
                "optimization_results": optimization_results,
                "sentiment_analysis": sentiment,
                "recommendation": recommendation,
                "confidence_score": confidence
            }
            return self.convert_np_types(analysis_result)
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {"success": False, "message": f"Analysis failed: {str(e)}"}
        
    def save_analysis_to_files(self, analysis, output_dir="crypto_analysis"):
        """Save cryptocurrency analysis results to JSON, CSV, and log files."""
        try:
            if not analysis.get("success", False):
                logger.error(f"Cannot save analysis: {analysis.get('message', 'Unknown error')}")
                return {"success": False, "message": analysis.get('message', 'Unknown error')}

            os.makedirs(output_dir, exist_ok=True)
            ticker = analysis.get("ticker", "UNKNOWN")
            timestamp = analysis.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")).replace(" ", "_").replace(":", "")
            sanitized_ticker = ticker.replace(".", "_").replace("-", "_")

            json_filename = os.path.join(output_dir, f"{sanitized_ticker}_analysis_{timestamp}.json")
            csv_filename = os.path.join(output_dir, f"{sanitized_ticker}_summary_{timestamp}.csv")
            log_filename = os.path.join(output_dir, "crypto_analysis_logs.txt")

            # Save full analysis to JSON
            json_data = self.convert_np_types(analysis)
            try:
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                logger.info(f"Saved full analysis to {json_filename}")
            except Exception as e:
                logger.error(f"Failed to save JSON file {json_filename}: {e}")
                return {"success": False, "message": f"Failed to save JSON file: {str(e)}"}

            # Prepare CSV summary
            ml_analysis = analysis.get("ml_analysis", {})
            current_price = analysis.get("current_price", "N/A")
            predicted_price = ml_analysis.get("prediction", {}).get("predicted_price", "N/A") if ml_analysis.get("success") else "N/A"
            predicted_change_pct = (
                ((predicted_price - current_price) / current_price * 100)
                if isinstance(current_price, (int, float)) and isinstance(predicted_price, (int, float)) and current_price != 0
                else "N/A"
            )
            confidence_score = ml_analysis.get("prediction", {}).get("confidence_score", "N/A") if ml_analysis.get("success") else "N/A"
            risk_level = analysis.get("risk_metrics", {}).get("risk_level", "N/A")
            recommendation = analysis.get("recommendation", "N/A")

            csv_data = {
                "Ticker": ticker,
                "Symbol": analysis.get("symbol", "N/A"),
                "Current_Price_USD": current_price if isinstance(current_price, (int, float)) else "N/A",
                "Predicted_Price_USD": predicted_price if isinstance(predicted_price, (int, float)) else "N/A",
                "Predicted_Change_Pct": f"{predicted_change_pct:.2f}" if isinstance(predicted_change_pct, (int, float)) else "N/A",
                "Confidence_Score": f"{confidence_score:.2f}" if isinstance(confidence_score, (int, float)) else "N/A",
                "Recommendation": recommendation,
                "Risk_Level": risk_level,
                "Market_Cap": analysis.get("market_cap", "N/A"),
                "Volume": analysis.get("volume", "N/A"),
                "Timestamp": analysis.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }

            try:
                with open(csv_filename, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                    writer.writeheader()
                    writer.writerow(csv_data)
                logger.info(f"Saved summary to {csv_filename}")
            except Exception as e:
                logger.error(f"Failed to save CSV file {csv_filename}: {e}")
                return {"success": False, "message": f"Failed to save CSV file: {str(e)}"}

            # Prepare log entry
            log_entry = f"\n[{timestamp}] Analysis for {ticker} ({analysis.get('symbol', 'N/A')})\n"
            log_entry += "=" * 60 + "\n"
            log_entry += "Cryptocurrency Analysis Report\n"
            log_entry += "=" * 60 + "\n"

            # Basic Info
            log_entry += f"Current Price (USD): {current_price:.2f}\n" if isinstance(current_price, (int, float)) else "Current Price (USD): N/A\n"
            log_entry += f"Predicted Price (USD): {predicted_price:.2f}\n" if isinstance(predicted_price, (int, float)) else "Predicted Price (USD): N/A\n"
            log_entry += f"Predicted Change (%): {predicted_change_pct:.2f}\n" if isinstance(predicted_change_pct, (int, float)) else "Predicted Change (%): N/A\n"
            log_entry += f"Confidence Score: {confidence_score:.2f}\n" if isinstance(confidence_score, (int, float)) else "Confidence Score: N/A\n"
            log_entry += f"Recommendation: {recommendation}\n"
            log_entry += f"Risk Level: {risk_level}\n"

            # Technical Indicators
            tech_indicators = analysis.get("technical_indicators", {})
            log_entry += "\nTechnical Indicators:\n"
            log_entry += f"  RSI: {tech_indicators.get('RSI', 'N/A'):.2f}\n"
            log_entry += f"  MACD: {tech_indicators.get('MACD', 'N/A'):.4f}\n"
            log_entry += f"  SMA_50: {tech_indicators.get('SMA_50', 'N/A'):.2f}\n"
            log_entry += f"  SMA_200: {tech_indicators.get('SMA_200', 'N/A'):.2f}\n"
            log_entry += f"  Volatility (20d): {tech_indicators.get('Volatility_20d', 'N/A'):.4f}\n"

            # Patterns Detected
            patterns = analysis.get("patterns_detected", [])
            log_entry += "\nPatterns Detected:\n"
            if patterns:
                for pattern in patterns:
                    log_entry += f"  {pattern.get('name', 'Unknown')}: {pattern.get('description', 'N/A')} "
                    log_entry += f"(Impact: {pattern.get('impact', 'N/A')}, Significance: {pattern.get('significance', 'N/A')})\n"
            else:
                log_entry += "  None\n"

            # Sentiment Analysis
            sentiment = analysis.get("sentiment_analysis", {}).get("aggregated", {})
            log_entry += "\nSentiment Analysis:\n"
            log_entry += f"  Positive: {sentiment.get('positive', 0)}\n"
            log_entry += f"  Negative: {sentiment.get('negative', 0)}\n"
            log_entry += f"  Neutral: {sentiment.get('neutral', 0)}\n"

            # ML Analysis
            if ml_analysis.get("success"):
                log_entry += "\nMachine Learning Analysis:\n"
                log_entry += f"  Predicted Trend: {ml_analysis.get('prediction', {}).get('trend_prediction', 'N/A')}\n"
                log_entry += "  Model Performance:\n"
                for model, scores in ml_analysis.get("model_performance", {}).items():
                    log_entry += f"    {model}:\n"
                    log_entry += f"      MSE: {scores.get('MSE', 'N/A'):.4f}\n"
                    log_entry += f"      RMSE: {scores.get('RMSE', 'N/A'):.4f}\n"
                    log_entry += f"      MAE: {scores.get('MAE', 'N/A'):.4f}\n"
                    log_entry += f"      R2: {scores.get('R2', 'N/A'):.4f}\n"

            # RL Optimization
            rl_optimization = analysis.get("rl_optimization", {})
            if rl_optimization.get("success"):
                log_entry += "\nReinforcement Learning Optimization:\n"
                log_entry += f"  Recommendation: {rl_optimization.get('recommendation', 'N/A')}\n"
                log_entry += f"  Performance (%): {rl_optimization.get('performance_pct', 'N/A'):.2f}\n"
                log_entry += f"  Average Reward: {rl_optimization.get('average_reward', 'N/A'):.4f}\n"
                log_entry += "  Actions Distribution:\n"
                for action, prob in rl_optimization.get("actions_distribution", {}).items():
                    log_entry += f"    {action}: {prob:.4f}\n"

            # Backtest Results
            backtest = analysis.get("backtest_results", {})
            if backtest.get("success"):
                log_entry += "\nBacktest Results:\n"
                log_entry += f"  Total Return (%): {backtest.get('total_return_pct', 'N/A'):.2f}\n"
                log_entry += f"  Number of Trades: {backtest.get('number_of_trades', 'N/A')}\n"
                log_entry += f"  Win Rate (%): {backtest.get('win_rate_pct', 'N/A'):.2f}\n"
                log_entry += f"  Total PnL: {backtest.get('total_pnl', 'N/A'):.2f}\n"

            try:
                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(log_entry)
                logger.info(f"Appended analysis log to {log_filename}")
            except Exception as e:
                logger.error(f"Failed to save log file {log_filename}: {e}")
                return {"success": False, "message": f"Failed to save log file: {str(e)}"}

            return {
                "success": True,
                "json_file": json_filename,
                "csv_file": csv_filename,
                "log_file": log_filename
            }

        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return {"success": False, "message": f"Error saving analysis: {str(e)}"}
# --- End of crypto.py ---

# --- Start of main.py ---
class CryptoTradingBot:
    def __init__(self, config):
        self.config = config
        self.data_feed = DataFeed(config["tickers"])
        self.portfolio = VirtualPortfolio(config)
        self.executor = PaperExecutor(self.portfolio, config)
        self.tracker = PortfolioTracker(self.portfolio, config)
        self.reporter = PerformanceReport(self.portfolio)
        self.crypto_analyzer = Crypto()
        self.initialize()

    def initialize(self):
        """Initialize the bot and its components."""
        logger.info("Initializing Crypto Trading Bot...")
        self.portfolio.initialize_portfolio()

    def run_analysis(self, ticker):
        """Run analysis for a given ticker and return the result."""
        try:
            analysis = self.crypto_analyzer.analyze_crypto(ticker, period=self.config.get("period", "3y"))
            if analysis.get("success"):
                # Log ML data points and features
                ml_data = analysis.get("ml_data", {})
                logger.info(f"ML Data Points for {ticker}: {len(ml_data)}")
                if len(ml_data) < self.config.get("min_data_points", 100):
                    logger.warning(f"Insufficient ML data points for {ticker}: {len(ml_data)} < {self.config.get('min_data_points', 100)}")
                if "features" in ml_data:
                    logger.info(f"ML Features for {ticker}: {ml_data['features']}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return {"success": False, "ticker": ticker, "message": str(e)}

    def make_trading_decision(self, analysis):
        """Make a trading decision based on a weighted combination of indicators, risk metrics, sentiment, ML/DL, and RL."""
        if not analysis.get("success"):
            logger.warning(f"Skipping trading decision for {analysis.get('ticker')} due to failed analysis: {analysis.get('message')}")
            return None

        ticker = analysis["ticker"]
        current_price = analysis["current_price"]
        metrics = self.portfolio.get_metrics()
        available_cash = metrics["cash"]
        total_value = metrics["total_value"]

        # Check cooldown period
        last_trade = self.portfolio.trade_log[-1] if self.portfolio.trade_log else {}
        cooldown_until = last_trade.get("cooldown_until")
        if cooldown_until:
            cooldown_time = datetime.strptime(cooldown_until, "%Y-%m-%d %H:%M:%S")
            if datetime.now() < cooldown_time:
                logger.info(f"Skipping {ticker} due to cooldown until {cooldown_until}")
                return None

        # Extract relevant analysis components
        technical_indicators = analysis.get("technical_indicators", {})
        patterns = analysis.get("patterns_detected", [])
        risk_metrics = analysis.get("risk_metrics", {})
        sentiment = analysis.get("sentiment_analysis", {}).get("aggregated", {})
        ml_analysis = analysis.get("ml_analysis", {})
        rl_optimization = analysis.get("rl_optimization", {})
        backtest_results = analysis.get("backtest_results", {})
        support_level = analysis.get("support_level", current_price * 0.95)
        resistance_level = analysis.get("resistance_level", current_price * 1.05)

        # Initialize decision score and weights
        decision_score = 0.0
        weights = {
            "technical": 0.25,  # Technical indicators and patterns
            "sentiment": 0.15,  # Sentiment analysis
            "ml_dl": 0.35,      # Increased weight for ML/DL
            "rl": 0.20,         # RL optimization
            "risk": 0.05        # Reduced weight for risk
        }

        # 1. Technical Indicator and Pattern Score
        technical_score = 0.0
        pattern_count = len([p for p in patterns if p["significance"] in ["High", "Medium"]])
        bullish_patterns = len([p for p in patterns if p["impact"] == "Bullish"])
        bearish_patterns = len([p for p in patterns if p["impact"] == "Bearish"])

        # RSI-based signal
        rsi = technical_indicators.get("RSI")
        if rsi:
            if rsi < 30:
                technical_score += 0.3  # Oversold, bullish
            elif rsi > 70:
                technical_score -= 0.3  # Overbought, bearish

        # MACD-based signal
        macd = technical_indicators.get("MACD")
        macd_signal = technical_indicators.get("MACD_Signal")
        if macd and macd_signal:
            if macd > macd_signal:
                technical_score += 0.2  # Bullish crossover
            elif macd < macd_signal:
                technical_score -= 0.2  # Bearish crossover

        # Moving Average signal
        sma_50 = technical_indicators.get("SMA_50")
        sma_200 = technical_indicators.get("SMA_200")
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                technical_score += 0.2  # Bullish trend
            else:
                technical_score -= 0.2  # Bearish trend

        # Pattern-based signal (increased weight)
        if pattern_count > 0:
            technical_score += (bullish_patterns - bearish_patterns) * 0.15  # Increased from 0.1

        # Normalize technical score to [-1, 1]
        technical_score = max(min(technical_score, 1.0), -1.0)
        decision_score += weights["technical"] * technical_score

        # 2. Sentiment Score
        sentiment_score = 0.0
        total_sentiment = sentiment.get("positive", 0) + sentiment.get("negative", 0) + sentiment.get("neutral", 0)
        if total_sentiment > 0:
            sentiment_score = (sentiment.get("positive", 0) - sentiment.get("negative", 0)) / total_sentiment
        decision_score += weights["sentiment"] * sentiment_score

        # 3. ML/DL Prediction Score (Updated)
        ml_score = 0.0
        ensemble_r2 = 0.0
        predicted_price = 0.0
        predicted_change = 0.0
        if ml_analysis.get("success"):
            ensemble_r2 = ml_analysis["model_performance"].get("Enhanced Ensemble", {}).get("R2", 0)
            predicted_price = ml_analysis["prediction"]["predicted_price"]
            confidence = ml_analysis["prediction"]["confidence_score"]
            predicted_change = (predicted_price - current_price) / current_price if current_price > 0 else 0
            confidence_threshold = 0.1  # Lowered from 0.5 to allow low-confidence predictions
            if ml_analysis["prediction"]["trend_prediction"] == "UPTREND" and confidence >= confidence_threshold:
                ml_score = confidence * predicted_change
                logger.info(f"ML Score set for {ticker}: UPTREND, Confidence={confidence:.2f}, Predicted_Change={predicted_change:.4f}")
            elif ml_analysis["prediction"]["trend_prediction"] == "DOWNTREND" and confidence >= confidence_threshold:
                ml_score = -confidence * abs(predicted_change)
                logger.info(f"ML Score set for {ticker}: DOWNTREND, Confidence={confidence:.2f}, Predicted_Change={predicted_change:.4f}")
            else:
                logger.info(f"ML Score remains 0 for {ticker}: Confidence {confidence:.2f} below threshold {confidence_threshold:.2f}")
            # Scale by R2 but allow negative R2 to reduce score magnitude rather than zero it
            r2_scaling = max(abs(ensemble_r2), 0.1) if ensemble_r2 != 0 else 0.1
            ml_score *= r2_scaling
            if ensemble_r2 <= 0:
                logger.info(f"Using ML score for {ticker} despite negative R2: {ensemble_r2:.2f}, Confidence: {confidence:.2f}, ML_Score: {ml_score:.4f}")
        else:
            logger.warning(f"ML analysis failed for {ticker}: {ml_analysis.get('message')}")
        decision_score += weights["ml_dl"] * max(min(ml_score, 1.0), -1.0)

        # 4. RL Optimization Score
        rl_score = 0.0
        rl_recommendation = "HOLD"
        performance_pct = 0.0
        if rl_optimization.get("success"):
            rl_recommendation = rl_optimization["recommendation"]
            performance_pct = rl_optimization["performance_pct"]
            if rl_recommendation == "BUY":
                rl_score = min(performance_pct / 100, 1.0) * 0.8
            elif rl_recommendation == "SELL":
                rl_score = -min(abs(performance_pct) / 100, 1.0) * 0.8
            elif rl_recommendation == "HOLD":
                rl_score = 0.0
        decision_score += weights["rl"] * rl_score

        # 5. Risk Metrics Score
        risk_score = 0.0
        volatility = risk_metrics.get("annual_volatility", 0)
        sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
        var_95 = risk_metrics.get("var_95", 0)
        max_drawdown = risk_metrics.get("max_drawdown", 0)
        if volatility > 0.5:  # Relaxed from 0.3
            risk_score -= 0.3
        if var_95 < -0.10:  # Relaxed from -0.05
            risk_score -= 0.2
        if sharpe_ratio > 1.0:
            risk_score += 0.3
        decision_score += weights["risk"] * risk_score

        # Dynamic Confidence Threshold
        base_confidence_threshold = 0.5  # Lowered from 0.6
        confidence_threshold = base_confidence_threshold + min(volatility * 0.1, 0.1)  # Reduced from 0.2

        # Position Sizing with Risk Adjustment
        risk_per_trade = total_value * 0.01  # Risk 1% of portfolio
        volatility_adjustment = max(1.0 - volatility, 0.5)
        stop_loss_price = support_level if decision_score > 0 else current_price * 0.95
        risk_per_unit = abs(current_price - stop_loss_price) if decision_score != 0 else current_price * 0.05
        qty = (risk_per_trade / risk_per_unit) * volatility_adjustment if risk_per_unit > 0 else 0
        qty = min(qty, available_cash / current_price) if current_price > 0 else 0

        # Calculate exposure
        current_exposure = metrics["total_exposure"]
        max_exposure = total_value * 0.5

        def format_value(value, format_spec=".2f"):
            return f"{value:{format_spec}}" if value is not None else "N/A"

        try:
            logger.info(f"Decision Details for {ticker}:")
            logger.info(f"Technical Score: {technical_score:.3f}, Signals: {technical_score > 0}")
            logger.info(f"Technical Indicator Details: "
                        f"RSI={format_value(rsi)}, "
                        f"MACD={format_value(macd)}, "
                        f"Signal={format_value(macd_signal)}, "
                        f"Price={current_price:.2f}, "
                        f"BB_Upper={technical_indicators.get('BB_Upper', 0):.2f}, "
                        f"SMA_50={format_value(sma_50)}, "
                        f"SMA_200={format_value(sma_200)}")
            logger.info(f"Sentiment Score: {sentiment_score:.3f}, Signals: {sentiment_score > 0}")
            logger.info(f"Sentiment Details: Positive={sentiment.get('positive', 0)}, "
                        f"Negative={sentiment.get('negative', 0)}, Neutral={sentiment.get('neutral', 0)}")
            logger.info(f"ML Score: {ml_score:.3f}, Signals: {ml_score > 0}")
            if ml_analysis.get("success"):
                logger.info(f"ML Details: Predicted_Price={predicted_price:.2f}, "
                            f"R2={ensemble_r2:.2f}, "
                            f"Price_Change_Pct={predicted_change * 100:.4f}%")
            logger.info(f"RL Score: {rl_score:.3f}, Signals: {rl_score > 0}")
            if rl_optimization.get("success"):
                logger.info(f"RL Details: Recommendation={rl_recommendation}, "
                            f"Performance_Pct={performance_pct:.2f}%")
            logger.info(f"Risk Score: {risk_score:.3f}, Signals: {risk_score > 0}")
            logger.info(f"Risk Details: Volatility={volatility:.2f}, Sharpe_Ratio={sharpe_ratio:.2f}, "
                        f"VaR_95={var_95:.2f}, Max_Drawdown={max_drawdown:.2f}")
            logger.info(f"Confidence Threshold: {confidence_threshold:.3f}")
            logger.info(f"Buy Signals: {bullish_patterns}, Sell Signals: {bearish_patterns}")
            logger.info(f"Available Cash: ${available_cash:.2f}, Qty: {qty:.4f}")
            logger.info(f"Current Asset Exposure: ${current_exposure:.2f}, Max: ${max_exposure:.2f}")
            logger.info(f"Portfolio Holdings: {self.portfolio.holdings}")
        except Exception as e:
            logger.error(f"Error logging decision details: {e}")

        # Trading Decision
        trade = None
        no_trade_reasons = []
        if (decision_score > confidence_threshold and
            qty * current_price <= available_cash and
            bullish_patterns >= bearish_patterns and
            current_price > support_level):
            logger.info(f"Executing BUY for {ticker}: {qty:.4f} units at ${current_price:.2f} "
                        f"(Score: {decision_score:.2f}, Patterns: {bullish_patterns}/{bearish_patterns})")
            success = self.executor.execute_trade(ticker, "buy", qty, current_price)
            trade = {"action": "buy", "ticker": ticker, "qty": qty, "price": current_price, "success": success}
        elif (decision_score < -confidence_threshold and
              bearish_patterns >= bullish_patterns and
              current_price < resistance_level):
            holding_qty = self.portfolio.holdings.get(ticker, {}).get("qty", 0)
            qty = min(qty, holding_qty) if holding_qty > 0 else qty  # Support short selling
            logger.info(f"Executing SELL for {ticker}: {qty:.4f} units at ${current_price:.2f} "
                        f"(Score: {decision_score:.2f}, Patterns: {bullish_patterns}/{bearish_patterns})")
            success = self.executor.execute_trade(ticker, "sell", qty, current_price)
            trade = {"action": "sell", "ticker": ticker, "qty": qty, "price": current_price, "success": success}
        else:
            if decision_score <= confidence_threshold:
                no_trade_reasons.append(f"Decision score {decision_score:.2f} <= threshold {confidence_threshold:.2f}")
            if qty * current_price > available_cash:
                no_trade_reasons.append(f"Insufficient cash: {qty * current_price:.2f} > {available_cash:.2f}")
            if bullish_patterns < bearish_patterns and decision_score > 0:
                no_trade_reasons.append(f"Bullish patterns {bullish_patterns} < Bearish patterns {bearish_patterns}")
            if current_price <= support_level and decision_score > 0:
                no_trade_reasons.append(f"Price {current_price:.2f} <= Support {support_level:.2f}")
            if decision_score >= -confidence_threshold:
                no_trade_reasons.append(f"Decision score {decision_score:.2f} >= -threshold {-confidence_threshold:.2f}")
            if bearish_patterns < bullish_patterns and decision_score < 0:
                no_trade_reasons.append(f"Bearish patterns {bearish_patterns} < Bullish patterns {bullish_patterns}")
            if current_price >= resistance_level and decision_score < 0:
                no_trade_reasons.append(f"Price {current_price:.2f} >= Resistance {resistance_level:.2f}")
            logger.info(f"No trade executed for {ticker}: {'; '.join(no_trade_reasons)}")

        # Cooldown period after a trade
        if trade and trade["success"]:
            self.portfolio.trade_log.append({
                "ticker": ticker,
                "action": trade["action"],
                "qty": qty,
                "price": current_price,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cooldown_until": (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            })
            self.portfolio.save_trade_log()

        return trade

    def run(self):
        """Main bot loop to run analysis, make trades, and generate reports."""
        logger.info("Starting Crypto Trading Bot...")
        while True:
            try:
                # Log metrics at the start of the loop
                logger.info("Logging portfolio metrics at start of trading cycle...")
                self.tracker.log_metrics()

                for ticker in self.config["tickers"]:
                    logger.info(f"Processing {ticker}...")
                    analysis = self.run_analysis(ticker)
                    if analysis.get("success"):
                        self.crypto_analyzer.save_analysis_to_files(analysis, ticker)
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
                time.sleep(self.config.get("sleep_interval", 3600))
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

def main():
    """Entry point for the trading bot."""
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
        "tickers": ["Solana"],
        "starting_balance": current_cash,
        "current_portfolio_value": portfolio_value,
        "current_pnl": current_pnl,
        "mode": "paper",
        "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_api_secret": os.getenv("ALPACA_API_SECRET"),
        "base_url": "https://paper-api.alpaca.markets",
        "period": "3y",  # Extended from 2y
        "sleep_interval": 3600,
        "min_data_points": 100,
        "ml_epochs": 200  
    }

    bot = CryptoTradingBot(config)
    bot.run()

if __name__ == "__main__":
    main()



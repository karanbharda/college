import json
import time
import os
import csv
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
import backtrader as bt
import optuna
from optuna.trial import Trial
import geocoder
import praw
import traceback
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gnews import GNews
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('commodity_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === PortfolioTracker Class ===
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

# === PaperExecutor Class ===
class PaperExecutor:
    """Executes simulated trades in paper trading mode."""
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

# === DataFeed Class ===
class DataFeed:
    """Fetches live market prices for specified tickers."""
    def __init__(self, tickers):
        self.tickers = tickers

    def get_live_prices(self):
        """Fetch live prices for specified tickers using yfinance."""
        data = {}
        for ticker in self.tickers:
            try:
                commodity = yf.Ticker(ticker)
                df = commodity.history(period="1d", interval="1m")
                if not df.empty:
                    latest = df.iloc[-1]
                    data[ticker] = {
                        "price": latest["Close"],
                        "volume": latest["Volume"]
                    }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }

# === VirtualPortfolio Class ===
class VirtualPortfolio:
    """Manages a virtual portfolio for paper trading."""
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
            print(f"Error executing buy order: {e}")
            return False

    def sell(self, asset, qty, price):
        """Execute a sell order in paper trading mode."""
        if asset not in self.holdings or self.holdings[asset]["qty"] < qty:
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
            print(f"Error executing sell order: {e}")
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
            "current_portfolio_value": self.config.get("current_portfolio_value", 0),
            "current_pnl": self.config.get("current_pnl", 0),
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
        with open(self.portfolio_file, "w") as f:
            json.dump({"cash": self.cash, "holdings": self.holdings}, f, indent=4)

    def save_trade_log(self):
        """Save trade log to JSON file."""
        with open(self.trade_log_file, "w") as f:
            json.dump(self.trade_log, f, indent=4)

    def get_current_prices(self):
        """Fetch current prices from Alpaca API."""
        prices = {}
        for asset in self.holdings:
            try:
                bar = self.api.get_bars(asset, timeframe="1Min", limit=1).df
                if not bar.empty:
                    prices[asset] = {"price": bar["close"].iloc[-1], "volume": bar["volume"].iloc[-1]}
            except Exception as e:
                print(f"Error fetching price for {asset}: {e}")
        return prices

# === PerformanceReport Class ===
class PerformanceReport:
    """Generates daily performance reports for the portfolio."""
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
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

        return report

# === Commodity Class ===
class Commodity:
    """Analyzes commodities and provides trading recommendations."""
    # Commodity symbols with ETF mapping
    commodity_symbols = {
        "gold": "GC=F",
        "gold_etf": "GLD",
        "silver": "SI=F",
        "silver_etf": "SLV",
        "platinum": "PL=F",
        "platinum_etf": "PPLT",
        "palladium": "PA=F",
        "palladium_etf": "PALL",
        "oil": "CL=F",
        "oil_etf": "USO",
        "crude oil": "CL=F",
        "crude_oil_etf": "USO",
        "brent crude": "BZ=F",
        "brent_crude_etf": "BNO",
        "natural gas": "NG=F",
        "natural_gas_etf": "UNG",
        "heating oil": "HO=F",
        "rbob gasoline": "RB=F",
        "ethanol": "ZK=F",
        "copper": "HG=F",
        "copper_etf": "COPX",
        "aluminum": "ALI=F",
        "zinc": "ZNC=F",
        "nickel": "LN=F",
        "lead": "LL=F",
        "tin": "LT=F",
        "corn": "ZC=F",
        "corn_etf": "CORN",
        "wheat": "ZW=F",
        "wheat_etf": "WEAT",
        "soybeans": "ZS=F",
        "soybean oil": "ZL=F",
        "soybean meal": "ZM=F",
        "oats": "ZO=F",
        "rough rice": "ZR=F",
        "coffee": "KC=F",
        "cocoa": "CC=F",
        "sugar": "SB=F",
        "cotton": "CT=F",
        "orange juice": "OJ=F",
        "live cattle": "LE=F",
        "feeder cattle": "GF=F",
        "lean hogs": "HE=F",
        "lumber": "LBS=F",
        "bloomberg commodity index": "BCOM=F",
        "s&p gsci": "SPGSCI=F",
        "thomson reuters crb": "CRY=F",
        "butter": "CB=F",
        "milk": "DC=F",
        "eu allowances carbon": "MO=F",
        "rubber": "JN=F",
        "palm oil": "KPO=F",
        "iron ore": "TIO=F",
        "uranium": "UX=F",
        "us dollar index": "DX=F",
        "euro fx": "6E=F",
        "british pound": "6B=F",
        "canadian dollar": "6C=F",
        "japanese yen": "6J=F",
        "swiss franc": "6S=F",
        "australian dollar": "6A=F",
        "t-notes 2yr": "ZT=F",
        "t-notes 5yr": "ZF=F",
        "t-notes 10yr": "ZN=F",
        "t-bonds 30yr": "ZB=F",
        "vix futures": "VX=F",
        "broad_commodity_etf": "DBC",
        "bitcoin futures": "BTC=F",
        "bitcoin_etf": "BITO",
        "ether futures": "ETH=F",
        "ether_etf": "EETH"
    }

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize API keys from environment variables
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.STOCKTWITS_BASE_URL = os.getenv("STOCKTWITS_BASE_URL", "https://api.stocktwits.com/api/2/streams/symbol")
        self.FMPC_API_KEY = os.getenv("FMPC_API_KEY")
        self.MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
        self.LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY")
        self.CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
        self.SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
        self.COINGECKO_API = os.getenv("COINGECKO_API", "https://api.coingecko.com/api/v3")
        self.REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
        self.REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
        self.REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

        # Validate required API keys
        required_keys = [
            "ALPHA_VANTAGE_API_KEY", "FMPC_API_KEY", "MARKETAUX_API_KEY",
            "CRYPTOPANIC_API_KEY", "SANTIMENT_API_KEY",
            "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"
        ]
        for key in required_keys:
            if not getattr(self, key):
                print(f"Warning: {key} is not set in environment variables")

        self.analyzer = SentimentIntensityAnalyzer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_epoch_logs = {}
        self.transformer_epoch_logs = {}

    @staticmethod
    def convert_np_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [Commodity.convert_np_types(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: Commodity.convert_np_types(v) for k, v in obj.items()}
        else:
            return obj.item() if hasattr(obj, "item") else obj

    def fetch_exchange_rates(self):
        """Fetch cryptocurrency exchange rates from CoinGecko."""
        try:
            url = f"{self.COINGECKO_API}/simple/price"
            params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            print(f"Error fetching exchange rates: {e}")
            return {"bitcoin": {"usd": 1}, "ethereum": {"usd": 1}}

    def convert_price(self, price, exchange_rates):
        """Convert price to USD, BTC, and ETH."""
        return {
            "USD": round(price, 2),
            "BTC": round(price / exchange_rates.get("bitcoin", {}).get("usd", 1), 8),
            "ETH": round(price / exchange_rates.get("ethereum", {}).get("usd", 1), 8)
        }

    def fetch_combined_commodity_sentiment(self, commodity_name):
        """Fetch and aggregate sentiment from Google News and Reddit."""
        google_news_sentiment = self._google_news_sentiment(commodity_name)
        reddit_sentiment = self._reddit_sentiment(commodity_name)
        sources = {"google_news": google_news_sentiment, "reddit": reddit_sentiment}
        aggregated = {"positive": 0, "negative": 0, "neutral": 0}
        for src, sentiment in sources.items():
            for key in aggregated:
                aggregated[key] += sentiment.get(key, 0)
        return {"sources": sources, "aggregated": aggregated}

    def _google_news_sentiment(self, commodity):
        """Analyze sentiment from Google News articles."""
        try:
            news = GNews()
            articles = news.get_news(commodity)
            positive, negative, neutral = 0, 0, 0
            for article in articles:
                content = f"{article['title']} {article['description']}".lower()
                sentiment = self.analyzer.polarity_scores(content)
                if sentiment['compound'] > 0.05:
                    positive += 1
                elif sentiment['compound'] < -0.05:
                    negative += 1
                else:
                    neutral += 1
            return {"positive": positive, "negative": negative, "neutral": neutral}
        except Exception as e:
            return {"error": f"Google News error: {str(e)}"}

    def _reddit_sentiment(self, commodity):
        """Analyze sentiment from Reddit posts."""
        if not all([self.REDDIT_CLIENT_ID, self.REDDIT_CLIENT_SECRET, self.REDDIT_USER_AGENT]):
            return {"error": "Reddit API credentials missing"}
        try:
            reddit = praw.Reddit(
                client_id=self.REDDIT_CLIENT_ID,
                client_secret=self.REDDIT_CLIENT_SECRET,
                user_agent=self.REDDIT_USER_AGENT
            )
            subreddit = reddit.subreddit("stocks")
            posts = subreddit.search(commodity, limit=100)
            positive, negative, neutral = 0, 0, 0
            for post in posts:
                title = post.title.lower()
                if any(word in title for word in ["buy", "bullish", "great", "strong", "rally"]):
                    positive += 1
                elif any(word in title for word in ["sell", "bearish", "weak", "bad", "plunge"]):
                    negative += 1
                else:
                    neutral += 1
            return {"positive": positive, "negative": negative, "neutral": neutral}
        except Exception as e:
            return {"error": f"Reddit error: {str(e)}"}

    def find_underperforming_commodities_with_potential(self):
        """Identify commodities with short-term underperformance but long-term potential."""
        try:
            results = []
            for name, symbol in self.commodity_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
                    if hist.empty:
                        continue
                    current_price = hist['Close'].iloc[-1]
                    one_month_ago_price = hist['Close'].iloc[-22] if len(hist) > 22 else hist['Close'].iloc[0]
                    six_months_ago_price = hist['Close'].iloc[-126] if len(hist) > 126 else hist['Close'].iloc[0]
                    one_year_ago_price = hist['Close'].iloc[0]
                    one_month_change = (current_price - one_month_ago_price) / one_month_ago_price * 100
                    six_month_change = (current_price - six_months_ago_price) / six_months_ago_price * 100
                    one_year_change = (current_price - one_year_ago_price) / one_year_ago_price * 100
                    if one_month_change < 0 and one_year_change > 10:
                        results.append({
                            "name": name,
                            "symbol": symbol,
                            "current_price": float(current_price),
                            "one_month_change": round(one_month_change, 2),
                            "six_month_change": round(six_month_change, 2),
                            "one_year_change": round(one_year_change, 2),
                            "score": round(one_year_change - one_month_change, 2)
                        })
                except Exception as e:
                    continue
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:3]
        except Exception as e:
            print(f"Error finding underperforming commodities: {e}")
            return []

    def get_user_location(self):
        """Get user's location based on IP address."""
        try:
            g = geocoder.ip('me')
            location = g.json
            if not location or not location.get('ok'):
                return None
            return {
                "city": location.get('city'),
                "state": location.get('state'),
                "country": location.get('country'),
                "continent": location.get('continent', ''),
                "lat": location.get('lat'),
                "lng": location.get('lng')
            }
        except Exception as e:
            print(f"Error getting user location: {e}")
            return None

    def get_location_based_recommendations(self, location):
        """Provide commodity recommendations based on user location."""
        if not location:
            return {"error": "Could not determine location"}
        try:
            country = location.get('country', '').lower()
            continent = location.get('continent', '').lower()
            recommendations = []
            if continent == 'north america' or country in ['us', 'usa', 'canada', 'mexico']:
                recommendations.extend([
                    {"commodity": "corn", "reason": "Major North American agricultural commodity with local price advantages"},
                    {"commodity": "natural gas", "reason": "North American natural gas prices often differ from global markets"}
                ])
            elif continent == 'europe' or country in ['uk', 'france', 'germany', 'italy', 'spain']:
                recommendations.extend([
                    {"commodity": "brent crude", "reason": "European benchmark for oil prices with local relevance"},
                    {"commodity": "eu allowances carbon", "reason": "Important for European energy markets and industrial planning"}
                ])
            elif continent == 'asia' or country in ['china', 'japan', 'india', 'singapore', 'korea']:
                recommendations.extend([
                    {"commodity": "rice", "reason": "Asian rice markets often have regional price dynamics"},
                    {"commodity": "palm oil", "reason": "Important commodity in Asian markets and food supply chains"}
                ])
            elif continent == 'oceania' or country in ['australia', 'new zealand']:
                recommendations.extend([
                    {"commodity": "wool", "reason": "Important Australian export with local market knowledge advantages"},
                    {"commodity": "iron ore", "reason": "Australia is a major producer with local price trends"}
                ])
            elif continent == 'africa' or country in ['south africa', 'nigeria', 'egypt', 'morocco']:
                recommendations.extend([
                    {"commodity": "cocoa", "reason": "Major African export with regional supply chain advantages"},
                    {"commodity": "gold", "reason": "Important commodity in several African economies"}
                ])
            if not recommendations:
                recommendations = [
                    {"commodity": "gold", "reason": "Global safe-haven asset with universal appeal"},
                    {"commodity": "crude oil", "reason": "Essential global commodity with impact on your local economy"}
                ]
            return {
                "location": {"country": location.get('country'), "continent": location.get('continent')},
                "recommendations": recommendations
            }
        except Exception as e:
            print(f"Error getting location-based recommendations: {e}")
            return {"error": str(e)}

    def _generate_detailed_commodity_recommendation(self, commodity_data, recommendation, buy_score, sell_score, price_to_sma200, trend_direction, sentiment_score, volatility, sharpe_ratio, seasonal_factors):
        """Generate a detailed recommendation explanation for a commodity."""
        symbol = commodity_data["symbol"]
        name = commodity_data["name"]
        price = commodity_data["current_price"]
        explanation = f"RECOMMENDATION FOR {name} ({symbol}): {recommendation}\n\n"
        if recommendation == "STRONG BUY":
            explanation += f"We have a STRONG BUY recommendation for {name} with a buy score of {buy_score:.2f}. Multiple technical, sentiment, and seasonal factors align positively. "
        elif recommendation == "BUY":
            explanation += f"We recommend a BUY for {name} with a buy score of {buy_score:.2f}. The balance of indicators suggests positive momentum. "
        elif recommendation == "HOLD":
            explanation += f"We recommend to HOLD {name} positions. Buy score: {buy_score:.2f}, Sell score: {sell_score:.2f}. Technical indicators are mixed, suggesting a neutral outlook. "
        elif recommendation == "SELL":
            explanation += f"We recommend a SELL for {name} with a sell score of {sell_score:.2f}. Several indicators point to potential downside. "
        elif recommendation == "STRONG SELL":
            explanation += f"We have a STRONG SELL recommendation for {name} with a sell_score of {sell_score:.2f}. Multiple technical, sentiment, and seasonal factors align negatively. "
        explanation += f"\nTrend Analysis: {name} is currently in a {trend_direction.lower()}. "
        if trend_direction == "UPTREND":
            explanation += f"The current price is {(price_to_sma200-1)*100:.1f}% above the 200-day moving average, indicating positive long-term momentum. "
        else:
            explanation += f"The current price is {(1-price_to_sma200)*100:.1f}% below the 200-day moving average, indicating negative long-term momentum. "
        sentiment_text = (
            "strongly positive" if sentiment_score > 0.7 else "positive" if sentiment_score > 0.6 else
            "slightly positive" if sentiment_score > 0.5 else "slightly negative" if sentiment_score > 0.4 else
            "negative" if sentiment_score > 0.3 else "strongly negative"
        )
        explanation += f"\nMarket Sentiment: Current sentiment for {name} is {sentiment_text}. "
        risk_level = "high" if volatility > 0.03 else "moderate" if volatility > 0.015 else "low"
        explanation += f"\nRisk Assessment: {name} currently shows {risk_level} volatility ({volatility:.3f}). "
        explanation += f"The Sharpe ratio is {sharpe_ratio:.2f}, "
        if sharpe_ratio > 1:
            explanation += "indicating good risk-adjusted returns. "
        elif sharpe_ratio > 0:
            explanation += "suggesting moderate risk-adjusted returns. "
        else:
            explanation += "pointing to poor risk-adjusted returns. "
        explanation += f"\nSeasonal Analysis: {seasonal_factors['description']}. "
        explanation += f"\nConclusion: Based on our comprehensive analysis, we recommend {recommendation} for {name} at the current price of ${price:.2f}. "
        if recommendation in ["STRONG BUY", "BUY"]:
            explanation += "Consider allocating capital in accordance with your risk tolerance and portfolio strategy. "
        elif recommendation in ["STRONG SELL", "SELL"]:
            explanation += "Consider reducing exposure in line with your overall portfolio strategy and risk management principles. "
        explanation += "Monitor for changes in trend direction and watch key support/resistance levels."
        return explanation

    def _generate_ml_insights_explanation(self, ml_analysis):
        """Generate explanation for machine learning analysis results."""
        prediction = ml_analysis["prediction"]
        explanation = f"The machine learning model predicts a price of ${prediction['predicted_price']:.2f} in {prediction['predicted_timeframe']}, representing a {prediction['predicted_change_pct']:.2f}% "
        if prediction['predicted_change_pct'] > 0:
            explanation += "increase from the current price. "
        else:
            explanation += "decrease from the current price. "
        explanation += f"The model shows a confidence score of {prediction['confidence_score']:.2f}. "
        if "patterns_detected" in ml_analysis and ml_analysis["patterns_detected"]:
            explanation += "\n\nKey patterns detected:\n"
            for pattern in ml_analysis["patterns_detected"]:
                explanation += f"- {pattern['name']}: {pattern['description']} ({pattern['impact']} with {pattern['significance'].lower()} significance)\n"
        if "top_factors" in ml_analysis and ml_analysis["top_factors"]:
            explanation += "\nTop influencing factors in prediction:\n"
            for factor, details in ml_analysis["top_factors"].items():
                explanation += f"- {factor}: {details['importance']:.3f} importance\n"
        if "model_performance" in ml_analysis:
            best_model = min(ml_analysis["model_performance"].items(), key=lambda x: x[1].get("MAE", float('inf')))[0]
            best_r2 = max([metrics.get("R2", 0) for _, metrics in ml_analysis["model_performance"].items()])
            explanation += f"\nAnalysis used ensemble of models with best performance from {best_model} (R² = {best_r2:.2f})"
        return explanation

    class LSTMModel(nn.Module):
        """LSTM model for time series prediction."""
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    class TransformerModel(nn.Module):
        """Transformer model for time series prediction."""
        def __init__(self, input_size, d_model, n_heads, num_layers, output_size):
            super().__init__()
            self.embedding = nn.Linear(input_size, d_model)
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=n_heads,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x):
            x = self.embedding(x)
            out = self.transformer(x, x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    class TimeSeriesDataset(Dataset):
        """Dataset for time series data."""
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def calculate_rsi(self, data, periods=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(com=periods-1, adjust=False).mean()
        roll_down = down.ewm(com=periods-1, adjust=False).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.replace([np.inf, -np.inf], np.nan)

    def calculate_mpt_metrics(self, commodity_history, benchmark_symbols=None):
        """Calculate Modern Portfolio Theory metrics."""
        daily_returns = commodity_history['Close'].pct_change().dropna()
        annual_return = (1 + daily_returns.mean()) ** 252 - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        benchmark_metrics = {}
        if benchmark_symbols:
            try:
                benchmark_data = yf.download(benchmark_symbols, period="1y")['Adj Close']
                benchmark_returns = benchmark_data.pct_change().dropna()
                def calculate_beta(commodity_returns, market_returns):
                    cov_matrix = np.cov(commodity_returns, market_returns)
                    market_variance = np.var(market_returns)
                    return cov_matrix[0, 1] / market_variance if market_variance != 0 else 0
                market_returns = benchmark_returns.iloc[:, 0] if isinstance(benchmark_returns, pd.DataFrame) and not benchmark_returns.empty else yf.download('^GSPC', period="1y")['Adj Close'].pct_change().dropna()
                beta = calculate_beta(daily_returns, market_returns)
                benchmark_metrics = {
                    "beta": beta,
                    "benchmark_returns": {
                        sym: {
                            "annual_return": float((1 + bench_returns.mean()) ** 252 - 1) if not bench_returns.empty else 0,
                            "annual_volatility": float(bench_returns.std() * np.sqrt(252)) if not bench_returns.empty else 0
                        } for sym, bench_returns in (benchmark_returns.items() if isinstance(benchmark_returns, pd.DataFrame) else {benchmark_symbols[0]: benchmark_returns}.items())
                    }
                }
            except Exception as e:
                print(f"Benchmark analysis error: {e}")
        return {
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "risk_free_rate": risk_free_rate,
            **benchmark_metrics
        }

    def fetch_with_retry(self, ticker, period, max_retries=3):
        """Fetch historical data with retry mechanism."""
        for attempt in range(max_retries):
            try:
                data = ticker.history(period=period)
                if not data.empty:
                    return data
                else:
                    print(f"{ticker.ticker}: No data found on attempt {attempt + 1}")
            except (ConnectionError, requests.exceptions.RequestException) as e:
                print(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                break
        return pd.DataFrame()

    def analyze_commodity(self, symbol, benchmark_symbols=None, prediction_days=30, training_period="3y"):
        """Perform comprehensive analysis of a commodity."""
        try:
            symbol = symbol.lower().strip()
            print(f"Fetching and analyzing data for {symbol}...")
            if symbol not in self.commodity_symbols:
                print(f"Invalid commodity name. Available commodities: {', '.join(self.commodity_symbols.keys())}")
                return {"success": False, "message": f"Invalid commodity: {symbol}"}
            ticker_symbol = self.commodity_symbols[symbol]
            commodity_name = symbol
            commodity = yf.Ticker(ticker_symbol)
            history = self.fetch_with_retry(commodity, "1y")
            if history.empty:
                print(f"Error: No price data found for {symbol}.")
                return {"success": False, "message": f"Unable to fetch data for {symbol}: No price data found"}
            history = history.replace([np.inf, -np.inf], np.nan).dropna()
            if len(history) < 50:
                return {"success": False, "message": f"Insufficient data points for {symbol}"}
            commodity_info = commodity.info
            current_price = float(history["Close"].iloc[-1])
            volume = commodity_info.get("volume", "N/A")
            high_52w = commodity_info.get("fiftyTwoWeekHigh", "N/A")
            low_52w = commodity_info.get("fiftyTwoWeekLow", "N/A")
            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()
            history["EMA_50"] = history["Close"].ewm(span=50, adjust=False).mean()
            history["RSI"] = self.calculate_rsi(history["Close"])
            history["BB_Middle"] = history["Close"].rolling(window=20).mean()
            std_dev = history["Close"].rolling(window=20).std()
            history["BB_Upper"] = history["BB_Middle"] + 2 * std_dev
            history["BB_Lower"] = history["BB_Middle"] - 2 * std_dev
            exp1 = history["Close"].ewm(span=12, adjust=False).mean()
            exp2 = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = exp1 - exp2
            history["Signal_Line"] = history["MACD"].ewm(span=9, adjust=False).mean()
            history["MACD_Histogram"] = history["MACD"] - history["Signal_Line"]
            history["Daily_Return"] = history["Close"].pct_change()
            history["Volatility"] = history["Daily_Return"].rolling(window=30).std()
            history = history.replace([np.inf, -np.inf], np.nan)
            mpt_metrics = self.calculate_mpt_metrics(history, benchmark_symbols or ['^GSPC'])
            sma_50 = float(history["SMA_50"].iloc[-1]) if not pd.isna(history["SMA_50"].iloc[-1]) else 0
            sma_200 = float(history["SMA_200"].iloc[-1]) if not pd.isna(history["SMA_200"].iloc[-1]) else 0
            ema_50 = float(history["EMA_50"].iloc[-1]) if not pd.isna(history["EMA_50"].iloc[-1]) else 0
            volatility = float(history["Volatility"].iloc[-1]) if not pd.isna(history["Volatility"].iloc[-1]) else 0
            rsi = float(history["RSI"].iloc[-1]) if not pd.isna(history["RSI"].iloc[-1]) else 50
            bb_upper = float(history["BB_Upper"].iloc[-1]) if not pd.isna(history["BB_Upper"].iloc[-1]) else current_price
            bb_lower = float(history["BB_Lower"].iloc[-1]) if not pd.isna(history["BB_Lower"].iloc[-1]) else current_price
            macd = float(history["MACD"].iloc[-1]) if not pd.isna(history["MACD"].iloc[-1]) else 0
            signal_line = float(history["Signal_Line"].iloc[-1]) if not pd.isna(history["Signal_Line"].iloc[-1]) else 0
            macd_histogram = float(history["MACD_Histogram"].iloc[-1]) if not pd.isna(history["MACD_Histogram"].iloc[-1]) else 0
            momentum = (current_price - history["Close"].iloc[-30]) / history["Close"].iloc[-30] if len(history) >= 30 and history["Close"].iloc[-30] != 0 else 0
            fundamental_data = {"inventory_levels": "N/A", "production_data": "N/A", "consumption_data": "N/A"}
            print(f"Fetching sentiment for {symbol}...")
            sentiment_data = self.fetch_combined_commodity_sentiment(symbol)
            sentiment = sentiment_data["aggregated"]
            total_sentiment = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]
            sentiment_score = sentiment["positive"] / total_sentiment if total_sentiment > 0 else 0.5
            price_to_sma200 = current_price / sma_200 if sma_200 > 0 else 1
            price_to_sma50 = current_price / sma_50 if sma_50 > 0 else 1
            trend_direction = "UPTREND" if sma_50 > sma_200 else "DOWNTREND"
            volume_trend = "HIGH" if isinstance(volume, (int, float)) and volume > 1000000 else "MODERATE"
            current_month = pd.Timestamp.now().month
            seasonal_patterns = {
                "GC=F": {"strong_months": [1, 2, 8, 9], "weak_months": [3, 4, 10]},
                "CL=F": {"strong_months": [1, 2, 11, 12], "weak_months": [7, 8, 9]},
                "GLD": {"strong_months": [1, 2, 8, 9], "weak_months": [3, 4, 10]},
                "SI=F": {"strong_months": [1, 2, 8, 9], "weak_months": [3, 4, 10]},
                "SLV": {"strong_months": [1, 2, 8, 9], "weak_months": [3, 4, 10]} 
            }
            commodity_pattern = seasonal_patterns.get(ticker_symbol, {"strong_months": [], "weak_months": []})
            seasonal_factors = {}
            if current_month in commodity_pattern["strong_months"]:
                seasonal_factors["seasonal_trend"] = "STRONG"
                seasonal_factors["description"] = f"Current month ({current_month}) historically strong for {symbol}"
                seasonal_adjustment = 0.1
            elif current_month in commodity_pattern["weak_months"]:
                seasonal_factors["seasonal_trend"] = "WEAK"
                seasonal_factors["description"] = f"Current month ({current_month}) historically weak for {symbol}"
                seasonal_adjustment = -0.1
            else:
                seasonal_factors["seasonal_trend"] = "NEUTRAL"
                seasonal_factors["description"] = f"Current month ({current_month}) historically neutral for {symbol}"
                seasonal_adjustment = 0
            buy_score = (
                (1 - price_to_sma200) * 0.2 + (1 if trend_direction == "UPTREND" else -0.5) * 0.2 +
                (sentiment_score - 0.5) * 0.25 + (mpt_metrics["sharpe_ratio"] * 0.15) +
                (0.5 - volatility) * 0.1 + seasonal_adjustment * 0.1
            )
            sell_score = (
                (price_to_sma200 - 1) * 0.2 + (1 if trend_direction == "DOWNTREND" else -0.5) * 0.2 +
                ((1 - sentiment_score) - 0.5) * 0.25 + (-mpt_metrics["sharpe_ratio"] * 0.15) +
                (volatility - 0.5) * 0.1 + (-seasonal_adjustment * 0.1)
            )
            recommendation = (
                "STRONG BUY" if buy_score > 0.2 else "BUY" if buy_score > 0 else
                "STRONG SELL" if sell_score > 0.2 else "SELL" if sell_score > 0 else "HOLD"
            )
            support_level = min(sma_200, sma_50) * 0.95
            resistance_level = max(current_price * 1.05, sma_50 * 1.05)
            risk_level = "HIGH" if volatility > 0.03 else "MEDIUM" if volatility > 0.015 else "LOW"
            timeframe = (
                "LONG_TERM" if trend_direction == "UPTREND" and sentiment_score > 0.6 else
                "MEDIUM_TERM" if trend_direction == "UPTREND" else "SHORT_TERM"
            )
            commodity_data = {"symbol": ticker_symbol, "name": symbol, "current_price": current_price, "sector": "Commodities"}
            explanation = self._generate_detailed_commodity_recommendation(
                commodity_data, recommendation, buy_score, sell_score, price_to_sma200, trend_direction,
                sentiment_score, volatility, mpt_metrics["sharpe_ratio"], seasonal_factors
            )
            print(f"Fetching extended data for {symbol} for ML analysis...")
            extended_history = self.fetch_with_retry(commodity, training_period)
            if extended_history.empty:
                print(f"Unable to fetch sufficient extended historical data for {symbol}")
                ml_analysis = {"success": False, "message": f"Unable to fetch sufficient historical data for ML analysis of {symbol}"}
            else:
                print(f"Engineering features for ML pattern recognition...")
                data = extended_history[['Close']].copy()
                data['SMA_5'] = extended_history['Close'].rolling(window=5).mean()
                data['SMA_20'] = extended_history['Close'].rolling(window=20).mean()
                data['SMA_50'] = extended_history['Close'].rolling(window=50).mean()
                data['SMA_200'] = extended_history['Close'].rolling(window=200).mean()
                data['EMA_12'] = extended_history['Close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = extended_history['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
                data['RSI'] = self.calculate_rsi(extended_history['Close'])
                data['BB_Middle'] = data['SMA_20']
                stddev = extended_history['Close'].rolling(window=20).std()
                data['BB_Upper'] = data['BB_Middle'] + 2 * stddev
                data['BB_Lower'] = data['BB_Middle'] - 2 * stddev
                data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle'].replace(0, np.finfo(float).eps)
                data['Volume_Change'] = extended_history['Volume'].pct_change()
                data['Volume_SMA_5'] = extended_history['Volume'].rolling(window=5).mean()
                data['Volume_SMA_20'] = extended_history['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = extended_history['Volume'] / data['Volume_SMA_5'].replace(0, np.finfo(float).eps)
                data['Price_Change'] = extended_history['Close'].pct_change()
                data['Price_Change_5d'] = extended_history['Close'].pct_change(periods=5)
                data['Price_Change_20d'] = extended_history['Close'].pct_change(periods=20)
                data['Volatility_5d'] = data['Price_Change'].rolling(window=5).std()
                data['Volatility_20d'] = data['Price_Change'].rolling(window=20).std()
                data['Price_to_SMA50'] = extended_history['Close'] / data['SMA_50'] - 1
                data['Price_to_SMA200'] = extended_history['Close'] / data['SMA_200'] - 1
                data['ROC_5'] = (extended_history['Close'] / extended_history['Close'].shift(5) - 1) * 100
                data['ROC_10'] = (extended_history['Close'] / extended_history['Close'].shift(10) - 1) * 100
                obv = pd.Series(index=extended_history.index)
                obv.iloc[0] = 0
                for i in range(1, len(extended_history)):
                    if extended_history['Close'].iloc[i] > extended_history['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + extended_history['Volume'].iloc[i]
                    elif extended_history['Close'].iloc[i] < extended_history['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - extended_history['Volume'].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                data['OBV'] = obv
                data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
                low_14 = extended_history['Low'].rolling(window=14).min()
                high_14 = extended_history['High'].rolling(window=14).max()
                data['%K'] = (extended_history['Close'] - low_14) / (high_14 - low_14) * 100
                data['%D'] = data['%K'].rolling(window=3).mean()
                tr1 = abs(extended_history['High'] - extended_history['Low'])
                tr2 = abs(extended_history['High'] - extended_history['Close'].shift())
                tr3 = abs(extended_history['Low'] - extended_history['Close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                plus_dm = extended_history['High'].diff()
                minus_dm = extended_history['Low'].diff().mul(-1)
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
                for month in range(1, 13):
                    data[f'Month_{month}'] = extended_history.index.month == month
                for day in range(0, 5):
                    data[f'Day_{day}'] = extended_history.index.dayofweek == day
                data['Target'] = data['Close'].shift(-prediction_days)
                data = data.replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) < 100:
                    ml_analysis = {"success": False, "message": f"Insufficient data points for {symbol} after feature engineering"}
                else:
                    sequence_length = 20
                    X_lstm, y_lstm = [], []
                    for i in range(len(data) - sequence_length - prediction_days):
                        X_lstm.append(data.iloc[i:i+sequence_length][['Close', 'SMA_20', 'RSI']].values)
                        y_lstm.append(data['Close'].iloc[i+sequence_length+prediction_days-1])
                    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                    if X_lstm.size == 0 or y_lstm.size == 0:
                        ml_analysis = {"success": False, "message": "Empty training data for LSTM/Transformer"}
                    else:
                        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
                        scaler_lstm = MinMaxScaler()
                        X_train_lstm_scaled = scaler_lstm.fit_transform(X_train_lstm.reshape(-1, X_train_lstm.shape[-1])).reshape(X_train_lstm.shape)
                        X_test_lstm_scaled = scaler_lstm.transform(X_test_lstm.reshape(-1, X_test_lstm.shape[-1])).reshape(X_test_lstm.shape)
                        self.lstm_epoch_logs = {}
                        print(f"Starting LSTM training for {symbol}...")
                        try:
                            lstm_model = self.LSTMModel(input_size=X_train_lstm.shape[2], hidden_size=64, num_layers=2, output_size=1).to(self.device)
                            optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                            criterion = nn.MSELoss()
                            train_dataset = self.TimeSeriesDataset(X_train_lstm_scaled, y_train_lstm)
                            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                            for epoch in range(50):
                                lstm_model.train()
                                epoch_loss = 0
                                for batch_X, batch_y in train_loader:
                                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                                    optimizer.zero_grad()
                                    output = lstm_model(batch_X)
                                    loss = criterion(output, batch_y.unsqueeze(1))
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item()
                                self.lstm_epoch_logs[epoch + 1] = epoch_loss / len(train_loader)
                                if (epoch + 1) % 10 == 0:
                                    print(f"LSTM Epoch {epoch + 1}: Loss = {self.lstm_epoch_logs[epoch + 1]:.4f}")
                            lstm_model.eval()
                            with torch.no_grad():
                                X_test_tensor = torch.FloatTensor(X_test_lstm_scaled).to(self.device)
                                lstm_pred = lstm_model(X_test_tensor).cpu().numpy().flatten()
                                latest_sequence = data.tail(sequence_length)[['Close', 'SMA_20', 'RSI']].values
                                latest_sequence_scaled = scaler_lstm.transform(latest_sequence.reshape(-1, X_lstm.shape[2])).reshape(1, sequence_length, X_lstm.shape[2])
                                lstm_future_pred = lstm_model(torch.FloatTensor(latest_sequence_scaled).to(self.device)).cpu().item()
                        except Exception as e:
                            print(f"LSTM training failed: {e}")
                            ml_analysis = {"success": False, "message": f"LSTM training failed: {str(e)}"}
                            return {
                                "success": False,
                                "message": f"Error analyzing {symbol}: LSTM training failed",
                                "error_details": str(e)
                            }
                        self.transformer_epoch_logs = {}
                        print(f"Starting Transformer training for {symbol}...")
                        try:
                            transformer_model = self.TransformerModel(input_size=X_train_lstm.shape[2], d_model=64, n_heads=4, num_layers=2, output_size=1).to(self.device)
                            optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
                            criterion = nn.MSELoss()
                            for epoch in range(50):
                                transformer_model.train()
                                epoch_loss = 0
                                for batch_X, batch_y in train_loader:
                                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                                    optimizer.zero_grad()
                                    output = transformer_model(batch_X)
                                    loss = criterion(output, batch_y.unsqueeze(1))
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item()
                                self.transformer_epoch_logs[epoch + 1] = epoch_loss / len(train_loader)
                                if (epoch + 1) % 10 == 0:
                                    print(f"Transformer Epoch {epoch + 1}: Loss = {self.transformer_epoch_logs[epoch + 1]:.4f}")
                            transformer_model.eval()
                            with torch.no_grad():
                                transformer_pred = transformer_model(X_test_tensor).cpu().numpy().flatten()
                                transformer_future_pred = transformer_model(torch.FloatTensor(latest_sequence_scaled).to(self.device)).cpu().item()
                        except Exception as e:
                            print(f"Transformer training failed: {e}")
                            ml_analysis = {"success": False, "message": f"Transformer training failed: {str(e)}"}
                            return {
                                "success": False,
                                "message": f"Error analyzing {symbol}: Transformer training failed",
                                "error_details": str(e)
                            }
                        X = data.drop(['Target', 'Close'], axis=1)
                        y = data['Target']
                        X.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
                        X = X.dropna()
                        if X.empty:
                            ml_analysis = {"success": False, "message": "Feature matrix X is empty after cleaning"}
                        else:
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
                                try:
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
                                except Exception as e:
                                    print(f"Model {name} training failed: {e}")
                                    model_results[name] = {"MSE": "N/A", "RMSE": "N/A", "MAE": "N/A", "R2": "N/A"}
                            ensemble_pred = np.mean([predictions[name] for name in predictions.keys() if name in predictions], axis=0) if predictions else np.zeros_like(y_test)
                            if len(ensemble_pred) > 0:
                                ensemble_mse = mean_squared_error(y_test, ensemble_pred)
                                ensemble_rmse = np.sqrt(ensemble_mse)
                                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                                ensemble_r2 = r2_score(y_test, ensemble_pred)
                                model_results["Ensemble"] = {"MSE": ensemble_mse, "RMSE": ensemble_rmse, "MAE": ensemble_mae, "R2": ensemble_r2}
                            else:
                                model_results["Ensemble"] = {"MSE": "N/A", "RMSE": "N/A", "MAE": "N/A", "R2": "N/A"}
                            latest_data = X.iloc[-1].values.reshape(1, -1)
                            latest_data_scaled = scaler.transform(latest_data)
                            future_predictions = {}
                            for name, model in models.items():
                                try:
                                    future_predictions[name] = model.predict(latest_data_scaled)[0]
                                except Exception as e:
                                    print(f"Prediction for {name} failed: {e}")
                                    future_predictions[name] = current_price
                            future_predictions["Ensemble"] = np.mean([pred for name, pred in future_predictions.items() if name != "Ensemble"]) if future_predictions else current_price
                            lstm_ensemble_pred = (lstm_pred + transformer_pred) / 2 if len(lstm_pred) > 0 and len(transformer_pred) > 0 else np.zeros_like(y_test_lstm)
                            lstm_ensemble_future_pred = (lstm_future_pred + transformer_future_pred) / 2
                            model_results["LSTM"] = {"MAE": float(mean_absolute_error(y_test_lstm, lstm_pred)) if len(lstm_pred) > 0 else "N/A", "R2": float(r2_score(y_test_lstm, lstm_pred)) if len(lstm_pred) > 0 else "N/A"}
                            model_results["Transformer"] = {"MAE": float(mean_absolute_error(y_test_lstm, transformer_pred)) if len(transformer_pred) > 0 else "N/A", "R2": float(r2_score(y_test_lstm, transformer_pred)) if len(transformer_pred) > 0 else "N/A"}
                            model_results["LSTM_Ensemble"] = {"MAE": float(mean_absolute_error(y_test_lstm, lstm_ensemble_pred)) if len(lstm_ensemble_pred) > 0 else "N/A", "R2": float(r2_score(y_test_lstm, lstm_ensemble_pred)) if len(lstm_ensemble_pred) > 0 else "N/A"}
                            price_volatility = data['Volatility_20d'].iloc[-1] if not pd.isna(data['Volatility_20d'].iloc[-1]) else 0.01
                            confidence_score = max((1 - price_volatility) * (ensemble_r2 if ensemble_r2 != "N/A" else 0.5), 0.1)
                            predicted_price = (future_predictions["Ensemble"] + lstm_ensemble_future_pred) / 2 if future_predictions["Ensemble"] != current_price else lstm_ensemble_future_pred
                            predicted_change = (predicted_price - current_price) / current_price
                            trend_prediction = "UPTREND" if predicted_change > 0 else "DOWNTREND"
                            predicted_change_pct = predicted_change * 100
                            patterns = []
                            if data['SMA_50'].iloc[-2] <= data['SMA_200'].iloc[-2] and data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
                                patterns.append({"name": "Golden Cross", "description": "50-day MA crossed above 200-day MA", "significance": "High", "impact": "Bullish"})
                            elif data['SMA_50'].iloc[-2] >= data['SMA_200'].iloc[-2] and data['SMA_50'].iloc[-1] < data['SMA_200'].iloc[-1]:
                                patterns.append({"name": "Death Cross", "description": "50-day MA crossed below 200-day MA", "significance": "High", "impact": "Bearish"})
                            if data['SMA_20'].iloc[-1] > data['SMA_20'].iloc[-20] and data['SMA_50'].iloc[-1] > data['SMA_50'].iloc[-20]:
                                patterns.append({"name": "Moving Average Uptrend", "description": "Short and medium-term MAs trending up", "significance": "Medium", "impact": "Bullish"})
                            elif data['SMA_20'].iloc[-1] < data['SMA_20'].iloc[-20] and data['SMA_50'].iloc[-1] < data['SMA_50'].iloc[-20]:
                                patterns.append({"name": "Moving Average Downtrend", "description": "Short and medium-term MAs trending down", "significance": "Medium", "impact": "Bearish"})
                            if data['RSI'].iloc[-1] > 70:
                                patterns.append({"name": "Overbought (RSI)", "description": "RSI above 70", "significance": "Medium", "impact": "Bearish"})
                            elif data['RSI'].iloc[-1] < 30:
                                patterns.append({"name": "Oversold (RSI)", "description": "RSI below 30", "significance": "Medium", "impact": "Bullish"})
                            if data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2] and data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
                                patterns.append({"name": "MACD Bullish Crossover", "description": "MACD crossed above signal", "significance": "Medium", "impact": "Bullish"})
                            elif data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2] and data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1]:
                                patterns.append({"name": "MACD Bearish Crossover", "description": "MACD crossed below signal", "significance": "Medium", "impact": "Bearish"})
                            if data['BB_Width'].iloc[-1] < data['BB_Width'].iloc[-20:].mean() * 0.7:
                                patterns.append({"name": "Bollinger Band Squeeze", "description": "Narrowing bands", "significance": "Medium", "impact": "Neutral"})
                            if data['Close'].iloc[-1] <= data['BB_Lower'].iloc[-1] * 1.02 and data['Close'].iloc[-2] > data['BB_Lower'].iloc[-2]:
                                patterns.append({"name": "Support Test", "description": "Price testing lower BB", "significance": "Medium", "impact": "Bullish"})
                            elif data['Close'].iloc[-1] >= data['BB_Upper'].iloc[-1] * 0.98 and data['Close'].iloc[-2] < data['BB_Upper'].iloc[-2]:
                                patterns.append({"name": "Resistance Test", "description": "Price testing upper BB", "significance": "Medium", "impact": "Bearish"})
                            if data['Volume_Ratio'].iloc[-1] > 2.0:
                                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                                    patterns.append({"name": "Bullish Volume Spike", "description": "Volume spike with price rise", "significance": "High", "impact": "Bullish"})
                                else:
                                    patterns.append({"name": "Bearish Volume Spike", "description": "Volume spike with price decline", "significance": "High", "impact": "Bearish"})
                            if data['Close'].iloc[-20:].max() < data['Close'].iloc[-1] and data['Volume_Ratio'].iloc[-1] > 1.5:
                                patterns.append({"name": "Bullish Breakout", "description": "Price breaking above recent high", "significance": "High", "impact": "Bullish"})
                            elif data['Close'].iloc[-20:].min() > data['Close'].iloc[-1] and data['Volume_Ratio'].iloc[-1] > 1.5:
                                patterns.append({"name": "Bearish Breakdown", "description": "Price breaking below recent low", "significance": "High", "impact": "Bearish"})
                            if data['ADX'].iloc[-1] > 25:
                                if data['Plus_DI'].iloc[-1] > data['Minus_DI'].iloc[-1]:
                                    patterns.append({"name": "Strong Uptrend", "description": "ADX > 25 with +DI > -DI", "significance": "High", "impact": "Bullish"})
                                else:
                                    patterns.append({"name": "Strong Downtrend", "description": "ADX > 25 with -DI > +DI", "significance": "High", "impact": "Bearish"})
                            price_higher = data['Close'].iloc[-1] > data['Close'].iloc[-10]
                            rsi_lower = data['RSI'].iloc[-1] < data['RSI'].iloc[-10]
                            price_lower = data['Close'].iloc[-1] < data['Close'].iloc[-10]
                            rsi_higher = data['RSI'].iloc[-1] > data['RSI'].iloc[-10]
                            if price_higher and rsi_lower:
                                patterns.append({"name": "Bearish RSI Divergence", "description": "Higher highs, lower RSI", "significance": "High", "impact": "Bearish"})
                            if price_lower and rsi_higher:
                                patterns.append({"name": "Bullish RSI Divergence", "description": "Lower lows, higher RSI", "significance": "High", "impact": "Bullish"})
                            pattern_score = sum(
                                0.2 if p["impact"] == "Bullish" and p["significance"] == "High" else
                                0.1 if p["impact"] == "Bullish" else
                                -0.2 if p["impact"] == "Bearish" and p["significance"] == "High" else
                                -0.1 if p["impact"] == "Bearish" else 0 for p in patterns
                            )
                            ml_analysis = {
                                "success": True,
                                "prediction": {
                                    "current_price": float(current_price),
                                    "predicted_price": float(predicted_price),
                                    "predicted_change_pct": float(predicted_change_pct),
                                    "predicted_timeframe": f"{prediction_days} days",
                                    "confidence_score": float(confidence_score),
                                    "trend_prediction": trend_prediction
                                },
                                "model_performance": model_results,
                                "patterns_detected": patterns,
                                "pattern_score": pattern_score,
                                "top_factors": {}
                            }
                            if "Random Forest" in feature_importance:
                                sorted_features = sorted(feature_importance["Random Forest"].items(), key=lambda x: x[1], reverse=True)
                                ml_analysis["top_factors"] = {f: {"importance": float(i), "value": float(data[f].iloc[-1])} for f, i in sorted_features[:5] if f in data.columns}
            if ml_analysis.get("success", False):
                predicted_change_pct = ml_analysis["prediction"]["predicted_change_pct"]
                ml_recommendation = (
                    "STRONG BUY" if predicted_change_pct > 5 else "BUY" if predicted_change_pct > 1 else
                    "STRONG SELL" if predicted_change_pct < -5 else "SELL" if predicted_change_pct < -1 else "HOLD"
                )
                if ml_analysis["prediction"]["confidence_score"] > 0.6:
                    if ml_recommendation == "STRONG BUY" and recommendation in ["BUY", "HOLD"]:
                        recommendation = "STRONG BUY"
                    elif ml_recommendation == "STRONG SELL" and recommendation in ["SELL", "HOLD"]:
                        recommendation = "STRONG SELL"
                    elif ml_recommendation == "BUY" and recommendation == "HOLD":
                        recommendation = "BUY"
                    elif ml_recommendation == "SELL" and recommendation == "HOLD":
                        recommendation = "SELL"
                buy_score += ml_analysis["pattern_score"] * 0.5
                sell_score -= ml_analysis["pattern_score"] * 0.5
                explanation += "\n\nMachine Learning Analysis:\n" + self._generate_ml_insights_explanation(ml_analysis)
            print(f"Running backtrader simulations for {symbol}...")
            sim_history = self.fetch_with_retry(commodity, "2y")
            if sim_history.empty:
                print(f"Unable to fetch sufficient data for backtrader simulation of {symbol}")
                backtest_results = {"success": False, "message": "Insufficient data for backtest"}
                optimization_results = {"success": False, "message": "Insufficient data for optimization"}
            else:
                class CommodityStrategy(bt.Strategy):
                    params = (
                        ('sma_short', 50),
                        ('sma_long', 200),
                        ('rsi_low', 30),
                        ('rsi_high', 70),
                        ('size', 100),
                    )

                    def __init__(self):
                        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_short)
                        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_long)
                        self.rsi = bt.indicators.RSI(self.data.close, period=14)
                        self.order = None

                    def next(self):
                        if self.order:
                            return
                        if not self.position:
                            if self.sma_short > self.sma_long and self.rsi < self.params.rsi_low:
                                self.order = self.buy(size=self.params.size)
                        else:
                            if self.sma_short < self.sma_long or self.rsi > self.params.rsi_high:
                                self.order = self.sell(size=self.position.size)

                class OptimizedCommodityStrategy(bt.Strategy):
                    params = (
                        ('sma_short', 50),
                        ('sma_long', 200),
                        ('rsi_low', 30),
                        ('rsi_high', 70),
                        ('size', 100),
                    )

                    def __init__(self):
                        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_short)
                        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_long)
                        self.rsi = bt.indicators.RSI(self.data.close, period=14)
                        self.order = None

                    def next(self):
                        if self.order:
                            return
                        if not self.position:
                            if self.sma_short > self.sma_long and self.rsi < self.params.rsi_low:
                                self.order = self.buy(size=self.params.size)
                        else:
                            if self.sma_short < self.sma_long or self.rsi > self.params.rsi_high:
                                self.order = self.sell(size=self.position.size)

                cerebro = bt.Cerebro()
                data_feed = bt.feeds.PandasData(dataname=sim_history)
                cerebro.adddata(data_feed)
                cerebro.addstrategy(CommodityStrategy)
                cerebro.broker.setcash(100000)
                cerebro.addsizer(bt.sizers.FixedSize, stake=100)
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                # Step 2: Run and capture result
                results = cerebro.run()
                run_results = results[0]  # FIXED
                sharpe_analyzer = run_results.analyzers.sharpe.get_analysis()
                drawdown_analyzer = run_results.analyzers.drawdown.get_analysis()
                trade_analyzer = run_results.analyzers.trades.get_analysis()
                initial_portfolio_value = cerebro.broker.startingcash
                final_portfolio_value = cerebro.broker.getvalue()
                total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
                backtest_results = {
                    "success": True,
                    "initial_portfolio_value": initial_portfolio_value,
                    "final_portfolio_value": final_portfolio_value,
                    "total_return": total_return,
                    "sharpe_ratio": sharpe_analyzer.get('sharperatio', 0) or 0,
                    "max_drawdown": drawdown_analyzer.get('max', {}).get('drawdown', 0) or 0,
                    "total_trades": trade_analyzer.get('total', {}).get('total', 0) or 0,
                    "win_rate": (
                        trade_analyzer.get('won', {}).get('total', 0) /
                        trade_analyzer.get('total', {}).get('total', 1)
                    ) if trade_analyzer.get('total', {}).get('total', 0) > 0 else 0
                }

                def objective(trial: Trial):
                    params = {
                       'sma_short': trial.suggest_int('sma_short', 10, 50),
                       'sma_long': trial.suggest_int('sma_long', 100, 200),
                       'rsi_low': trial.suggest_int('rsi_low', 20, 40),
                       'rsi_high': trial.suggest_int('rsi_high', 60, 80),
                        'size': 100
                         }
                    cerebro_opt = bt.Cerebro()
                    cerebro_opt.adddata(data_feed)
                    cerebro_opt.addstrategy(OptimizedCommodityStrategy, **params)
                    cerebro_opt.broker.setcash(100000)
                    cerebro_opt.addsizer(bt.sizers.FixedSize, stake=100)
                    cerebro_opt.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
                   # 💥 CAPTURE run results correctly
                    results = cerebro_opt.run()
                    run_results = results[0]

                    sharpe = run_results.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
                    return sharpe

                try:
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=10)
                    best_params = study.best_params
                    best_value = study.best_value
                    optimization_results = {
                        "success": True,
                        "best_parameters": best_params,
                        "best_sharpe_ratio": best_value
                    }
                except Exception as e:
                    print(f"Optimization failed: {e}")
                    optimization_results = {"success": False, "message": f"Optimization failed: {str(e)}"}

            location = self.get_user_location()
            location_recommendations = self.get_location_based_recommendations(location) if location else {"error": "Location unavailable"}
            exchange_rates = self.fetch_exchange_rates()
            converted_prices = self.convert_price(current_price, exchange_rates)
            recommendation_data = {
                "success": True,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "commodity": {
                    "name": commodity_name,
                    "symbol": ticker_symbol,
                    "sector": commodity_data["sector"],
                    "current_price": converted_prices,
                    "volume": volume,
                    "52_week_high": high_52w,
                    "52_week_low": low_52w
                },
                "technical_indicators": {
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "ema_50": ema_50,
                    "rsi": rsi,
                    "bollinger_bands": {"upper": bb_upper, "lower": bb_lower},
                    "macd": {"macd": macd, "signal_line": signal_line, "histogram": macd_histogram},
                    "momentum": momentum,
                    "volatility": volatility,
                    "price_to_sma50": price_to_sma50,
                    "price_to_sma200": price_to_sma200
                },
                "fundamental_data": fundamental_data,
                "sentiment_analysis": sentiment_data,
                "portfolio_metrics": mpt_metrics,
                "recommendation": {
                    "action": recommendation,
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "timeframe": timeframe,
                    "support_level": support_level,
                    "resistance_level": resistance_level,
                    "risk_level": risk_level,
                    "explanation": explanation
                },
                "ml_analysis": ml_analysis,
                "backtest_results": backtest_results,
                "optimization_results": optimization_results,
                "market_conditions": {
                    "trend_direction": trend_direction,
                    "volume_trend": volume_trend,
                    "seasonal_factors": seasonal_factors
                },
                "location_based_recommendations": location_recommendations
            }
            return self.convert_np_types(recommendation_data)
        except Exception as e:
            error_message = f"Error analyzing {symbol}: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message,
                "error_details": traceback.format_exc()
            }

    def backtest_commodity_strategy(self, symbol, start_date, end_date, initial_cash=100000):
        """Backtest a commodity trading strategy."""
        try:
            if symbol not in self.commodity_symbols:
                return {"success": False, "message": f"Invalid commodity: {symbol}"}
            ticker_symbol = self.commodity_symbols[symbol]
            data = yf.download(ticker_symbol, start=start_date, end=end_date)
            if data.empty:
                return {"success": False, "message": f"No data found for {symbol}"}

            class CommodityStrategy(bt.Strategy):
                params = (
                    ('sma_short', 50),
                    ('sma_long', 200),
                    ('rsi_low', 30),
                    ('rsi_high', 70),
                    ('size', 100),
                )

                def __init__(self):
                    self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_short)
                    self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_long)
                    self.rsi = bt.indicators.RSI(self.data.close, period=14)
                    self.order = None

                def next(self):
                    if self.order:
                        return
                    if not self.position:
                        if self.sma_short > self.sma_long and self.rsi < self.params.rsi_low:
                            self.order = self.buy(size=self.params.size)
                    else:
                        if self.sma_short < self.sma_long or self.rsi > self.params.rsi_high:
                            self.order = self.sell(size=self.position.size)

            cerebro = bt.Cerebro()
            data_feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(data_feed)
            cerebro.addstrategy(CommodityStrategy)
            cerebro.broker.setcash(initial_cash)
            cerebro.addsizer(bt.sizers.FixedSize, stake=100)
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.run()
            run_results = cerebro.runstrats[0]
            sharpe_analyzer = run_results.analyzers.sharpe.get_analysis()
            drawdown_analyzer = run_results.analyzers.drawdown.get_analysis()
            trade_analyzer = run_results.analyzers.trades.get_analysis()
            initial_portfolio_value = cerebro.broker.startingcash
            final_portfolio_value = cerebro.broker.getvalue()
            total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
            backtest_results = {
                "success": True,
                "initial_portfolio_value": initial_portfolio_value,
                "final_portfolio_value": final_portfolio_value,
                "total_return": total_return,
                "sharpe_ratio": sharpe_analyzer.get('sharperatio', 0) or 0,
                "max_drawdown": drawdown_analyzer.get('max', {}).get('drawdown', 0) or 0,
                "total_trades": trade_analyzer.get('total', {}).get('total', 0) or 0,
                "win_rate": (
                    trade_analyzer.get('won', {}).get('total', 0) /
                    trade_analyzer.get('total', {}).get('total', 1)
                ) if trade_analyzer.get('total', {}).get('total', 0) > 0 else 0
            }
            return backtest_results
        except Exception as e:
            return {
                "success": False,
                "message": f"Error backtesting {symbol}: {str(e)}",
                "error_details": traceback.format_exc()
            }

    def save_analysis_to_files(self, analysis, output_dir="commodity_analysis"):
        """Save commodity analysis results to JSON, CSV, and log files."""
        try:
            if not analysis.get("success", False):
                logger.error(f"Cannot save analysis: {analysis.get('message', 'Unknown error')}")
                return {"success": False, "message": analysis.get('message', 'Unknown error')}

            os.makedirs(output_dir, exist_ok=True)
            ticker = analysis.get("commodity", {}).get("symbol", "UNKNOWN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_ticker = ticker.replace(".", "_")

            json_filename = os.path.join(output_dir, f"{sanitized_ticker}_analysis_{timestamp}.json")
            csv_filename = os.path.join(output_dir, f"{sanitized_ticker}_summary_{timestamp}.csv")
            log_filename = os.path.join(output_dir, "ml_logs.txt")

            # Save full analysis to JSON
            json_data = self.convert_np_types(analysis)
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved full analysis to {json_filename}")

            # Prepare CSV summary
            commodity_data = analysis.get("commodity", {})
            ml_analysis = analysis.get("ml_analysis", {})
            current_price = commodity_data.get("current_price", {}).get("USD", "N/A")
            predicted_price = ml_analysis.get("prediction", {}).get("predicted_price", "N/A") if ml_analysis.get("success", False) else "N/A"
            confidence_score = ml_analysis.get("prediction", {}).get("confidence_score", "N/A") if ml_analysis.get("success", False) else "N/A"
            predicted_change_pct = (
                ((predicted_price - current_price) / current_price * 100)
                if isinstance(current_price, (int, float)) and isinstance(predicted_price, (int, float)) and current_price != 0
                else "N/A"
            )

            csv_data = {
                "Symbol": ticker,
                "Name": commodity_data.get("name", "N/A"),
                "Current_Price_USD": current_price,
                "Predicted_Price_USD": predicted_price,
                "Predicted_Change_Pct": predicted_change_pct,
                "Confidence_Score": confidence_score,
                "Recommendation": analysis.get("recommendation", {}).get("action", "N/A"),
                "Buy_Score": analysis.get("recommendation", {}).get("buy_score", "N/A"),
                "Sell_Score": analysis.get("recommendation", {}).get("sell_score", "N/A"),
                "Risk_Level": analysis.get("recommendation", {}).get("risk_level", "N/A"),
                "Trend_Direction": analysis.get("market_conditions", {}).get("trend_direction", "N/A"),
                "Timestamp": timestamp
            }

            with open(csv_filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
            logger.info(f"Saved summary to {csv_filename}")

            # Log key analysis details
            log_entry = f"\n[{timestamp}] Analysis for {ticker}\n"
            log_entry += "=" * 50 + "\n"
            log_entry += "Commodity Analysis Report\n"
            log_entry += f"Name: {commodity_data.get('name', 'N/A')}\n"
            log_entry += f"Recommendation: {csv_data['Recommendation']}\n"
            log_entry += f"Current Price (USD): {current_price}\n"
            log_entry += f"Predicted Price (USD): {predicted_price}\n"
            log_entry += f"Predicted Change (%): {predicted_change_pct:.2f}\n" if predicted_change_pct != "N/A" else "Predicted Change (%): N/A\n"
            log_entry += f"Confidence Score: {confidence_score}\n"
            log_entry += f"Buy Score: {csv_data['Buy_Score']}\n"
            log_entry += f"Sell Score: {csv_data['Sell_Score']}\n"
            log_entry += f"Risk Level: {csv_data['Risk_Level']}\n"
            log_entry += f"Trend Direction: {csv_data['Trend_Direction']}\n"
            if ml_analysis.get("success", False):
                log_entry += "Model Performance:\n"
                for model, scores in ml_analysis.get("model_performance", {}).items():
                    log_entry += f"  {model}:\n"
                    for metric, value in scores.items():
                        log_entry += f"    {metric}: {value:.4f}\n"
                if self.lstm_epoch_logs:
                    log_entry += "LSTM Epoch Logs:\n"
                    for epoch, loss in self.lstm_epoch_logs.items():
                        log_entry += f"  Epoch {epoch}: Loss = {loss:.4f}\n"
                if self.transformer_epoch_logs:
                    log_entry += "Transformer Epoch Logs:\n"
                    for epoch, loss in self.transformer_epoch_logs.items():
                        log_entry += f"  Epoch {epoch}: Loss = {loss:.4f}\n"

            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.info(f"Appended analysis report to {log_filename}")

            return {
                "success": True,
                "json_file": json_filename,
                "csv_file": csv_filename,
                "log_file": log_filename
            }

        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error saving analysis: {str(e)}"
            }
    
class CommodityTradingBot:
    def __init__(self, config):
        self.config = config
        self.commodity_analyzer = Commodity()
        self.portfolio = VirtualPortfolio(config)
        self.executor = PaperExecutor(self.portfolio, config)
        self.tracker = PortfolioTracker(self.portfolio, config)
        self.reporter = PerformanceReport(self.portfolio)
        
        self.ticker_symbols = [
            self.commodity_analyzer.commodity_symbols[f"{name}_etf"]
            for name in ["gold"]
            if f"{name}_etf" in self.commodity_analyzer.commodity_symbols
        ]
        self.data_feed = DataFeed(self.ticker_symbols)
        
    def initialize(self):
        """Initialize the bot and its components."""
        logger.info("Initializing Commodity Trading Bot...")
        self.portfolio.initialize_portfolio()

    def make_decision(self, commodity, ticker, prices, portfolio_metrics):
        """Make trading decision for a given commodity based on analysis."""
        logger.info(f"Starting decision-making for commodity: {commodity}, ticker: {ticker}")
        logger.info(f"Current prices: {prices}")
        logger.info(f"Portfolio metrics: {portfolio_metrics}")

        max_exposure_per_asset = self.config["starting_balance"] * 0.2  # Limit to 20% of portfolio per asset
        available_cash = portfolio_metrics["cash"]
        logger.info(f"Max exposure per asset: ${max_exposure_per_asset:.2f}, Available cash: ${available_cash:.2f}")

        if not ticker or ticker not in prices:
            logger.error(f"No price data available for {commodity} ({ticker})")
            return

        logger.info(f"Fetching analysis for {commodity}...")
        analysis = self.commodity_analyzer.analyze_commodity(commodity)
        logger.info(f"Analysis result: success={analysis.get('success')}, message={analysis.get('message', 'N/A')}")
        if analysis.get("success"):
            logger.info(f"ML Analysis: {analysis['ml_analysis'].get('prediction', 'N/A')}")
            logger.info(f"Sentiment Analysis: {analysis['sentiment_analysis']}")
            logger.info(f"Backtest Results: {analysis['backtest_results']}")
        # Save analysis to files
            save_result = self.commodity_analyzer.save_analysis_to_files(analysis)
            if save_result.get("success"):
               logger.info(f"Analysis saved: JSON={save_result.get('json_file')}, CSV={save_result.get('csv_file')}, Log={save_result.get('log_file')}")
            else:
               logger.error(f"Failed to save analysis: {save_result.get('message')}")

        if not analysis.get("success"):
           logger.error(f"Analysis failed for {commodity}: {analysis.get('message')}")
           return

        # Extract relevant data from analysis
        logger.info("Extracting analysis components...")
        recommendation = analysis["recommendation"]["action"]
        buy_score = analysis["recommendation"]["buy_score"]
        sell_score = analysis["recommendation"]["sell_score"]
        confidence_score = analysis["ml_analysis"].get("prediction", {}).get("confidence_score", 0.5)
        risk_level = analysis["recommendation"]["risk_level"]
        volatility = analysis["technical_indicators"]["volatility"]
        trend_direction = analysis["market_conditions"]["trend_direction"]
        sentiment_data = analysis["sentiment_analysis"]["aggregated"]
        sentiment_score = (
            sentiment_data["positive"] /
            (sentiment_data["positive"] + sentiment_data["negative"] + sentiment_data["neutral"] or 1)
        )
        ml_pred_change = analysis["ml_analysis"].get("prediction", {}).get("predicted_change_pct", 0)
        backtest_sharpe = analysis["backtest_results"].get("sharpe_ratio", 0)
        backtest_win_rate = analysis["backtest_results"].get("win_rate", 0)
        technical_indicators = analysis["technical_indicators"]
        rsi = technical_indicators["rsi"]
        macd_histogram = technical_indicators["macd"]["histogram"]
        price_to_sma50 = technical_indicators["price_to_sma50"]

        logger.info(f"Recommendation: {recommendation}, Buy Score: {buy_score:.3f}, Sell Score: {sell_score:.3f}")
        logger.info(f"Confidence Score: {confidence_score:.3f}, Risk Level: {risk_level}, Volatility: {volatility:.5f}")
        logger.info(f"Trend Direction: {trend_direction}, Sentiment Score: {sentiment_score:.3f}")
        logger.info(f"ML Predicted Change: {ml_pred_change:.2f}%, Backtest Sharpe: {backtest_sharpe:.2f}, Win Rate: {backtest_win_rate:.2f}")
        logger.info(f"Technical Indicators - RSI: {rsi:.2f}, MACD Histogram: {macd_histogram:.5f}, Price/SMA50: {price_to_sma50:.3f}")

        current_price = prices[ticker]["price"]
        logger.info(f"Current price for {ticker}: ${current_price:.2f}")

        # Calculate composite decision score with adjusted weights
        logger.info("Calculating composite decision score...")
        technical_component = (
            (1 if trend_direction == "UPTREND" else -1) * 0.4 +
            (1 if rsi < 30 else -1 if rsi > 70 else 0) * 0.3 +
            (1 if macd_histogram > 0 else -1) * 0.2 +
            (1 if price_to_sma50 > 1 else -1) * 0.1
        )
        sentiment_component = (sentiment_score - 0.5) * 2
        ml_component = (ml_pred_change / 100) * confidence_score if confidence_score >= 0.3 else 0
        backtest_component = (backtest_sharpe * 0.5 + backtest_win_rate * 0.5)
        composite_score = (
            0.5 * technical_component +  # Increased weight for technicals
            0.15 * sentiment_component +  # Reduced weight for sentiment
            0.2 * ml_component +  # Reduced weight for ML
            0.15 * backtest_component  # Reduced weight for backtest
        )
        logger.info(f"Composite Score Breakdown:")
        logger.info(f"  Technical Component: {technical_component:.3f} (Trend: {(1 if trend_direction == 'UPTREND' else -1)*0.4:.2f}, "
                    f"RSI: {(1 if rsi < 30 else -1 if rsi > 70 else 0)*0.3:.2f}, "
                    f"MACD: {(1 if macd_histogram > 0 else -1)*0.2:.2f}, "
                    f"Price/SMA50: {(1 if price_to_sma50 > 1 else -1)*0.1:.2f})")
        logger.info(f"  Sentiment Component: {sentiment_component:.3f}")
        logger.info(f"  ML Component: {ml_component:.3f} (Ignored if confidence < 0.3)")
        logger.info(f"  Backtest Component: {backtest_component:.3f}")
        logger.info(f"Final Composite Score: {composite_score:.3f}")

        # Adjust recommendation with relaxed confidence thresholds
        logger.info("Determining final recommendation...")
        if composite_score > 0.3 and confidence_score > 0.6:
            final_recommendation = "STRONG BUY"
        elif composite_score > 0.1 and confidence_score > 0.5:
            final_recommendation = "BUY"
        elif composite_score < -0.3 and confidence_score > 0.6:
            final_recommendation = "STRONG SELL"
        elif composite_score < -0.1 and confidence_score > 0.5:
            final_recommendation = "SELL"
        else:
            final_recommendation = "HOLD"
            logger.info("HOLD chosen: Composite score or confidence below thresholds")
        logger.info(f"Initial Final Recommendation: {final_recommendation}")

        # Validate signal with relaxed technical thresholds
        logger.info("Validating signal with technical indicators...")
        is_valid_signal = True
        if final_recommendation in ["BUY", "STRONG BUY"]:
            confirming_signals = sum([
                trend_direction == "UPTREND",
                rsi < 50,  # Relaxed from 40
                macd_histogram > 0,
                price_to_sma50 > 0.99  # Relaxed from 1
            ])
            is_valid_signal = confirming_signals >= 2
            logger.info(f"BUY Signal Validation: {confirming_signals}/2 confirming signals "
                        f"(Trend: {trend_direction == 'UPTREND'}, RSI<50: {rsi < 50}, "
                        f"MACD>0: {macd_histogram > 0}, Price/SMA50>0.99: {price_to_sma50 > 0.99})")
        elif final_recommendation in ["SELL", "STRONG SELL"]:
            confirming_signals = sum([
                trend_direction == "DOWNTREND",
                rsi > 60,
                macd_histogram < 0,
                price_to_sma50 < 1
            ])
            is_valid_signal = confirming_signals >= 2
            logger.info(f"SELL Signal Validation: {confirming_signals}/2 confirming signals "
                        f"(Trend: {trend_direction == 'DOWNTREND'}, RSI>60: {rsi > 60}, "
                        f"MACD<0: {macd_histogram < 0}, Price/SMA50<1: {price_to_sma50 < 1})")

        if not is_valid_signal:
            final_recommendation = "HOLD"
            logger.info(f"Signal invalidated. Final Recommendation adjusted to: {final_recommendation}")

        # Dynamic position sizing based on risk and portfolio constraints
        logger.info("Calculating position size...")
        base_qty = 10  # Base quantity
        risk_adjustment = (
            0.5 if risk_level == "HIGH" else
            0.75 if risk_level == "MEDIUM" else
            1.0
        )
        volatility_adjustment = max(0.1, min(1.0, 0.02 / (volatility or 0.02)))
        adjusted_qty = int(base_qty * risk_adjustment * volatility_adjustment)
        logger.info(f"Position Sizing: Base Qty={base_qty}, Risk Adjustment={risk_adjustment}, "
                    f"Volatility Adjustment={volatility_adjustment:.3f}, Adjusted Qty={adjusted_qty}")

        # Ensure trade size respects portfolio constraints
        trade_value = adjusted_qty * current_price
        logger.info(f"Trade Value: ${trade_value:.2f} ({adjusted_qty} units at ${current_price:.2f})")
        if trade_value > available_cash * 0.5:
            adjusted_qty = int((available_cash * 0.5) / current_price)
            trade_value = adjusted_qty * current_price
            logger.info(f"Adjusted for cash limit: New Qty={adjusted_qty}, New Trade Value=${trade_value:.2f}")
        if trade_value > max_exposure_per_asset:
            adjusted_qty = int(max_exposure_per_asset / current_price)
            trade_value = adjusted_qty * current_price
            logger.info(f"Adjusted for exposure limit: New Qty={adjusted_qty}, New Trade Value=${trade_value:.2f}")

        # Ensure minimum trade size
        adjusted_qty = max(1, adjusted_qty)
        trade_value = adjusted_qty * current_price
        logger.info(f"Final Trade Size: Qty={adjusted_qty}, Trade Value=${trade_value:.2f}")

        # Log decision
        logger.info(f"[{datetime.now()}] Decision Summary for {commodity} ({ticker}): "
                    f"Final Recommendation={final_recommendation}, "
                    f"Composite Score={composite_score:.3f}, "
                    f"Confidence={confidence_score:.3f}, "
                    f"Qty={adjusted_qty}, "
                    f"Trade Value=${trade_value:.2f}")

        # Execute trades based on final recommendation
        logger.info(f"Executing trade for {final_recommendation}...")
        if final_recommendation in ["BUY", "STRONG BUY"]:
            if trade_value <= available_cash and trade_value <= max_exposure_per_asset:
                logger.info(f"Executing BUY: {adjusted_qty} units of {ticker} at ${current_price:.2f}")
                if self.executor.execute_trade(ticker, "buy", adjusted_qty, current_price):
                    logger.info(f"Successfully bought {adjusted_qty} shares of {ticker} at ${current_price:.2f}")
                else:
                    logger.error(f"Failed to buy {ticker}")
            else:
                logger.warning(f"Cannot buy {ticker}: Insufficient cash (${available_cash:.2f}) or exceeds exposure limit (${max_exposure_per_asset:.2f})")
        elif final_recommendation in ["SELL", "STRONG SELL"]:
            holdings = portfolio_metrics["holdings"].get(ticker, {}).get("qty", 0)
            sell_qty = min(adjusted_qty, holdings)
            logger.info(f"Checking SELL: Holdings={holdings}, Sell Qty={sell_qty}")
            if sell_qty > 0:
                logger.info(f"Executing SELL: {sell_qty} units of {ticker} at ${current_price:.2f}")
                if self.executor.execute_trade(ticker, "sell", sell_qty, current_price):
                    logger.info(f"Successfully sold {sell_qty} shares of {ticker} at ${current_price:.2f}")
                else:
                    logger.error(f"Failed to sell {ticker}")
            else:
                logger.warning(f"No holdings to sell for {ticker}")
        else:
            logger.info(f"No trade executed for {ticker} (Recommendation: HOLD)")

        logger.info(f"Completed decision-making for {commodity} ({ticker})")

    def run(self):
        """Main trading loop for commodity trading bot."""
        try:
            # Initialize portfolio and log initial metrics
            self.initialize()
            self.tracker.log_metrics()

            while True:
                prices = self.data_feed.get_live_prices()
                portfolio_metrics = self.portfolio.get_metrics()

                for commodity in ["gold"]:
                    ticker = self.commodity_analyzer.commodity_symbols.get(f"{commodity}_etf")
                    self.make_decision(commodity, ticker, prices, portfolio_metrics)

                # Generate and log performance report
                report = self.reporter.generate_report()
                logger.info(f"[{datetime.now()}] Generated daily report: {report}")

                # Log portfolio metrics periodically
                self.tracker.log_metrics()

                time.sleep(300)  # Run every 5 minutes

        except KeyboardInterrupt:
            logger.info("Shutting down commodity trading bot")
            self.tracker.log_metrics()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            self.tracker.log_metrics()


def main():
    """Main function to run the commodity trading bot."""
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
        "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_api_secret": os.getenv("ALPACA_API_SECRET"),
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        "starting_balance": current_cash,
        "current_portfolio_value": portfolio_value,
        "current_pnl": current_pnl,
        "mode": "paper",
    }

    if not config["alpaca_api_key"] or not config["alpaca_api_secret"]:
        logger.error("Alpaca API credentials are missing")
        return

    bot = CommodityTradingBot(config)
    bot.run()


if __name__ == "__main__":
    main()

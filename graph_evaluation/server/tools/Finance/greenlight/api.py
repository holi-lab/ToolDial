import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_signals(coin: str, exchange: str, market_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Provides BUY/SELL signals and trends for BTC/USDT, ETH/USDT, TRX/USDT 1 hour timeframe.**
		- Types of signals: BUY, SELL, HOLD
		- Types of trends: UP, DOWN, FLAT
		**Use telegram bot for extended functionality**
		Any pair, 6 different timeframes, indicator settings and much more:
		https://t.me/crypto_greenlight_bot?start=CEB66C31"
    coin: BTC, ETH or TRX
        exchange: Binance, Bybit, Huobi, Kucoin, Coinex, MXC, Gate
        market_type: SPOT or FUTURES
        
    """
    url = f"https://greenlight3.p.rapidapi.com/signals"
    querystring = {'coin': coin, 'exchange': exchange, 'market_type': market_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greenlight3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


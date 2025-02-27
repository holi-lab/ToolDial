import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def candles(to: int, symbol: str, resolution: str, is_from: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest candlestick/OHLC data for stocks"
    to: UNIX timestamp. Interval end value
        symbol: Symbol
        resolution: Supported resolution includes 1, 5, 15, 30, 60, D, W, M .Some timeframes might not be available depending on the exchange.
        from: UNIX timestamp. Interval initial value
        
    """
    url = f"https://finnhub-realtime-stock-price.p.rapidapi.com/stock/candle"
    querystring = {'to': to, 'symbol': symbol, 'resolution': resolution, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finnhub-realtime-stock-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote(symbol: str='AAPL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quote"
    symbol: Symbol
        
    """
    url = f"https://finnhub-realtime-stock-price.p.rapidapi.com/quote"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finnhub-realtime-stock-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_symbols(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of stock symbols by exchange"
    exchange: Exchange code
        
    """
    url = f"https://finnhub-realtime-stock-price.p.rapidapi.com/stock/symbol"
    querystring = {'exchange': exchange, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finnhub-realtime-stock-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


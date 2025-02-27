import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_average_price(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current Average Price"
    
    """
    url = f"https://crypto-markets.p.rapidapi.com/avgPrice"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kline_candlestick_data(interval: str, symbol: str, starttime: str=None, endtime: str=None, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kline/Candlestick Data"
    limit: Default 500; max 1000.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/klines"
    querystring = {'interval': interval, 'symbol': symbol, }
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compressed_aggregate_trades_list(symbol: str, limit: int=500, fromid: str=None, endtime: str=None, starttime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compressed/Aggregate Trades List"
    limit: Default 500; max 1000.
        fromid: id to get aggregate trades from INCLUSIVE.
        endtime: Timestamp in ms to get aggregate trades until INCLUSIVE.
        starttime: Timestamp in ms to get aggregate trades from INCLUSIVE.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/aggTrades"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    if fromid:
        querystring['fromId'] = fromid
    if endtime:
        querystring['endTime'] = endtime
    if starttime:
        querystring['startTime'] = starttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_information(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current exchange trading rules and symbol information"
    
    """
    url = f"https://crypto-markets.p.rapidapi.com/exchangeInfo"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_book(symbol: str, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Order Book"
    limit: Default 100; max 5000.
If limit > 5000, then the response will truncate to 5000.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/depth"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_trades_list(symbol: str, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recent Trades List"
    limit: Default 500; max 1000.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/trades"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def old_trade_lookup(symbol: str, fromid: str=None, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get older market trades."
    fromid: Trade id to fetch from. Default gets most recent trades.
        limit: Default 500; max 1000.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/historicalTrades"
    querystring = {'symbol': symbol, }
    if fromid:
        querystring['fromId'] = fromid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_order_book_ticker(symbol: str='BTCUSDT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Symbol Order Book Ticker"
    symbol: If neither parameter is sent, bookTickers for all symbols will be returned in an array.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/ticker/bookTicker"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_price_ticker(symbol: str='BTCUSDT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Symbol Price Ticker"
    symbol: If neither parameter is sent, prices for all symbols will be returned in an array.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/ticker/price"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_24hr_ticker_price_change_statistics(symbol: str='BTCUSDT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24hr Ticker Price Change Statistics"
    symbol: If neither parameter is sent, tickers for all symbols will be returned in an array.
        
    """
    url = f"https://crypto-markets.p.rapidapi.com/ticker/24hr"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_average_price(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current average price for a symbol."
    
    """
    url = f"https://binance46.p.rapidapi.com/ticker/price"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test connectivity to the Rest API and get the current server time."
    
    """
    url = f"https://binance46.p.rapidapi.com/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_info(symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "rateLimits array contains objects related to the exchange's RAW_REQUESTS, REQUEST_WEIGHT, and ORDERS rate limits. These are further defined in the ENUM definitions section under Rate limiters (rateLimitType)."
    
    """
    url = f"https://binance46.p.rapidapi.com/exchangeInfo"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_price_ticker(symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest price for a symbol or symbols."
    symbol: If the symbol is not sent, prices for all symbols will be returned in an array.


        
    """
    url = f"https://binance46.p.rapidapi.com/ticker/price"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_order_book_ticker(symbol: str='BNBUSDT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Best price/qty on the order book for a symbol or symbols."
    symbol: If the symbol is not sent, bookTickers for all symbols will be returned in an array.


        
    """
    url = f"https://binance46.p.rapidapi.com/bookTicker"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_trades_list(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent trades."
    
    """
    url = f"https://binance46.p.rapidapi.com/trades"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def order_book(symbol: str, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adjusted based on the limit:
		
		5, 10, 20, 50, 100 => 1
		500 => 5
		1000 => 10
		5000 => 50"
    limit: Default 100; max 5000. Valid limits:[5, 10, 20, 50, 100, 500, 1000, 5000]


        
    """
    url = f"https://binance46.p.rapidapi.com/depth"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kline_candlestick_data(symbol: str, interval: str, limit: int=500, starttime: str=None, endtime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kline/candlestick bars for a symbol.
		Klines are uniquely identified by their open time."
    limit: Default 500; max 1000.

        starttime: If startTime and endTime are not sent, the most recent klines are returned.


        
    """
    url = f"https://binance46.p.rapidapi.com/klines"
    querystring = {'symbol': symbol, 'interval': interval, }
    if limit:
        querystring['limit'] = limit
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compressed_aggregate_trades_list(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get compressed, aggregate trades. Trades that fill at the time, from the same order, with the same price will have the quantity aggregated."
    
    """
    url = f"https://binance46.p.rapidapi.com/aggTrades"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_24hr_ticker_price_change_statistics(symbol: str='BNBUSDT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24 hour rolling window price change statistics. Careful when accessing this with no symbol."
    
    """
    url = f"https://binance46.p.rapidapi.com/ticker/24hr"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance46.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


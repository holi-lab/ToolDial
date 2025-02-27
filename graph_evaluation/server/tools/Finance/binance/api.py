import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_24hr_ticker_price_change_statistics(symbols: str=None, symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24 hour rolling window price change statistics. Careful when accessing this with no symbol."
    symbols: Required JSON array.

Example: `[\"ETHBTC\",\"BNBBTC\"]`
        symbol: If the symbol is not sent, tickers for all symbols will be returned in an array.
        
    """
    url = f"https://binance43.p.rapidapi.com/ticker/24hr"
    querystring = {}
    if symbols:
        querystring['symbols'] = symbols
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_average_price(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current average price for a symbol."
    
    """
    url = f"https://binance43.p.rapidapi.com/avgPrice"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compressed_aggregate_trades_list(symbol: str, endtime: str=None, limit: int=500, starttime: str=None, fromid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get compressed, aggregate trades. Trades that fill at the time, from the same order, with the same price will have the quantity aggregated."
    endtime: Timestamp in ms to get aggregate trades until INCLUSIVE.

- If startTime and endTime are sent, time between startTime and endTime must be less than 1 hour.
- If fromId, startTime, and endTime are not sent, the most recent aggregate trades will be returned.
        limit: Default 500; max 1000.
        starttime: Timestamp in ms to get aggregate trades from INCLUSIVE.
        fromid: id to get aggregate trades from INCLUSIVE.
        
    """
    url = f"https://binance43.p.rapidapi.com/aggTrades"
    querystring = {'symbol': symbol, }
    if endtime:
        querystring['endTime'] = endtime
    if limit:
        querystring['limit'] = limit
    if starttime:
        querystring['startTime'] = starttime
    if fromid:
        querystring['fromId'] = fromid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
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
		
		5, 10, 20, 50, 100	=> 1
		500 => 5
		1000 => 10
		5000 => 50"
    limit: Default 100; max 5000. Valid limits:[5, 10, 20, 50, 100, 500, 1000, 5000]
        
    """
    url = f"https://binance43.p.rapidapi.com/depth"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_information(permissions: str=None, symbol: str='ETHBTC', symbols: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current exchange trading rules and symbol information"
    permissions: Required STRING or JSON array.

Examples:

- `SPOT`
- `[\"MARGIN\",\"LEVERAGED\"]`
        symbols: Required JSON array.

Example: `[\"BTCUSDT\",\"BNBBTC\"]`
        
    """
    url = f"https://binance43.p.rapidapi.com/exchangeInfo"
    querystring = {}
    if permissions:
        querystring['permissions'] = permissions
    if symbol:
        querystring['symbol'] = symbol
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kline_candlestick_data(interval: str, symbol: str, starttime: str=None, limit: int=500, endtime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kline/candlestick bars for a symbol.
		Klines are uniquely identified by their open time."
    starttime: If startTime and endTime are not sent, the most recent klines are returned.
        limit: Default 500; max 1000.
        
    """
    url = f"https://binance43.p.rapidapi.com/klines"
    querystring = {'interval': interval, 'symbol': symbol, }
    if starttime:
        querystring['startTime'] = starttime
    if limit:
        querystring['limit'] = limit
    if endtime:
        querystring['endTime'] = endtime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_trades_list(symbol: str, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent trades."
    limit: Default 500; max 1000.
        
    """
    url = f"https://binance43.p.rapidapi.com/trades"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
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
    url = f"https://binance43.p.rapidapi.com/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_order_book_ticker(symbols: str=None, symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Best price/qty on the order book for a symbol or symbols."
    symbols: Required JSON array.

Example: `[\"ETHBTC\",\"BNBBTC\"]`
        symbol: If the symbol is not sent, bookTickers for all symbols will be returned in an array.
        
    """
    url = f"https://binance43.p.rapidapi.com/ticker/bookTicker"
    querystring = {}
    if symbols:
        querystring['symbols'] = symbols
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_price_ticker(symbols: str=None, symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest price for a symbol or symbols."
    symbols: Required JSON array.

Example: `[\"ETHBTC\",\"BNBBTC\"]`
        symbol: If the symbol is not sent, prices for all symbols will be returned in an array.
        
    """
    url = f"https://binance43.p.rapidapi.com/ticker/price"
    querystring = {}
    if symbols:
        querystring['symbols'] = symbols
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


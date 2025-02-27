import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_exchange(q: str='B', category: str='Cryptocurrency', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search exchanges by query or category"
    
    """
    url = f"https://stock-cryptocurrency-forex-market-data.p.rapidapi.com/search/exchange"
    querystring = {}
    if q:
        querystring['q'] = q
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-cryptocurrency-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_timeframes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list TimeFrames"
    
    """
    url = f"https://stock-cryptocurrency-forex-market-data.p.rapidapi.com/get/timeframes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-cryptocurrency-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_stock_cryptocurrency_forex(q: str='BBCA', exchange: str='IDX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Stock, Cryptocurrency, Forex price by name or symbol or exchange"
    
    """
    url = f"https://stock-cryptocurrency-forex-market-data.p.rapidapi.com/search/stock"
    querystring = {}
    if q:
        querystring['q'] = q
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-cryptocurrency-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_price(timeframe: str, symbol: str, exchange: str, length: int=60, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stock, Cryptocurrency, Forex price by timeframe"
    
    """
    url = f"https://stock-cryptocurrency-forex-market-data.p.rapidapi.com/prices/stock"
    querystring = {'timeframe': timeframe, 'symbol': symbol, 'exchange': exchange, }
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-cryptocurrency-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange_category(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list available categories of exchanges"
    
    """
    url = f"https://stock-cryptocurrency-forex-market-data.p.rapidapi.com/get/exchange-category"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-cryptocurrency-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


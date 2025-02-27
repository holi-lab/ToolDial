import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forex_symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available foreign exchange currencies"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/symbol/forex"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_symbols(includenames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available crypto  currencies"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/symbol/crypto"
    querystring = {}
    if includenames:
        querystring['includeNames'] = includenames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_symbols(includenames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available stock symbols"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/symbol/stock"
    querystring = {}
    if includenames:
        querystring['includeNames'] = includenames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_symbols(includenames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available ETF symbols"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/symbol/etf"
    querystring = {}
    if includenames:
        querystring['includeNames'] = includenames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_crypto_price(target: str, interval: str, source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historical data of a specific time interval and crypto currency"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/historical/crypto"
    querystring = {'target': target, 'interval': interval, 'source': source, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_forex_price(target: str, interval: str, source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historical data of a specific time interval and foreign exchange currency pair"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/historical/forex"
    querystring = {'target': target, 'interval': interval, 'source': source, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_stock_price(interval: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historical data of a specific time interval and stock symbol"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/historical/stock"
    querystring = {'interval': interval, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_quotes(source: str, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the real time price of a crypto currency pair"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/realtime/crypto"
    querystring = {'source': source, 'target': target, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_quotes(target: str, source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the real time price of a forex currency pair"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/realtime/forex"
    querystring = {'target': target, 'source': source, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_quotes(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the real time price of one or multiple companies"
    
    """
    url = f"https://real-time-quotes1.p.rapidapi.com/api/v1/realtime/stock"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-quotes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


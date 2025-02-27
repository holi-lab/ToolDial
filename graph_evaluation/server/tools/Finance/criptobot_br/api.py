import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_one_pair(base: str, quote: str, exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one coin pair data"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/exchange/{exchange}/market/{base}/{quote}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def signal_by_strategy(strategy: str, symbol: str, exchange: str, timeframe: str, parameters: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "receive candle (or exchange) data and returns signal calculated by strategy.
		If you do not send ohlcv data, you need to send the following informations in request body:
		
		- exchange code
		- symbol (base currency / quote currency)
		-  timeframe 
		- parameters (used from strategy, blank for default)"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/strategy/{strategy}/signal"
    querystring = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'parameters': parameters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pairs(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coin pairs (base coin and quote coin) available from exchange"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/exchange/{exchange}/markets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_strategies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available strategies"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/strategy"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "LIst all exchanges available"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/exchange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_candle_data(quote: str, base: str, timeframe: str, exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data from last 100 candles from a pair by an exchange in a determinate timeframe.
		A list of exchanges and pairs available can be obtained using another endpoints from API. 
		From timeframes, have a list of determinate values that can be used:
		- 1m
		- 5m
		- 15m
		- 30m
		-1h
		- 2h
		- 4h
		- 1d"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/exchange/{exchange}/market/{base}/{quote}/timeframe/{timeframe}/candles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def strategy_parameters(strategy: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return a list of accept parameters for a strategy and your default values"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/strategy/{strategy}/parameter"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_prices(base: str, exchange: str, quote: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get price data (low, high, bid, ask, last and volumes) from a pair on an exchange"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/exchange/{exchange}/pair/{base}/{quote}/ticker"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_strategy(strategy: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns only the strategy indicated in path parameter"
    
    """
    url = f"https://criptobot-br.p.rapidapi.com/v1/strategy/{strategy}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "criptobot-br.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


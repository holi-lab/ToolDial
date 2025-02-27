import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_symbols_from_exchange(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbol list from exchange."
    
    """
    url = f"https://tradingview-ta-api-technical-analysis.p.rapidapi.com/get_symbols_from_exchange"
    querystring = {'exchange': exchange, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradingview-ta-api-technical-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_analysis_from_symbol(screener: str, symbol: str, interval: str, exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis data from symbol."
    
    """
    url = f"https://tradingview-ta-api-technical-analysis.p.rapidapi.com/get_analysis_from_symbol"
    querystring = {'screener': screener, 'symbol': symbol, 'interval': interval, 'exchange': exchange, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradingview-ta-api-technical-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchanges_from_screener(screener: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exchange list from screener."
    
    """
    url = f"https://tradingview-ta-api-technical-analysis.p.rapidapi.com/get_exchanges_from_screener"
    querystring = {'screener': screener, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradingview-ta-api-technical-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_screeners(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of screeners."
    
    """
    url = f"https://tradingview-ta-api-technical-analysis.p.rapidapi.com/get_screeners"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradingview-ta-api-technical-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_intervals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get valid intervals."
    
    """
    url = f"https://tradingview-ta-api-technical-analysis.p.rapidapi.com/get_intervals"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradingview-ta-api-technical-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


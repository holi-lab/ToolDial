import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def analysis(exchange: str, screener: str, interval: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the results of a technical analysis performed by TradingView for a specified symbol. Please note that the 'screener' parameter can be set to values such as 'america', 'indonesia', 'forex', or 'crypto' to filter the analysis based on the specific market."
    
    """
    url = f"https://ekoview.p.rapidapi.com/analysis"
    querystring = {'exchange': exchange, 'screener': screener, 'interval': interval, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ekoview.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchsymbol(exchange: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /search_symbol endpoint allows users to search for financial symbols on a specified exchange."
    
    """
    url = f"https://ekoview.p.rapidapi.com/search_symbol"
    querystring = {'exchange': exchange, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ekoview.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethist(ctype: str, interval: str, n_bars: str, extended_session: str, fut_contract: str, symbol: str, exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /get_hist endpoint retrieves historical data for a given symbol, including open, high, low, close, and volume values over a specified interval."
    
    """
    url = f"https://ekoview.p.rapidapi.com/get_hist"
    querystring = {'ctype': ctype, 'interval': interval, 'n_bars': n_bars, 'extended_session': extended_session, 'fut_contract': fut_contract, 'symbol': symbol, 'exchange': exchange, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ekoview.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


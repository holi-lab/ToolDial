import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daily(datestart: str, symbol: str, dateend: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Daily historical data for Stocks and ETFs."
    
    """
    url = f"https://apistocks.p.rapidapi.com/daily"
    querystring = {'dateStart': datestart, 'symbol': symbol, 'dateEnd': dateend, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apistocks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def monthly(datestart: str, dateend: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly historical data for stocks and ETFs."
    
    """
    url = f"https://apistocks.p.rapidapi.com/monthly"
    querystring = {'dateStart': datestart, 'dateEnd': dateend, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apistocks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly(symbol: str, datestart: str, dateend: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly historical data for stocks and ETFs."
    symbol: Upper case Stock or ETF symbol. 
        
    """
    url = f"https://apistocks.p.rapidapi.com/weekly"
    querystring = {'symbol': symbol, 'dateStart': datestart, 'dateEnd': dateend, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apistocks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def intraday(symbol: str, maxreturn: int=100, interval: str='5min', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get intraday price for stocks and ETFs."
    maxreturn: Maximum number of items(data points) to be returned. Default is 1000. Maximum value is 1000.
        interval: Optional parameter. Default value is \\\\\\\\\\\\\\\"5min\\\\\\\\\\\\\\\". Supported values: 1min, 5min, 15min, 30min, 60min.
        
    """
    url = f"https://apistocks.p.rapidapi.com/intraday"
    querystring = {'symbol': symbol, }
    if maxreturn:
        querystring['maxreturn'] = maxreturn
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apistocks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


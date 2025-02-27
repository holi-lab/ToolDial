import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_stock_or_forex_technical_data(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve most current technical data for any stock or forex pair.
		*All previous issued resolved*"
    
    """
    url = f"https://live-stock-and-forex-market-data.p.rapidapi.com/api/symbols/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-stock-and-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_forex_currencies_data(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve most current forex currencies data.
		select Forex pair type(example: major, minor, exotic, americas ect.)"
    
    """
    url = f"https://live-stock-and-forex-market-data.p.rapidapi.com/api/currencies/forex/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-stock-and-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_stock_market_data(type: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve most current stock market data.
		select country( example: usa, france, bahrain ect. ) and stock type(example: large capital stocks: large-cap, small capital stocks: small-cap, most active stocks: active"
    
    """
    url = f"https://live-stock-and-forex-market-data.p.rapidapi.com/api/market/{country}/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-stock-and-forex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


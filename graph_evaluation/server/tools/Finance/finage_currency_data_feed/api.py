import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_currency_last_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request any Currency Last Price"
    
    """
    url = f"https://finage-currency-data-feed.p.rapidapi.com/last?currencies=USDGBP&apikey=API_KEYUX0RG2F4BXCV5JJ50P0K3BLFQOS0Q0DP"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finage-currency-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_currencies_last_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert any two currencies  Last Price"
    
    """
    url = f"https://finage-currency-data-feed.p.rapidapi.com/convert?from=BTC&to=USD&amount=2&apikey=API_KEYUX0RG2F4BXCV5JJ50P0K3BLFQOS0Q0DP"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finage-currency-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_currency_historical_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request any Currency Historical Last Price"
    
    """
    url = f"https://finage-currency-data-feed.p.rapidapi.com/history?currency=EURUSD&from=2015-01-05&to=2015-01-07&apikey=API_KEYUX0RG2F4BXCV5JJ50P0K3BLFQOS0Q0DP"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finage-currency-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


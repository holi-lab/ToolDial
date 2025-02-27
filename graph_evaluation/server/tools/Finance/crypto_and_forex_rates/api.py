import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchangecurrencies(is_from: str, to: str, amount: int, bfee: int=1, sfee: int=1, withdraw: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exchange between two currency ( USD is base ) 
		User can define Buy and Sell Fees also can define withdraw fee in percentage"
    
    """
    url = f"https://crypto-and-forex-rates.p.rapidapi.com/api/exchange/{is_from}/{to}/{amount}/{bfee}/{sfee}/{withdraw}"
    querystring = {}
    if bfee:
        querystring['bfee'] = bfee
    if sfee:
        querystring['sfee'] = sfee
    if withdraw:
        querystring['withdraw'] = withdraw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-and-forex-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_assetinfo(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about single asset"
    
    """
    url = f"https://crypto-and-forex-rates.p.rapidapi.com/api/assetInfo/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-and-forex-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getprices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of prices for all assets"
    
    """
    url = f"https://crypto-and-forex-rates.p.rapidapi.com/api/getPrices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-and-forex-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return list of supported assets"
    
    """
    url = f"https://crypto-and-forex-rates.p.rapidapi.com/api/getAssets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-and-forex-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


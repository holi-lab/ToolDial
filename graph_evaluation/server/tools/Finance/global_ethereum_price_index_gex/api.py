import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ticker_per_symbol(market: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns ticker data for specified symbol"
    market: Possible values: global, local
        symbol: ETH<fiat>, where <fiat> is valid ISO currency (ex. ETHUSD, ETHEUR)
        
    """
    url = f"https://bitcoinaverage-global-ethereum-index-v1.p.rapidapi.com/indices/{market}/ticker/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-ethereum-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def short_ticker(market: str, crypto: str='ETH', fiats: str='USD,EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns basic ticker denoting last and daily average price for all symbols"
    market: Possible values: global, local
        crypto: Valid value: ETH
        fiats: If fiats parameter is included then only the values for those fiats will be returned (ETHUSD and ETHEUR in this example). If it's missing, then the response will contain ticker values of all available fiats for ETH.
        
    """
    url = f"https://bitcoinaverage-global-ethereum-index-v1.p.rapidapi.com/indices/{market}/ticker/short"
    querystring = {}
    if crypto:
        querystring['crypto'] = crypto
    if fiats:
        querystring['fiats'] = fiats
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-ethereum-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ticker_changes(market: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns ticker values and price changes for specified market and symbol."
    market: Possible values: global, local
        symbol: Possible values: ETH<fiat> where <fiat> is valid ISO currency (ex. ETHUSD)
        
    """
    url = f"https://bitcoinaverage-global-ethereum-index-v1.p.rapidapi.com/indices/{market}/ticker/{symbol}/changes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-ethereum-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


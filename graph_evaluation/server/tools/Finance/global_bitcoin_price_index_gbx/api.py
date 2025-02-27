import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ticker_per_symbol(market: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns ticker data for specified market symbol."
    market: Possible values: global, local
        symbol: BTC<fiat>, where <fiat> is valid ISO currency (ex. BTCUSD, BTCEUR)
        
    """
    url = f"https://bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com/indices/{market}/ticker/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ticker_data(market: str, crypto: str='BTC', fiat: str='USD,EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If no query parameters are sent, then returns ticker data for every supported symbol. If fiat(s) are sent as parameters, then only the ticker for those values is sent."
    market: Possible values: global, local
        crypto: valid value: BTC
        fiat: Comma separated list of ISO currency codes (ex. USD,EUR)
        
    """
    url = f"https://bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com/indices/{market}/ticker/all"
    querystring = {}
    if crypto:
        querystring['crypto'] = crypto
    if fiat:
        querystring['fiat'] = fiat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com"
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
        symbol: Possible values: BTC<fiat> where <fiat> is valid ISO currency (ex. BTCUSD)
        
    """
    url = f"https://bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com/indices/{market}/ticker/{symbol}/changes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def short_ticker(market: str, crypto: str='BTC', fiats: str='USD,EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns basic ticker denoting last and daily average price for all symbols"
    market: Possible values: global, local
        crypto: Valid value: BTC
        fiats: If fiats parameter is included then only the values for those fiats will be returned (BTCUSD and BTCEUR in this example). If it's missing, then the response will contain ticker values of all available fiats for BTC.
        
    """
    url = f"https://bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com/indices/{market}/ticker/short"
    querystring = {}
    if crypto:
        querystring['crypto'] = crypto
    if fiats:
        querystring['fiats'] = fiats
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_ticker(symbol: str, inex: str, exchanges: str='bitfinex,bitstamp', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used to generate a custom index in a certain currency. The “inex” path parameter represents “include” or “exclude”, you can choose to generate an index removing specified exchanges, or only including the few that you require."
    symbol: BTC<fiat> where <fiat> is valid ISO currency (ex. BTCUSD)
        inex: include - if you want the ticker to be calculated using only the exchanges sent as query parameter; exclude - if you want the price to be calculated using all exchanges, but the ones sent as query parameter.
        exchanges: Comma separated list of exchanges.
        
    """
    url = f"https://bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com/indices/ticker/custom/{inex}/{symbol}"
    querystring = {}
    if exchanges:
        querystring['exchanges'] = exchanges
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoinaverage-global-bitcoin-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


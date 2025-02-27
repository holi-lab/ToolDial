import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_1_public_symbol_orderbook(symbol: str, format_price: str=None, format_amount: str=None, format_amount_unit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    format_price: optional, "string" (default) or "number"
        format_amount: optional, "string" (default) or "number"
        format_amount_unit: optional, "currency" (default) or "lot"
        
    """
    url = f"https://community-hitbtc.p.rapidapi.com/api/1/public/{symbol}/orderbook"
    querystring = {}
    if format_price:
        querystring['format_price'] = format_price
    if format_amount:
        querystring['format_amount'] = format_amount
    if format_amount_unit:
        querystring['format_amount_unit'] = format_amount_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hitbtc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_1_public_symbol_trades(is_from: str, by: str, sort: str, start_index: str, max_results: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-hitbtc.p.rapidapi.com/api/1/public/{symbol}/trades"
    querystring = {'from': is_from, 'by': by, 'sort': sort, 'start_index': start_index, 'max_results': max_results, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hitbtc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_1_public_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-hitbtc.p.rapidapi.com/api/1/public/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hitbtc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_1_public_symbol_ticker(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24h means last 24h + last incomplete minute high - highest trade price / 24 h low - lowest trade price / 24 h volume - volume / 24h"
    
    """
    url = f"https://community-hitbtc.p.rapidapi.com/api/1/public/{symbol}/ticker"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hitbtc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_1_public_symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-hitbtc.p.rapidapi.com/api/1/public/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hitbtc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


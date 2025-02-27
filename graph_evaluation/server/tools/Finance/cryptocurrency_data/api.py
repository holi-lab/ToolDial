import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def metadata(symbol: str=None, base: str=None, market_venue: str=None, data_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of all available assets"
    symbol: Define to check the coverage for specific symbol
Example values: btc, eth
        base: Define to check the coverage for specific base
Example values: usd, usdt
        market_venue: Define to check the coverage for specific data type
Example values:  binance, ftx
        data_type: Define to check the coverage for specific exchange
Example values: metrics, trades
        
    """
    url = f"https://cryptocurrency-data2.p.rapidapi.com/metadata"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if base:
        querystring['base'] = base
    if market_venue:
        querystring['market_venue'] = market_venue
    if data_type:
        querystring['data_type'] = data_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-data2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def prices(provider: str, symbol: str='AR', symbols: str='BTC,USD,AR,ETH,BNB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Redstone HTTP API currently has a single yet very powerful endpoint, which allows you to fetch prices data."
    
    """
    url = f"https://redstone.p.rapidapi.com/prices"
    querystring = {'provider': provider, }
    if symbol:
        querystring['symbol'] = symbol
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redstone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


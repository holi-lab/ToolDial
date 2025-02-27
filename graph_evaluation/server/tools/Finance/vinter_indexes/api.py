import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vinter_single_assets_daily(symbol: str='btc-usd-p-d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vinter's daily single assets indexes"
    
    """
    url = f"https://vinter-indexes.p.rapidapi.com/"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vinter-indexes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vinter_multi_assets_daily(symbol: str='vntr-mc-5-d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vinter's daily multi assets indexes"
    
    """
    url = f"https://vinter-indexes.p.rapidapi.com/multi_assets_daily/"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vinter-indexes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


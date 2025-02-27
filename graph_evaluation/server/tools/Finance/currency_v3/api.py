import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historical_rates(get_2005_02_03: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find currency exchange rates for any day since 1999!"
    
    """
    url = f"https://mlatman-currency-v1.p.rapidapi.com/{get_2005_02_03}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlatman-currency-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_rates(base: str='USD', callback: str=None, symbols: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest rates for all supported countries."
    base: Set the base country to see differences between currencies.
        callback: Set a JSONP callback
        symbols: Filter exchange rates (ex: MXN,USD)
        
    """
    url = f"https://mlatman-currency-v1.p.rapidapi.com/latest"
    querystring = {}
    if base:
        querystring['base'] = base
    if callback:
        querystring['callback'] = callback
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlatman-currency-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ohlc(pair: str, limit: int, step: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a dictionary of ticker data for selected trading pair. Each tick in the dictionary is represented as a list of OHLC data."
    
    """
    url = f"https://crypto-ohlc.p.rapidapi.com/api/v2/ohlc/{pair}"
    querystring = {'limit': limit, 'step': step, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-ohlc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


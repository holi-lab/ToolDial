import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcryptocoins(accept: str, x_cmc_pro_api_key: str, convert: str, start: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get coins"
    
    """
    url = f"https://crypto39.p.rapidapi.com/v1/cryptocurrency/listings/new"
    querystring = {'Accept': accept, 'X-CMC_PRO_API_KEY': x_cmc_pro_api_key, 'convert': convert, 'start': start, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto39.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


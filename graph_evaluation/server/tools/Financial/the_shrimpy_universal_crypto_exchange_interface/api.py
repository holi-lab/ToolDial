import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_ticker(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves all Shrimpy supported exchange assets for a particular exchange along with pricing information. The pricing information is updated once per minute."
    exchange: The exchange for which you are collecting data
        
    """
    url = f"https://the-shrimpy-universal-crypto-exchange-interface.p.rapidapi.comhttps://dev-api.shrimpy.io/v1/exchanges/{exchange}/ticker"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-shrimpy-universal-crypto-exchange-interface.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


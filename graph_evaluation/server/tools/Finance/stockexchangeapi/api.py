import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def price_assetid(assetid: str='AAPL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get price quotes and details from an specific stock exchange asset"
    
    """
    url = f"https://stockexchangeapi.p.rapidapi.com/price/{assetid}"
    querystring = {}
    if assetid:
        querystring['assetID'] = assetid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockexchangeapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


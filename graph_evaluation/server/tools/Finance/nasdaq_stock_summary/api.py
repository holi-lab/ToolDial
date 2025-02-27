import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def summary(content_type: str, ticker: str, assetclass: str, host: str='api.nasdaq.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns summary of an instrument"
    
    """
    url = f"https://nasdaq-stock-summary.p.rapidapi.com/api/quote/{ticker}/summary"
    querystring = {'Content-Type': content_type, 'assetclass': assetclass, }
    if host:
        querystring['Host'] = host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nasdaq-stock-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


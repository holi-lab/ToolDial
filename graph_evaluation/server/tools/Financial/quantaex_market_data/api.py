import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_24_hours_tickers(quantaex_com: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24 hours trading data"
    
    """
    url = f"https://quantaex-market-data.p.rapidapi.com/api/v2/peatio/public/markets/tickers"
    querystring = {}
    if quantaex_com:
        querystring['quantaex.com'] = quantaex_com
    if quantaex_com:
        querystring['quantaex.com'] = quantaex_com
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quantaex-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


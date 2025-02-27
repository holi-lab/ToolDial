import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mostrecentshortvolume(ticker: str='TSLA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "MostRecentShortVolume"
    
    """
    url = f"https://short2.p.rapidapi.com/MostRecentShortVolume"
    querystring = {}
    if ticker:
        querystring['ticker'] = ticker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "short2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


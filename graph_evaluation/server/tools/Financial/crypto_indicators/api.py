import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crypto coins pair list. You can use this in this API."
    
    """
    url = f"https://crypto-indicators1.p.rapidapi.com/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-indicators1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rsi(pair: str='BTCUSDT', interval: str='60', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Relative Strength Index
		
		 Data refresh interval: 1 3 5 15 30 60 120 240 360 720 "D" "M" "W""
    
    """
    url = f"https://crypto-indicators1.p.rapidapi.com/rsi"
    querystring = {}
    if pair:
        querystring['pair'] = pair
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-indicators1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_coins_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get a collection of all coins.
		
		**API alterations**
		- **Get one coin:** https://api.minerstat.com/v2/coins?list=BTC
		- **Get list of coins:** https://api.minerstat.com/v2/coins?list=BTC,BCH,BSV
		- **Get all coins from one algorithm:** https://api.minerstat.com/v2/coins?algo=SHA-256
		- **Get all coins from multiple algorithms:** https://api.minerstat.com/v2/coins?algo=SHA-256,Scrypt,Ethash"
    
    """
    url = f"https://mineable-coins.p.rapidapi.com/coins"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mineable-coins.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_exchangerate(pair: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Exchange Rate API endpoint."
    pair: Currency pair to query. Must be in the form of **(currency1)_(currency2)** (e.g. **USD_EUR**)
        
    """
    url = f"https://exchange-rate-by-api-ninjas.p.rapidapi.com/v1/exchangerate"
    querystring = {'pair': pair, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rate-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


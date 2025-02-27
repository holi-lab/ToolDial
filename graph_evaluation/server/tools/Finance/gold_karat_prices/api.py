import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_gold_karat_prices_in_a_certain_currency(usd: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will show the live prices of the different gold karats where the prices are presented in the provided currency"
    
    """
    url = f"https://gold-karat-prices.p.rapidapi.com/{currency}"
    querystring = {'USD': usd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-karat-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


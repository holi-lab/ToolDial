import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_price(to_fait: str, to_coin: str, amount: int, from_fait: str, from_coin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Multiple Symbols Price"
    
    """
    url = f"https://convert-coin-price.p.rapidapi.com/"
    querystring = {'to_fait': to_fait, 'to_coin': to_coin, 'amount': amount, 'from_fait': from_fait, 'from_coin': from_coin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-coin-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def price(usd_price: int, country: str, method: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows the requester to get the PPP of a specific price for a country."
    
    """
    url = f"https://purchasing-power-parity.p.rapidapi.com/price"
    querystring = {'usd_price': usd_price, 'country': country, }
    if method:
        querystring['method'] = method
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "purchasing-power-parity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


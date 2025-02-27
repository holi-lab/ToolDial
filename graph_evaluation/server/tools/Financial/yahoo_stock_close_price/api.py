import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stockprice(stockticker: str, stockprice: str='sbux', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the close price."
    
    """
    url = f"https://yahoo-stock-close-price.p.rapidapi.com/stockprice/{stockticker}"
    querystring = {}
    if stockprice:
        querystring['stockprice'] = stockprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-stock-close-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


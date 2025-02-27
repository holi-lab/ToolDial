import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_historical_price(item_url: str, period: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the historical price information of an Amazon product by item_url."
    item_url: Product detail page url
        period: How many days of data, the default is 360
        
    """
    url = f"https://amazon-historical-price.p.rapidapi.com/api/sc/amazon/historical_price"
    querystring = {'item_url': item_url, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-historical-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


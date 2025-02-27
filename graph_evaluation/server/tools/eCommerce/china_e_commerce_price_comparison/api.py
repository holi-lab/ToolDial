import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product_historical_price(item_id: int, period: str, plat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product historical price by 'item_id'"
    period: How many days of data, the default is 360
        plat: Which platform's product
        
    """
    url = f"https://china-e-commerce-price-comparison.p.rapidapi.com/api/sc/item/historical_price/v2"
    querystring = {'item_id': item_id, 'period': period, 'plat': plat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "china-e-commerce-price-comparison.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


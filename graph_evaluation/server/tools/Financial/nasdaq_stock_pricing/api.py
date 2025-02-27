import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_price(allsymbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Price of Stock.
		Prices are updated every 15 minutes."
    
    """
    url = f"https://nasdaq-stock-pricing.p.rapidapi.com/StockV2/GetPrice"
    querystring = {'allSymbols': allsymbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nasdaq-stock-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


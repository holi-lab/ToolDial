import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def matic_pivot_resistence_support(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If the given exchange is **"matic"**, the application will retrieve historical data for the **Polygon (MATIC)** cryptocurrency exchange using the yfinance library. It will then calculate the pivot point, three resistance levels, and three support levels based on this data. The calculated values, along with the current date, will be returned in a JSON response when accessing the corresponding route."
    
    """
    url = f"https://cryptocurrency-pivot-calculator1.p.rapidapi.com/matic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-pivot-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btc_pivot_resistence_support(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If the given exchange is **"btc"**, the application will retrieve historical data for the **Bitcoin (BTC)** cryptocurrency exchange using the yfinance library. It will then calculate the pivot point, three resistance levels, and three support levels based on this data. The calculated values, along with the current date, will be returned in a JSON response when accessing the corresponding route."
    
    """
    url = f"https://cryptocurrency-pivot-calculator1.p.rapidapi.com/btc"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-pivot-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


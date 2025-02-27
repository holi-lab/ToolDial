import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_inflation(type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Inflation API endpoint. Returns current monthly and annual inflation percentages."
    type: inflation indicator type. Can be either CPI (Consumer Price Index) or HICP (Harmonized Index of Consumer Prices). If not provided, the CPI will be used by default.
        
    """
    url = f"https://inflation-by-api-ninjas.p.rapidapi.com/v1/inflation"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "inflation-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


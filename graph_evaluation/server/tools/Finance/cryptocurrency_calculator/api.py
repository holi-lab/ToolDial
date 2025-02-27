import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def value_change(initial: int, purchase: int, final: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return the unit amount of the cryptocurrency that was purchased, it's change in value, the amount that it changed by, and the percentage difference."
    
    """
    url = f"https://cryptocurrency-calculator.p.rapidapi.com/value/{purchase}/{initial}/{final}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

